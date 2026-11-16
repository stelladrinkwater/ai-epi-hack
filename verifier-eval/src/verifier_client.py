"""Verifier call layer (DeepSeek only).

Wires the frozen prompts, DeepSeek backend, strict parser, persistent cache,
configurable retry/abort policy, resume passes and JSONL logging together.
Everything that can vary between calls is recorded per log row so results stay
auditable.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "scripts") not in sys.path:  # pragma: no cover - import plumbing
    sys.path.insert(0, str(_ROOT / "scripts"))

from common import VERDICTS  # noqa: E402,F401  (imported for the stated vocabulary)
from backend import (  # noqa: E402
    AuthError,
    BackendError,
    BackendTimeout,
    RateLimitError,
    ServerError,
    ThinkingConfig,
    redact,
)
from escaping import escape_tags  # noqa: E402
from parsing import ParseError, is_fenced, parse_response  # noqa: E402
from prompts import PromptBundle, load_prompts, render_evidence, render_user  # noqa: E402

RETRYABLE_OUTCOMES = frozenset({"api_error", "parse_failure"})
BLOCKED_FINISH_REASONS = frozenset(
    {"content_filter", "insufficient_system_resource"}
)
MAX_BACKOFF_SECONDS = 60.0
RETRY_AFTER_MAX_SECONDS = 900.0
CACHE_REPLACE_ATTEMPTS = 3
LOG_APPEND_ATTEMPTS = 4
LOG_APPEND_BACKOFF_SECONDS = 0.05
RUN_SLEEP_BUDGET_SECONDS = 1800.0

_LOG_LOCKS: dict[str, "LogLock"] = {}
_LOG_LOCKS_GUARD = threading.Lock()


class LogLock:
    """An exclusive OS lock on one log, held for the process lifetime.

    A second *process* that opens the same log is refused before any API
    call. Clients in this process share the one handle: they are already
    serialized by the per-client append lock.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._handle = open(self.path, "a+b")
        self._handle.seek(0, os.SEEK_END)
        if self._handle.tell() == 0:
            self._handle.write(b"\0")
            self._handle.flush()
        self._handle.seek(0)
        try:
            if os.name == "nt":  # pragma: no cover - platform branch
                import msvcrt

                msvcrt.locking(self._handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:  # pragma: no cover - platform branch
                import fcntl

                fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self._handle.close()
            raise


def acquire_log_lock(log_path: Path) -> LogLock:
    """Take the process-wide lock for ``log_path`` or raise StateError."""
    key = os.path.normcase(str(Path(log_path).resolve()))
    with _LOG_LOCKS_GUARD:
        existing = _LOG_LOCKS.get(key)
        if existing is not None:
            return existing
        try:
            lock = LogLock(Path(key + ".lock"))
        except OSError as exc:
            raise StateError(
                f"another process holds the log {log_path}; refusing to write "
                "concurrent rows"
            ) from exc
        _LOG_LOCKS[key] = lock
        return lock


class ConfigError(ValueError):
    """Raised when model config is missing or still null."""


class RunAborted(Exception):
    """401/402/403: the whole run must stop, not retry."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(redact(message))
        self.status = status


class StateError(Exception):
    """The cache and the log disagree; refuse to run until that is resolved."""


def _non_null(value: Any, key: str) -> Any:
    if value is None:
        raise ConfigError(f"config key 'models.deepseek.{key}' is null; set it before running")
    return value


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _raw_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return repr(value)


def _redact_row(value: Any) -> Any:
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, list):
        return [_redact_row(item) for item in value]
    if isinstance(value, dict):
        return {key: _redact_row(item) for key, item in value.items()}
    return value


@dataclass(frozen=True)
class ModelConfig:
    base_url: str
    model: str
    api_key_env: str
    temperature: float
    top_p: float | None
    max_tokens: int | None
    thinking_enabled: bool
    thinking_budget_tokens: int | None
    concurrency: int
    max_retries: int
    retry_on: tuple[str, ...]
    blocked_body_markers: tuple[str, ...]
    price_table: dict[str, float] | None

    @property
    def config_hash(self) -> str:
        payload = json.dumps(self.output_config(), sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def output_config(self) -> dict[str, Any]:
        """The settings that can change model output (cache/log identity)."""
        return {
            "model": self.model,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "thinking": {
                "enabled": self.thinking_enabled,
                "budget_tokens": self.thinking_budget_tokens,
            },
        }

    @classmethod
    def from_dict(cls, section: dict[str, Any]) -> "ModelConfig":
        if not isinstance(section, dict):
            raise ConfigError("config key 'models.deepseek' must be a mapping")
        base_url = _non_null(section.get("base_url"), "base_url")
        model = _non_null(section.get("model"), "model")
        api_key_env = _non_null(section.get("api_key_env"), "api_key_env")
        temperature = _non_null(section.get("temperature"), "temperature")
        concurrency = _non_null(section.get("concurrency"), "concurrency")
        max_retries = _non_null(section.get("max_retries"), "max_retries")
        retry_on = _non_null(section.get("retry_on"), "retry_on")

        if not isinstance(base_url, str) or not base_url:
            raise ConfigError("config key 'models.deepseek.base_url' must be a non-empty string")
        if not isinstance(model, str) or not model:
            raise ConfigError("config key 'models.deepseek.model' must be a non-empty string")
        if not isinstance(api_key_env, str) or not api_key_env:
            raise ConfigError("config key 'models.deepseek.api_key_env' must be a non-empty string")
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
            raise ConfigError("config key 'models.deepseek.temperature' must be a number")
        if isinstance(concurrency, bool) or not isinstance(concurrency, int) or concurrency < 1:
            raise ConfigError("config key 'models.deepseek.concurrency' must be a positive integer")
        if isinstance(max_retries, bool) or not isinstance(max_retries, int) or max_retries < 0:
            raise ConfigError("config key 'models.deepseek.max_retries' must be a non-negative integer")

        if not isinstance(retry_on, list) or not retry_on:
            raise ConfigError("config key 'models.deepseek.retry_on' must be a non-empty list")
        for outcome in retry_on:
            if not isinstance(outcome, str):
                raise ConfigError("config key 'models.deepseek.retry_on' must contain strings")
        unknown = sorted(set(retry_on) - RETRYABLE_OUTCOMES)
        if unknown:
            raise ConfigError(
                "config key 'models.deepseek.retry_on' has unknown entries "
                f"{unknown}; allowed: {sorted(RETRYABLE_OUTCOMES)}"
            )
        retry_on = tuple(dict.fromkeys(retry_on))

        markers = section.get("blocked_body_markers") or []
        if not isinstance(markers, list) or not all(
            isinstance(marker, str) and marker for marker in markers
        ):
            raise ConfigError(
                "config key 'models.deepseek.blocked_body_markers' must be a list of non-empty strings"
            )
        blocked_body_markers = tuple(markers)

        thinking = section.get("thinking")
        if not isinstance(thinking, dict):
            raise ConfigError("config key 'models.deepseek.thinking' must be a mapping")
        if "enabled" not in thinking or thinking["enabled"] is None:
            raise ConfigError("config key 'models.deepseek.thinking.enabled' is null; set it before running")
        if not isinstance(thinking["enabled"], bool):
            raise ConfigError("config key 'models.deepseek.thinking.enabled' must be a boolean")
        budget = thinking.get("budget_tokens")
        if budget is not None and (
            isinstance(budget, bool) or not isinstance(budget, int) or budget < 1
        ):
            raise ConfigError("config key 'models.deepseek.thinking.budget_tokens' must be a positive integer or null")

        top_p = section.get("top_p")
        if top_p is not None and (
            isinstance(top_p, bool) or not isinstance(top_p, (int, float)) or not 0 <= top_p <= 1
        ):
            raise ConfigError("config key 'models.deepseek.top_p' must be a number in [0, 1] or null")
        max_tokens = section.get("max_tokens")
        if max_tokens is not None and (
            isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1
        ):
            raise ConfigError("config key 'models.deepseek.max_tokens' must be a positive integer or null")

        price_table = section.get("price_table")
        if price_table is not None:
            if not isinstance(price_table, dict) or set(price_table) != {
                "input_per_mtok",
                "output_per_mtok",
            }:
                raise ConfigError(
                    "config key 'models.deepseek.price_table' must be "
                    "{input_per_mtok, output_per_mtok} or null"
                )

        return cls(
            base_url=base_url,
            model=model,
            api_key_env=api_key_env,
            temperature=float(temperature),
            top_p=None if top_p is None else float(top_p),
            max_tokens=max_tokens,
            thinking_enabled=thinking["enabled"],
            thinking_budget_tokens=budget,
            concurrency=concurrency,
            max_retries=max_retries,
            retry_on=retry_on,
            blocked_body_markers=blocked_body_markers,
            price_table=price_table,
        )


def load_model_config(config_path: Path) -> ModelConfig:
    config_path = Path(config_path)
    if not config_path.is_file():
        raise ConfigError(f"config not found: {config_path}")
    with config_path.open(encoding="utf-8") as fh:
        config = yaml.safe_load(fh) or {}
    section = (config.get("models") or {}).get("deepseek")
    if section is None:
        raise ConfigError("config key 'models.deepseek' is missing")
    return ModelConfig.from_dict(section)


def load_cache_dir(config_path: Path) -> Path:
    """Resolve ``cache.dir`` against the directory containing the config file."""
    config_path = Path(config_path)
    if not config_path.is_file():
        raise ConfigError(f"config not found: {config_path}")
    with config_path.open(encoding="utf-8") as fh:
        config = yaml.safe_load(fh) or {}
    section = config.get("cache")
    if not isinstance(section, dict):
        raise ConfigError("config key 'cache' must be a mapping")
    directory = section.get("dir")
    if not isinstance(directory, str) or not directory:
        raise ConfigError("config key 'cache.dir' must be a non-empty string")
    path = Path(directory)
    return path if path.is_absolute() else config_path.resolve().parent / path


def api_key_from_env(config: ModelConfig) -> str:
    key = os.environ.get(config.api_key_env)
    if not key:
        raise ConfigError(f"environment variable {config.api_key_env} is not set")
    cleaned = key.strip()
    if not cleaned:
        raise ConfigError(
            f"environment variable {config.api_key_env} is blank after stripping whitespace"
        )
    if any(ord(char) < 32 or ord(char) > 126 for char in cleaned):
        raise ConfigError(
            f"environment variable {config.api_key_env} contains control or "
            "non-ASCII characters; refusing to use it"
        )
    return cleaned


def _backoff_seconds(attempt: int) -> float:
    """Exponential backoff after failure ``attempt`` (1-based), capped at 60 s."""
    return min(MAX_BACKOFF_SECONDS, 0.5 * (2 ** (attempt - 1)))


class VerifierClient:
    """One verifier call = prompt render, backend call(s), parse, cache, log."""

    def __init__(
        self,
        config: ModelConfig,
        backend: Any,
        prompts: PromptBundle,
        log_path: Path,
        *,
        cache_dir: Path | None = None,
        sleep: Any = time.sleep,
    ) -> None:
        self.config = config
        self.backend = backend
        self.prompts = prompts
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        # One writer per log across processes (SPEC v1.5 §7).
        self.log_lock = acquire_log_lock(self.log_path)
        self.cache_dir = None if cache_dir is None else Path(cache_dir)
        self.prompt_store = (
            None if self.cache_dir is None else self.cache_dir.parent / "prompts_rendered"
        )
        self._sleep = sleep
        self._sleep_total = 0.0
        self._cache: dict[str, dict[str, Any]] = {}
        self._state_lock = threading.RLock()
        self._log_lock = threading.Lock()
        self._key_locks: dict[str, threading.Lock] = {}
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._load_disk_cache()
        if self.prompt_store is not None:
            self.prompt_store.mkdir(parents=True, exist_ok=True)

    def _store_prompt(self, text: str) -> None:
        """Write a rendered prompt once, addressed by its sha256."""
        assert self.prompt_store is not None
        target = self.prompt_store / f"{_sha256(text)}.txt"
        if not target.exists():
            target.write_text(text, encoding="utf-8")

    # ------------------------------------------------------------------ #
    # Cache helpers
    # ------------------------------------------------------------------ #
    def _load_disk_cache(self) -> None:
        assert self.cache_dir is not None
        for path in sorted(self.cache_dir.glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise ConfigError(f"unreadable cache entry {path}: {exc}") from exc
            key = payload.pop("key", None)
            if not isinstance(key, str) or not key:
                raise ConfigError(f"cache entry {path} has no 'key'")
            self._cache[key] = payload

    def _disk_store(self, key: str, entry: dict[str, Any]) -> None:
        assert self.cache_dir is not None
        target = self.cache_dir / f"{key}.json"
        temporary = self.cache_dir / f".{key}.{uuid.uuid4().hex}.tmp"
        temporary.write_text(
            json.dumps({"key": key, **entry}, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        for attempt in range(1, CACHE_REPLACE_ATTEMPTS + 1):
            try:
                os.replace(temporary, target)
                return
            except PermissionError:
                if attempt == CACHE_REPLACE_ATTEMPTS:
                    raise
                time.sleep(0.05 * attempt)

    def _lock_for(self, key: str) -> threading.Lock:
        with self._state_lock:
            lock = self._key_locks.get(key)
            if lock is None:
                lock = threading.Lock()
                self._key_locks[key] = lock
            return lock

    def cache_key(self, condition: str | None, user_prompt: str, sample_idx: int) -> str:
        """sha256 over every input that can change a cached answer."""
        payload = {
            "condition": condition,
            "model": self.config.model,
            "base_url": self.config.base_url,
            "output_config": self.config.output_config(),
            "retry_on": list(self.config.retry_on),
            "max_retries": self.config.max_retries,
            "timeout": getattr(self.backend, "timeout", None),
            "blocked_body_markers": list(self.config.blocked_body_markers),
            "system_prompt_sha256": self.prompts.system_sha256,
            "user_template_sha256": self.prompts.user_sha256,
            "user_prompt_sha256": _sha256(user_prompt),
            "sample_idx": sample_idx,
        }
        return _sha256(json.dumps(payload, sort_keys=True))

    def _cost(self, input_tokens: int | None, output_tokens: int | None) -> float | None:
        table = self.config.price_table
        if table is None or input_tokens is None or output_tokens is None:
            return None
        return (
            input_tokens * table["input_per_mtok"]
            + output_tokens * table["output_per_mtok"]
        ) / 1_000_000

    def _sleep_for(self, delay: float) -> None:
        """Sleep, enforcing the per-run cumulative cap (SPEC v1.5 §7)."""
        if self._sleep_total + delay > RUN_SLEEP_BUDGET_SECONDS:
            raise RunAborted(
                0,
                f"cumulative retry sleep would exceed {RUN_SLEEP_BUDGET_SECONDS:.0f}s",
            )
        self._sleep_total += delay
        self._sleep(delay)

    def _log(self, row: dict[str, Any]) -> None:
        line = json.dumps(_redact_row(row), ensure_ascii=False)
        with self._log_lock:
            for attempt in range(1, LOG_APPEND_ATTEMPTS + 1):
                try:
                    with self.log_path.open("a", encoding="utf-8", newline="\n") as fh:
                        fh.write(line + "\n")
                    return
                except PermissionError:
                    # A sync client or scanner holds the log for a moment; the
                    # paid row must not be lost, so wait and append again.
                    if attempt == LOG_APPEND_ATTEMPTS:
                        raise
                    self._sleep(LOG_APPEND_BACKOFF_SECONDS * attempt)

    def _row(
        self,
        *,
        claim_id: str,
        condition: str | None,
        sample_idx: int,
        cache_key: str,
        user_prompt: str = "",
        user_prompt_sha256: str,
        escapes: list[dict[str, str]],
        outcome: str,
        raw_response: str | None,
        parsed: dict[str, Any] | None,
        result: Any = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        latency_ms: float | None = None,
        attempt_idx: int = 0,
        total_attempts: int = 1,
        first_attempt_outcome: str | None = None,
        used_attempt_idx: int | None = None,
        cached: bool = False,
        attempts: int = 0,
        is_result: bool = True,
        finish_reason: str | None = None,
        http_status: int | None = None,
        order_seed: int | None = None,
        resume_pass: int = 0,
        cache_write_failed: bool = False,
        error: str | None = None,
    ) -> dict[str, Any]:
        row = {
            "claim_id": claim_id,
            "condition": condition,
            "model": self.config.model,
            "sample_idx": sample_idx,
            "order_seed": order_seed,
            "resume_pass": resume_pass,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "cache_key": cache_key,
            "config_hash": self.config.config_hash,
            "system_prompt_sha256": self.prompts.system_sha256,
            "user_template_sha256": self.prompts.user_sha256,
            "user_prompt": user_prompt,
            "user_prompt_sha256": user_prompt_sha256,
            "outcome": outcome,
            "verdict": parsed["verdict"] if parsed else None,
            "confidence": parsed["confidence"] if parsed else None,
            "rationale": parsed["rationale"] if parsed else None,
            "evidence_concerns": parsed["evidence_concerns"] if parsed else None,
            "raw_response": raw_response,
            "fenced": is_fenced(raw_response),
            "finish_reason": finish_reason,
            "finish_reason_last": finish_reason,
            "http_status": http_status
            if http_status is not None
            else getattr(result, "http_status", None),
            "reasoning_tokens": getattr(result, "reasoning_tokens", None),
            "response_model": getattr(result, "model", None),
            "system_fingerprint": getattr(result, "system_fingerprint", None),
            "tokens": {"input": input_tokens, "output": output_tokens},
            "latency_ms": latency_ms,
            "attempt_idx": attempt_idx,
            "total_attempts": total_attempts,
            "first_attempt_outcome": first_attempt_outcome,
            "used_attempt_idx": used_attempt_idx,
            "is_result": is_result,
            "cached": cached,
            "cache_write_failed": cache_write_failed,
            "attempts": attempts,
            "escapes": escapes,
            "error": redact(error) if error else None,
        }
        cost = self._cost(input_tokens, output_tokens)
        if cost is not None:
            row["cost"] = cost
        return row

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def verify(
        self,
        *,
        claim_id: str,
        claim: str,
        evidence_sentences: list[dict[str, Any]],
        sample_idx: int = 0,
        condition: str | None = None,
        order_seed: int | None = None,
        resume_pass: int = 0,
        is_last_pass: bool = True,
        in_batch: bool = False,
    ) -> dict[str, Any]:
        """Run one verifier call (with retries/cache) and log the result."""
        claim_text, claim_tags = escape_tags(claim)
        evidence_raw = render_evidence(evidence_sentences)
        evidence_text, evidence_tags = escape_tags(evidence_raw)
        escapes = [
            {"claim_id": claim_id, "field": "claim", "tag": tag} for tag in claim_tags
        ]
        escapes += [
            {"claim_id": claim_id, "field": "evidence", "tag": tag}
            for tag in evidence_tags
        ]

        user_prompt = render_user(self.prompts.user_template, claim_text, evidence_text)
        user_prompt_sha256 = _sha256(user_prompt)
        if self.prompt_store is not None:
            self._store_prompt(self.prompts.system)
            self._store_prompt(user_prompt)
        key = self.cache_key(condition, user_prompt, sample_idx)

        lock = self._lock_for(key)
        with lock:
            with self._state_lock:
                cached_entry = self._cache.get(key)
            if cached_entry is not None:
                return self._replay(
                    cached_entry, claim_id=claim_id, condition=condition,
                    sample_idx=sample_idx, key=key,
                    user_prompt=user_prompt, user_prompt_sha256=user_prompt_sha256, escapes=escapes,
                    order_seed=order_seed, resume_pass=resume_pass,
                )

            attempts_limit = self.config.max_retries + 1
            attempt_idx = 0
            first_attempt_outcome: str | None = None
            retry_api = "api_error" in self.config.retry_on
            retry_parse = "parse_failure" in self.config.retry_on
            thinking = ThinkingConfig(
                enabled=self.config.thinking_enabled,
                budget_tokens=self.config.thinking_budget_tokens,
            )
            while True:
                call_error: Exception | None = None
                retryable_type = False
                try:
                    result = self.backend.complete(
                        self.prompts.system,
                        user_prompt,
                        temperature=self.config.temperature,
                        top_p=self.config.top_p,
                        max_tokens=self.config.max_tokens,
                        thinking=thinking,
                    )
                except AuthError as exc:
                    self._log(
                        self._row(
                            claim_id=claim_id, condition=condition,
                            sample_idx=sample_idx, cache_key=key,
                            user_prompt=user_prompt, user_prompt_sha256=user_prompt_sha256, escapes=escapes,
                            outcome="api_error", raw_response=None, parsed=None,
                            attempt_idx=attempt_idx, total_attempts=attempt_idx + 1,
                            first_attempt_outcome=first_attempt_outcome or "api_error",
                            used_attempt_idx=attempt_idx, attempts=attempt_idx,
                            is_result=False, order_seed=order_seed,
                            resume_pass=resume_pass, error=str(exc),
                            http_status=exc.status,
                        )
                    )
                    raise RunAborted(exc.status, str(exc)) from exc
                except (RateLimitError, ServerError, BackendTimeout) as exc:
                    call_error = exc
                    retryable_type = True
                except BackendError as exc:
                    call_error = exc

                if call_error is not None:
                    status = getattr(call_error, "status", None)
                    body = getattr(call_error, "body", None) or ""
                    if (
                        status is not None
                        and 400 <= status < 500
                        and not isinstance(
                            call_error, (RateLimitError, ServerError, BackendTimeout)
                        )
                    ):
                        marked = any(
                            marker.lower() in str(body).lower()
                            for marker in self.config.blocked_body_markers
                        )
                        row = self._row(
                            claim_id=claim_id, condition=condition,
                            sample_idx=sample_idx, cache_key=key,
                            user_prompt=user_prompt,
                            user_prompt_sha256=user_prompt_sha256, escapes=escapes,
                            outcome="blocked" if marked else "api_error",
                            raw_response=str(body) if marked else None,
                            parsed=None,
                            attempt_idx=attempt_idx,
                            total_attempts=attempt_idx + 1,
                            first_attempt_outcome=first_attempt_outcome or (
                                "blocked" if marked else "api_error"
                            ),
                            used_attempt_idx=attempt_idx, attempts=attempt_idx,
                            is_result=marked or (in_batch and is_last_pass),
                            finish_reason=None,
                            http_status=status, order_seed=order_seed,
                            resume_pass=resume_pass, error=str(call_error),
                        )
                        self._log(row)
                        return row
                    outcome = "api_error"
                    if first_attempt_outcome is None:
                        first_attempt_outcome = outcome
                    can_retry = (
                        retryable_type
                        and retry_api
                        and attempt_idx + 1 < attempts_limit
                    )
                    if can_retry:
                        self._log(
                            self._row(
                                claim_id=claim_id, condition=condition,
                                sample_idx=sample_idx, cache_key=key,
                                user_prompt=user_prompt, user_prompt_sha256=user_prompt_sha256, escapes=escapes,
                                outcome=outcome, raw_response=None, parsed=None,
                                attempt_idx=attempt_idx, total_attempts=attempt_idx + 1,
                                first_attempt_outcome=first_attempt_outcome,
                                used_attempt_idx=None, attempts=attempt_idx,
                                is_result=False, order_seed=order_seed,
                                resume_pass=resume_pass, error=str(call_error),
                                http_status=getattr(call_error, "status", None),
                            )
                        )
                        attempt_idx += 1
                        delay = (
                            min(RETRY_AFTER_MAX_SECONDS, call_error.retry_after)
                            if isinstance(call_error, RateLimitError)
                            and call_error.retry_after is not None
                            else _backoff_seconds(attempt_idx)
                        )
                        self._sleep_for(delay)
                        continue
                    row = self._row(
                        claim_id=claim_id, condition=condition,
                        sample_idx=sample_idx, cache_key=key,
                        user_prompt=user_prompt, user_prompt_sha256=user_prompt_sha256, escapes=escapes,
                        outcome=outcome, raw_response=None, parsed=None,
                        attempt_idx=attempt_idx, total_attempts=attempt_idx + 1,
                        first_attempt_outcome=first_attempt_outcome,
                        used_attempt_idx=attempt_idx, attempts=attempt_idx,
                        is_result=is_last_pass, order_seed=order_seed,
                        resume_pass=resume_pass, error=str(call_error),
                        http_status=getattr(call_error, "status", None),
                    )
                    self._log(row)
                    return row

                finish_reason = result.finish_reason
                if finish_reason == "insufficient_system_resource":
                    # DeepSeek documents this as a server-side interruption, so
                    # it is an API failure to retry, not a model reply.
                    if first_attempt_outcome is None:
                        first_attempt_outcome = "api_error"
                    can_retry = retry_api and attempt_idx + 1 < attempts_limit
                    if can_retry:
                        self._log(
                            self._row(
                                claim_id=claim_id, condition=condition,
                                sample_idx=sample_idx, cache_key=key,
                                user_prompt=user_prompt, user_prompt_sha256=user_prompt_sha256, escapes=escapes,
                                outcome="api_error", raw_response=None, parsed=None,
                                attempt_idx=attempt_idx,
                                total_attempts=attempt_idx + 1,
                                first_attempt_outcome=first_attempt_outcome,
                                used_attempt_idx=None, attempts=attempt_idx,
                                is_result=False, finish_reason=finish_reason,
                                http_status=503, order_seed=order_seed,
                                resume_pass=resume_pass,
                                error="insufficient_system_resource",
                            )
                        )
                        attempt_idx += 1
                        self._sleep_for(_backoff_seconds(attempt_idx))
                        continue
                    row = self._row(
                        claim_id=claim_id, condition=condition,
                        sample_idx=sample_idx, cache_key=key,
                        user_prompt=user_prompt, user_prompt_sha256=user_prompt_sha256, escapes=escapes,
                        outcome="api_error", raw_response=None, parsed=None,
                        attempt_idx=attempt_idx, total_attempts=attempt_idx + 1,
                        first_attempt_outcome=first_attempt_outcome,
                        used_attempt_idx=attempt_idx, attempts=attempt_idx,
                        is_result=is_last_pass, finish_reason=finish_reason,
                        http_status=503, order_seed=order_seed,
                        resume_pass=resume_pass,
                        error="insufficient_system_resource",
                    )
                    self._log(row)
                    return row

                if finish_reason != "stop":
                    outcome = (
                        "blocked"
                        if finish_reason in BLOCKED_FINISH_REASONS
                        else "truncated"
                    )
                    row = self._row(
                        claim_id=claim_id, condition=condition,
                        sample_idx=sample_idx, cache_key=key,
                        user_prompt=user_prompt, user_prompt_sha256=user_prompt_sha256, escapes=escapes,
                        outcome=outcome, raw_response=_raw_text(result.text),
                        parsed=None, result=result,
                        input_tokens=result.input_tokens,
                        output_tokens=result.output_tokens,
                        latency_ms=result.latency_ms,
                        attempt_idx=attempt_idx, total_attempts=attempt_idx + 1,
                        first_attempt_outcome=first_attempt_outcome or outcome,
                        used_attempt_idx=attempt_idx, attempts=attempt_idx,
                        is_result=True, finish_reason=finish_reason,
                        order_seed=order_seed, resume_pass=resume_pass,
                    )
                    self._log(row)
                    return row

                if not isinstance(result.text, str):
                    # No string content = the provider returned no reply. This is
                    # an API failure, not the model's non-compliance. Pass 0 never
                    # produces a result row; only the final resume pass does.
                    row = self._row(
                        claim_id=claim_id, condition=condition,
                        sample_idx=sample_idx, cache_key=key,
                        user_prompt=user_prompt,
                        user_prompt_sha256=user_prompt_sha256, escapes=escapes,
                        outcome="api_error",
                        raw_response=_raw_text(result.text), parsed=None,
                        result=result,
                        input_tokens=result.input_tokens,
                        output_tokens=result.output_tokens,
                        latency_ms=result.latency_ms,
                        attempt_idx=attempt_idx, total_attempts=attempt_idx + 1,
                        first_attempt_outcome=first_attempt_outcome or "api_error",
                        used_attempt_idx=attempt_idx, attempts=attempt_idx,
                        is_result=(in_batch and is_last_pass),
                        finish_reason=finish_reason,
                        order_seed=order_seed, resume_pass=resume_pass,
                        error="response carried no string content",
                    )
                    self._log(row)
                    return row

                try:
                    parsed = parse_response(result.text)
                except ParseError as exc:
                    outcome = "parse_failure"
                    if first_attempt_outcome is None:
                        first_attempt_outcome = outcome
                    can_retry = retry_parse and attempt_idx + 1 < attempts_limit
                    if can_retry:
                        self._log(
                            self._row(
                                claim_id=claim_id, condition=condition,
                                sample_idx=sample_idx, cache_key=key,
                                user_prompt=user_prompt, user_prompt_sha256=user_prompt_sha256, escapes=escapes,
                                outcome=outcome, raw_response=_raw_text(result.text),
                                parsed=None, result=result,
                                input_tokens=result.input_tokens,
                                output_tokens=result.output_tokens,
                                latency_ms=result.latency_ms,
                                attempt_idx=attempt_idx,
                                total_attempts=attempt_idx + 1,
                                first_attempt_outcome=first_attempt_outcome,
                                used_attempt_idx=None, attempts=attempt_idx,
                                is_result=False, finish_reason=finish_reason,
                                order_seed=order_seed, resume_pass=resume_pass,
                                error=str(exc),
                            )
                        )
                        attempt_idx += 1
                        continue
                    row = self._row(
                        claim_id=claim_id, condition=condition,
                        sample_idx=sample_idx, cache_key=key,
                        user_prompt=user_prompt, user_prompt_sha256=user_prompt_sha256, escapes=escapes,
                        outcome=outcome, raw_response=_raw_text(result.text),
                        parsed=None, result=result,
                        input_tokens=result.input_tokens,
                        output_tokens=result.output_tokens,
                        latency_ms=result.latency_ms,
                        attempt_idx=attempt_idx, total_attempts=attempt_idx + 1,
                        first_attempt_outcome=first_attempt_outcome,
                        used_attempt_idx=attempt_idx, attempts=attempt_idx,
                        is_result=True, finish_reason=finish_reason,
                        order_seed=order_seed, resume_pass=resume_pass,
                        error=str(exc),
                    )
                    self._log(row)
                    return row

                entry = {
                    "parsed": parsed,
                    "raw": _raw_text(result.text),
                    "input_tokens": result.input_tokens,
                    "output_tokens": result.output_tokens,
                    "latency_ms": result.latency_ms,
                    "finish_reason": finish_reason,
                    "http_status": result.http_status,
                    "reasoning_tokens": result.reasoning_tokens,
                    "response_model": result.model,
                    "system_fingerprint": result.system_fingerprint,
                    "attempt_idx": attempt_idx,
                    "total_attempts": attempt_idx + 1,
                    "first_attempt_outcome": first_attempt_outcome or "ok",
                    "attempts": attempt_idx,
                }
                with self._state_lock:
                    self._cache[key] = entry
                cache_write_failed = False
                if self.cache_dir is not None:
                    try:
                        self._disk_store(key, entry)
                    except OSError:
                        # The paid reply must still reach the log; the disk cache
                        # is an optimisation, not the record of truth.
                        cache_write_failed = True
                row = self._row(
                    claim_id=claim_id, condition=condition,
                    sample_idx=sample_idx, cache_key=key,
                    user_prompt=user_prompt, user_prompt_sha256=user_prompt_sha256, escapes=escapes,
                    outcome="ok", raw_response=_raw_text(result.text),
                    parsed=parsed, result=result,
                    input_tokens=result.input_tokens,
                    output_tokens=result.output_tokens,
                    latency_ms=result.latency_ms,
                    attempt_idx=attempt_idx, total_attempts=attempt_idx + 1,
                    first_attempt_outcome=first_attempt_outcome or "ok",
                    used_attempt_idx=attempt_idx, attempts=attempt_idx,
                    is_result=True, finish_reason=finish_reason,
                    order_seed=order_seed, resume_pass=resume_pass,
                    cache_write_failed=cache_write_failed,
                )
                self._log(row)
                return row

    def _replay(
        self,
        entry: dict[str, Any],
        *,
        claim_id: str,
        condition: str | None,
        sample_idx: int,
        key: str,
        user_prompt: str,
        user_prompt_sha256: str,
        escapes: list[dict[str, str]],
        order_seed: int | None,
        resume_pass: int,
    ) -> dict[str, Any]:
        raw = entry.get("raw")
        if self.prompt_store is not None:
            self._store_prompt(self.prompts.system)
            self._store_prompt(user_prompt)
        try:
            parsed = parse_response(raw)
            outcome = "ok"
            error = None
        except ParseError as exc:
            parsed = None
            outcome = "parse_failure"
            error = str(exc)
        row = self._row(
            claim_id=claim_id, condition=condition, sample_idx=sample_idx,
            cache_key=key, user_prompt=user_prompt, user_prompt_sha256=user_prompt_sha256, escapes=escapes,
            outcome=outcome, raw_response=raw, parsed=parsed,
            input_tokens=entry.get("input_tokens"),
            output_tokens=entry.get("output_tokens"),
            latency_ms=entry.get("latency_ms"),
            attempt_idx=entry.get("attempt_idx", 0),
            total_attempts=entry.get("total_attempts", 1),
            first_attempt_outcome=entry.get("first_attempt_outcome"),
            used_attempt_idx=entry.get("attempt_idx"),
            cached=True, attempts=entry.get("attempts", 0),
            is_result=False, finish_reason=entry.get("finish_reason"),
            order_seed=order_seed, resume_pass=resume_pass, error=error,
        )
        self._log(row)
        return row

    def _unit_cache_key(self, item: dict[str, Any]) -> str:
        claim_text, _ = escape_tags(str(item.get("claim", "")))
        evidence_raw = render_evidence(item.get("evidence_sentences") or [])
        evidence_text, _ = escape_tags(evidence_raw)
        user_prompt = render_user(self.prompts.user_template, claim_text, evidence_text)
        return self.cache_key(
            item.get("condition"), user_prompt, int(item.get("sample_idx", 0))
        )

    def run_batch(
        self,
        items: list[dict[str, Any]],
        order_seed: int,
        resume_passes: int = 2,
    ) -> list[dict[str, Any]]:
        """Run verify() calls in a seeded shuffle with crash-resume support.

        Units are matched to rows by cache key (never by position) and the
        returned list follows the input order.
        """
        if isinstance(order_seed, bool) or not isinstance(order_seed, int):
            raise TypeError("order_seed must be an int")
        if resume_passes < 0:
            raise ValueError("resume_passes must be >= 0")

        ordered = [dict(item) for item in items]
        ordered.sort(key=lambda item: (
            str(item.get("claim_id")),
            str(item.get("condition")),
            int(item.get("sample_idx", 0)),
        ))
        random.Random(order_seed).shuffle(ordered)

        done = self._finished_cache_keys()
        recorded = self._recorded_result_rows()
        self._assert_cache_matches_log(recorded)
        rows_by_key: dict[str, dict[str, Any]] = dict(recorded)
        pending = [item for item in ordered if self._unit_cache_key(item) not in done]

        for pass_index in range(resume_passes + 1):
            if pass_index == 0:
                batch = pending
            else:
                batch = [
                    item for item in pending
                    if rows_by_key[self._unit_cache_key(item)]["outcome"] == "api_error"
                ]
            if not batch:
                break
            final = pass_index == resume_passes
            # The abort counter is per pass and resets on any success.
            consecutive_4xx = 0
            for item in batch:
                row = self.verify(
                    **item, order_seed=order_seed, resume_pass=pass_index,
                    is_last_pass=final, in_batch=True,
                )
                rows_by_key[row["cache_key"]] = row
                status = row.get("http_status")
                if (
                    row["outcome"] == "api_error"
                    and isinstance(status, int)
                    and 400 <= status < 500
                ):
                    consecutive_4xx += 1
                else:
                    consecutive_4xx = 0
                if consecutive_4xx >= 3:
                    raise RunAborted(
                        int(status), "three consecutive unmarked 4xx responses"
                    )

        return [
            rows_by_key[key]
            for key in (self._unit_cache_key(item) for item in items)
            if key in rows_by_key
        ]

    def _assert_cache_matches_log(self, recorded: dict[str, dict[str, Any]]) -> None:
        """Every cached key must be accounted for by a result row in this log."""
        with self._state_lock:
            unaccounted = [key for key in self._cache if key not in recorded]
        if unaccounted:
            raise StateError(
                f"cache has {len(unaccounted)} key(s) with no result row in "
                f"{self.log_path}; refusing to run (first: {unaccounted[0]})"
            )

    def _recorded_result_rows(self) -> dict[str, dict[str, Any]]:
        """The first result row per cache key, read from the log (SPEC §7)."""
        recorded: dict[str, dict[str, Any]] = {}
        if not self.log_path.is_file():
            return recorded
        with self.log_path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                key = row.get("cache_key")
                if row.get("is_result") and key and key not in recorded:
                    recorded[str(key)] = row
        return recorded

    def _finished_cache_keys(self) -> set[str]:
        done: set[str] = set()
        if not self.log_path.is_file():
            return done
        with self.log_path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if row.get("is_result") and row.get("cache_key"):
                    done.add(str(row["cache_key"]))
        return done
