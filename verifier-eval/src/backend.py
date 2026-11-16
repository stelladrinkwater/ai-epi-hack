"""DeepSeek backend for the verifier call layer.

Only the chat-completions endpoint is used. Errors are translated into the
small exception hierarchy below so the client can decide what is retryable.
Auth errors (401/402/403) are separate because they abort a run instead of
being retried.
"""

from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass
from typing import Any

import requests

_BEARER = re.compile(r"Bearer\s+\S+", re.IGNORECASE)


def redact(text: str) -> str:
    """Remove any bearer token from text that may reach a log or exception."""
    return _BEARER.sub("Bearer [REDACTED]", str(text))


class BackendError(Exception):
    """Base class for backend failures (non-retryable unless subclassed)."""

    def __init__(
        self, message: str, status: int | None = None, body: str | None = None
    ) -> None:
        super().__init__(message)
        self.status = status
        self.body = body


class AuthError(BackendError):
    """HTTP 401/402/403: the run must abort, not retry."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(redact(message), status)
        self.status = status


class RateLimitError(BackendError):
    """HTTP 429, optionally carrying Retry-After seconds."""

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(redact(message), 429)
        self.retry_after = retry_after


class ServerError(BackendError):
    """HTTP 5xx."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(redact(message), status)
        self.status = status


class BackendTimeout(BackendError):
    """The request exceeded the client timeout."""


@dataclass(frozen=True)
class BackendResult:
    text: Any
    input_tokens: int | None
    output_tokens: int | None
    latency_ms: float
    finish_reason: str | None = None
    http_status: int | None = None
    model: str | None = None
    system_fingerprint: str | None = None
    reasoning_tokens: int | None = None


@dataclass(frozen=True)
class ThinkingConfig:
    enabled: bool
    budget_tokens: int | None = None

    def payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "type": "enabled" if self.enabled else "disabled",
        }
        if self.enabled and self.budget_tokens is not None:
            payload["budget_tokens"] = self.budget_tokens
        return payload


def _retry_after(response: requests.Response) -> float | None:
    value = response.headers.get("Retry-After")
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return None


class DeepSeekBackend:
    """Thin HTTP client for https://api.deepseek.com/chat/completions."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = 120.0,
        session: requests.Session | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        # A backend is shared by the verifier worker pool. Keep a separate
        # requests.Session per worker thread; callers may still inject one
        # session explicitly (primarily for tests and custom integrations).
        self._provided_session = session
        self._thread_local = threading.local()

    def _session(self) -> requests.Session:
        if self._provided_session is not None:
            return self._provided_session
        session = getattr(self._thread_local, "session", None)
        if session is None:
            session = requests.Session()
            self._thread_local.session = session
        return session

    def complete(
        self,
        system: str,
        user: str,
        *,
        temperature: float,
        top_p: float | None = None,
        max_tokens: int | None = None,
        thinking: ThinkingConfig | None = None,
    ) -> BackendResult:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
        }
        if top_p is not None:
            payload["top_p"] = top_p
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if thinking is not None:
            payload["thinking"] = thinking.payload()

        started = time.monotonic()
        request_error: BackendError | None = None
        try:
            response = self._session().post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=self.timeout,
            )
        except requests.Timeout as exc:
            raise BackendTimeout(f"request timed out after {self.timeout}s") from exc
        except requests.RequestException as exc:
            # Raised outside the except block: requests echoes the header value
            # (and therefore the key) in its message, which must not survive in
            # the exception chain either.
            request_error = BackendError(redact(f"request failed: {exc}"))
        if request_error is not None:
            raise request_error
        latency_ms = (time.monotonic() - started) * 1000.0

        status = response.status_code
        if status in (401, 402, 403):
            raise AuthError(status, f"authentication failed {status}: {response.text[:500]}")
        if status == 429:
            raise RateLimitError(
                f"rate limited: {response.text[:500]}", retry_after=_retry_after(response)
            )
        if status >= 500:
            raise ServerError(status, f"server error {status}: {response.text[:500]}")
        if status >= 400:
            raise BackendError(
                redact(f"request rejected {status}: {response.text[:500]}"),
                status,
                response.text,
            )

        try:
            data = response.json()
            choice = data["choices"][0]
            text = choice["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise BackendError(redact(f"malformed API response: {exc}")) from exc

        usage = data.get("usage") or {}
        details = usage.get("completion_tokens_details") or {}
        return BackendResult(
            text=text,
            input_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
            latency_ms=latency_ms,
            finish_reason=choice.get("finish_reason") if isinstance(choice, dict) else None,
            http_status=status,
            model=data.get("model"),
            system_fingerprint=data.get("system_fingerprint"),
            reasoning_tokens=details.get("reasoning_tokens"),
        )
