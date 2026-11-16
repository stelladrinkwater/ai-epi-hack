"""TASK B2 review findings 1-11.

Written before the fixes: each test targets one finding listed in the B2
brief. Only the tests for behaviour that already exists pass on the old code.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import random
import re
from pathlib import Path

import pytest

import verifier_client as vc
from backend import (
    AuthError,
    BackendError,
    BackendResult,
    BackendTimeout,
    DeepSeekBackend,
    RateLimitError,
    ServerError,
    ThinkingConfig,
)
from escaping import escape_tags
from parsing import ParseError, parse_response
from prompts import (
    PromptBundle,
    PromptError,
    load_prompts,
    render_evidence,
    render_user,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures"
VALID = '{"verdict": "SUPPORTED", "confidence": 80, "rationale": "ok", "evidence_concerns": []}'
VALID2 = '{"verdict": "REFUTED", "confidence": 40, "rationale": "no", "evidence_concerns": []}'


def result(text, finish="stop", **overrides):
    values = {
        "text": text,
        "input_tokens": 100,
        "output_tokens": 20,
        "latency_ms": 5.0,
        "finish_reason": finish,
        "http_status": 200,
        "model": "deepseek-test",
        "system_fingerprint": "fp-test",
        "reasoning_tokens": 3,
    }
    values.update(overrides)
    return BackendResult(**values)


class FakeBackend:
    def __init__(self, items):
        self.items = list(items)
        self.calls = []

    def complete(self, system, user, **kwargs):
        self.calls.append({"system": system, "user": user, **kwargs})
        if not self.items:
            raise AssertionError("fake backend ran out of responses")
        item = self.items.pop(0)
        if isinstance(item, Exception):
            raise item
        if isinstance(item, BackendResult):
            return item
        if isinstance(item, tuple):
            return result(item[0], item[1])
        return result(item)


def config_dict(**overrides):
    config = {
        "base_url": "https://api.example.test",
        "model": "deepseek-test",
        "api_key_env": "DEEPSEEK_API_KEY",
        "temperature": 0.7,
        "top_p": None,
        "max_tokens": 50,
        "thinking": {"enabled": False, "budget_tokens": None},
        "concurrency": 2,
        "max_retries": 4,
        "retry_on": ["api_error"],
        "price_table": None,
    }
    for key, value in overrides.items():
        if key == "thinking":
            config["thinking"] = {**config["thinking"], **value}
        else:
            config[key] = value
    return config


def make_client(tmp_path, responses, *, cache=True, prompts=None, **overrides):
    model = vc.ModelConfig.from_dict(config_dict(**overrides))
    backend = FakeBackend(responses)
    sleeps = []
    client = vc.VerifierClient(
        model,
        backend,
        prompts or load_prompts(FIXTURES),
        tmp_path / "calls.jsonl",
        cache_dir=(tmp_path / "cache") if cache else None,
        sleep=sleeps.append,
    )
    return client, backend, sleeps


def call(client, **overrides):
    kwargs = {
        "claim_id": "c_1",
        "claim": "The sky is blue.",
        "evidence_sentences": [{"idx": 0, "text": "The sky is blue."}],
        "sample_idx": 0,
        "condition": "C0",
    }
    kwargs.update(overrides)
    return client.verify(**kwargs)


def log_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def item(claim_id="c_1", sample_idx=0, condition="C0"):
    return {
        "claim_id": claim_id,
        "claim": f"Claim {claim_id}.",
        "evidence_sentences": [{"idx": 0, "text": f"Evidence {claim_id}."}],
        "sample_idx": sample_idx,
        "condition": condition,
    }


# --------------------------------------------------------------------------- #
# Finding 1: condition is part of the cache key
# --------------------------------------------------------------------------- #
def test_c0_and_c0p_are_two_calls_through_run_batch_concurrency_4(tmp_path):
    client, backend, _ = make_client(
        tmp_path, [VALID, VALID2], concurrency=4
    )
    rows = client.run_batch(
        [item(condition="C0"), item(condition="C0p")], order_seed=3, resume_passes=0
    )
    assert len(backend.calls) == 2
    assert len({row["cache_key"] for row in rows}) == 2


# --------------------------------------------------------------------------- #
# Finding 2: full cache-key composition
# --------------------------------------------------------------------------- #
def test_cache_key_changes_with_every_component(tmp_path):
    base, _, _ = make_client(tmp_path, [])
    key = base.cache_key("C0", "prompt", 0)

    assert base.cache_key("C0p", "prompt", 0) != key
    assert base.cache_key("C0", "prompt", 1) != key
    assert base.cache_key("C0", "other prompt", 0) != key

    variants = {
        "base_url": config_dict(base_url="https://other.example.test"),
        "retry_on": config_dict(retry_on=["api_error", "parse_failure"]),
        "temperature": config_dict(temperature=0.8),
        "max_tokens": config_dict(max_tokens=51),
    }
    keys = {}
    for name, values in variants.items():
        other_client = vc.VerifierClient(
            vc.ModelConfig.from_dict(values),
            base.backend,
            base.prompts,
            tmp_path / f"{name}.jsonl",
            cache_dir=None,
        )
        keys[name] = other_client.cache_key("C0", "prompt", 0)
    assert len(set(keys.values())) == len(keys)

    other_prompts = dataclasses.replace(base.prompts, system_sha256="a" * 64)
    other = vc.VerifierClient(
        base.config, base.backend, other_prompts, tmp_path / "other.jsonl",
        cache_dir=None,
    )
    assert other.cache_key("C0", "prompt", 0) != key

    user_prompts = dataclasses.replace(base.prompts, user_sha256="b" * 64)
    third = vc.VerifierClient(
        base.config, base.backend, user_prompts, tmp_path / "third.jsonl",
        cache_dir=None,
    )
    assert third.cache_key("C0", "prompt", 0) != key


# --------------------------------------------------------------------------- #
# Finding 3: retry/abort per failure type
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("status", [401, 402, 403])
def test_auth_status_aborts_without_retry(tmp_path, status):
    client, backend, _ = make_client(tmp_path, [AuthError(status, "Bearer sk-secret")])
    with pytest.raises(vc.RunAborted) as caught:
        call(client)
    assert "sk-secret" not in str(caught.value)
    assert len(backend.calls) == 1


def test_auth_abort_propagates_from_run_batch(tmp_path):
    client, _, _ = make_client(tmp_path, [AuthError(403, "forbidden")])
    with pytest.raises(vc.RunAborted):
        client.run_batch([item()], order_seed=1, resume_passes=0)


def test_five_attempts_with_capped_backoff(tmp_path):
    client, backend, sleeps = make_client(tmp_path, [BackendTimeout("t")] * 5)
    row = call(client)
    assert row["outcome"] == "api_error"
    assert row["is_result"] is True
    assert len(backend.calls) == 5
    assert sleeps == [0.5, 1.0, 2.0, 4.0]


def test_retry_after_header_is_honored(tmp_path):
    client, backend, sleeps = make_client(
        tmp_path, [RateLimitError("429", retry_after=7.0), VALID]
    )
    row = call(client)
    assert row["outcome"] == "ok"
    assert sleeps == [7.0]
    assert len(backend.calls) == 2


def test_non_retryable_backend_error_is_immediate(tmp_path):
    client, backend, sleeps = make_client(tmp_path, [BackendError("400 bad")])
    row = call(client)
    assert row["outcome"] == "api_error"
    assert len(backend.calls) == 1
    assert sleeps == []


# --------------------------------------------------------------------------- #
# Finding 6: finish_reason gates ok
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "finish, outcome",
    [
        ("content_filter", "blocked"),
        ("length", "truncated"),
    ],
)
def test_non_stop_finish_reasons_are_not_ok(tmp_path, finish, outcome):
    client, _, _ = make_client(tmp_path, [(VALID, finish)])
    row = call(client)
    assert row["outcome"] == outcome
    assert row["verdict"] is None
    assert row["raw_response"] == VALID
    assert row["is_result"] is True


def test_insufficient_system_resource_is_a_retryable_api_error(tmp_path):
    client, backend, _ = make_client(
        tmp_path,
        [('{"verdict": "SUPP', "insufficient_system_resource"), VALID],
    )
    row = call(client)
    assert row["outcome"] == "ok"
    assert row["first_attempt_outcome"] == "api_error"
    assert len(backend.calls) == 2


def test_stop_and_valid_json_is_ok(tmp_path):
    client, _, _ = make_client(tmp_path, [(VALID, "stop")])
    assert call(client)["outcome"] == "ok"


# --------------------------------------------------------------------------- #
# Finding 4: hostile payloads become parse_failure without losing rows
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bad", [None, ["not", "a", "string"]])
def test_non_string_content_is_an_api_error(tmp_path, bad):
    client, _, _ = make_client(tmp_path, [result(bad, "stop")])
    row = call(client)
    assert row["outcome"] == "api_error"
    assert row["verdict"] is None
    assert row["is_result"] is False
    assert len(log_rows(tmp_path / "calls.jsonl")) == 1


def test_deep_nesting_is_parse_failure(tmp_path):
    client, _, _ = make_client(tmp_path, ["[" * 5000])
    row = call(client)
    assert row["outcome"] == "parse_failure"
    assert len(log_rows(tmp_path / "calls.jsonl")) == 1


def test_5000_digit_confidence_is_parse_failure(tmp_path):
    raw = (
        '{"verdict": "SUPPORTED", "confidence": '
        + "9" * 5000
        + ', "rationale": "r", "evidence_concerns": []}'
    )
    client, _, _ = make_client(tmp_path, [raw])
    row = call(client)
    assert row["outcome"] == "parse_failure"
    assert len(log_rows(tmp_path / "calls.jsonl")) == 1


# --------------------------------------------------------------------------- #
# Finding 8: duplicate and extra keys rejected
# --------------------------------------------------------------------------- #
def test_duplicate_json_keys_rejected():
    raw = (
        '{"verdict": "SUPPORTED", "verdict": "REFUTED", "confidence": 10,'
        ' "rationale": "r", "evidence_concerns": []}'
    )
    with pytest.raises(ParseError, match="duplicate key"):
        parse_response(raw)


def test_extra_json_keys_rejected():
    raw = (
        '{"verdict": "SUPPORTED", "confidence": 10, "rationale": "r",'
        ' "evidence_concerns": [], "extra": 1}'
    )
    with pytest.raises(ParseError, match="unexpected field"):
        parse_response(raw)


# --------------------------------------------------------------------------- #
# Finding 7: CR never reaches a log line or exception text
# --------------------------------------------------------------------------- #
def test_cr_in_input_never_appears_raw_in_log(tmp_path):
    client, _, _ = make_client(tmp_path, [VALID])
    call(
        client,
        claim="Line one.\r\nLine two.",
        evidence_sentences=[{"idx": 0, "text": "Evi\r\ndence."}],
    )
    data = (tmp_path / "calls.jsonl").read_bytes()
    assert b"\r" not in data
    assert len(data.decode("utf-8").splitlines()) == 1
    row = log_rows(tmp_path / "calls.jsonl")[0]
    assert re.fullmatch(r"[0-9a-f]{64}", row["cache_key"])


# --------------------------------------------------------------------------- #
# Finding 5: crash + resume leaves exactly one result per key
# --------------------------------------------------------------------------- #
def test_crash_and_resume_yields_one_result_row_per_key(tmp_path):
    items = [item("c_1"), item("c_2")]
    first, backend, _ = make_client(
        tmp_path, [BackendTimeout("t")] * 12, max_retries=0
    )
    first.run_batch(items, order_seed=5, resume_passes=2)
    assert len(backend.calls) == 6

    resumed, backend2, _ = make_client(tmp_path, [], max_retries=0)
    resumed.run_batch(items, order_seed=5, resume_passes=2)
    assert backend2.calls == []

    rows = log_rows(tmp_path / "calls.jsonl")
    results = {}
    for row in rows:
        if row["is_result"]:
            key = (row["claim_id"], row["condition"], row["sample_idx"])
            results[key] = results.get(key, 0) + 1
    assert results == {("c_1", "C0", 0): 1, ("c_2", "C0", 0): 1}


# --------------------------------------------------------------------------- #
# Finding 11: single-pass placeholders and strict sentences
# --------------------------------------------------------------------------- #
def test_placeholder_inside_claim_is_not_substituted_twice():
    rendered = render_user("{claim}|{evidence}", "{evidence}", "E")
    assert rendered == "{evidence}|E"


@pytest.mark.parametrize("bad", ["   \n", 123, None])
def test_render_evidence_rejects_blank_or_non_string(bad):
    with pytest.raises(PromptError):
        render_evidence([{"idx": 0, "text": bad}])


def test_render_evidence_requires_text_key():
    with pytest.raises(PromptError):
        render_evidence([{"idx": 0}])


# --------------------------------------------------------------------------- #
# Finding 9: tag escaping variants
# --------------------------------------------------------------------------- #
def test_tag_escaping_is_case_and_whitespace_insensitive():
    text, tags = escape_tags("<CLAIM> a </ Evidence > b <claim\t>")
    assert "&lt;CLAIM&gt;" in text
    assert "&lt;/ Evidence &gt;" in text
    assert "&lt;claim\t&gt;" in text
    assert tags == ["<claim>", "</evidence>"]


# --------------------------------------------------------------------------- #
# Finding 10: order_seed, deterministic order, keys on rows
# --------------------------------------------------------------------------- #
def test_run_batch_requires_int_order_seed(tmp_path):
    client, _, _ = make_client(tmp_path, [VALID])
    with pytest.raises(TypeError):
        client.run_batch([item()], order_seed="7")


def test_run_batch_sorts_before_shuffle_and_rows_carry_keys(tmp_path):
    client, backend, _ = make_client(tmp_path, [VALID] * 4, concurrency=1)
    items = [item("c_4"), item("c_2"), item("c_3"), item("c_1")]
    execution = [entry["claim_id"] for entry in sorted(items, key=lambda e: e["claim_id"])]
    random.Random(9).shuffle(execution)
    rows = client.run_batch(items, order_seed=9, resume_passes=0)
    # Execution follows sorted-then-shuffled order; the returned list is input order.
    assert [call["user"].split("Claim ")[1].split(".")[0] for call in backend.calls] == execution
    assert [row["claim_id"] for row in rows] == ["c_4", "c_2", "c_3", "c_1"]
    assert all(row["cache_key"] for row in rows)
    assert all(row["order_seed"] == 9 for row in rows)


# --------------------------------------------------------------------------- #
# End-to-end payload + prompt hashes
# --------------------------------------------------------------------------- #
class Session:
    def __init__(self):
        self.payloads = []

    def post(self, url, headers, json, timeout):
        self.payloads.append(json)
        return Response()


class Response:
    status_code = 200

    def json(self):
        return {
            "choices": [{"message": {"content": VALID}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 4},
            "model": "deepseek-test",
            "system_fingerprint": "fp",
        }


def test_end_to_end_payload_from_verifier_client(tmp_path):
    session = Session()
    backend = DeepSeekBackend(
        "https://api.example.test", "sk-test", "deepseek-test", session=session
    )
    model = vc.ModelConfig.from_dict(config_dict())
    client = vc.VerifierClient(
        model, backend, load_prompts(FIXTURES), tmp_path / "calls.jsonl",
        cache_dir=None,
    )
    call(client)
    payload = session.payloads[0]
    assert payload["model"] == "deepseek-test"
    assert payload["temperature"] == 0.7
    assert payload["max_tokens"] == 50
    assert payload["thinking"] == {"type": "disabled"}
    assert payload["messages"][0]["role"] == "system"
    assert "The sky is blue." in payload["messages"][1]["content"]
    assert "<claim>" in payload["messages"][1]["content"]


def test_frozen_prompt_hashes_match_changelog():
    text = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    for name, path in (
        ("verifier_system.txt", ROOT / "prompts" / "verifier_system.txt"),
        ("verifier_user.txt", ROOT / "prompts" / "verifier_user.txt"),
    ):
        match = re.search(rf"`prompts/{name}`\s*\|\s*`([0-9a-f]{{64}})`", text)
        assert match, name
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == match.group(1)


# --------------------------------------------------------------------------- #
# Log append-only, cache replay, row metadata
# --------------------------------------------------------------------------- #
def test_existing_log_lines_are_never_truncated(tmp_path):
    log = tmp_path / "calls.jsonl"
    log.write_text("PRE-EXISTING\n", encoding="utf-8")
    client, _, _ = make_client(tmp_path, [VALID])
    call(client)
    lines = log.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "PRE-EXISTING"
    assert len(lines) == 2


def test_cache_replay_is_not_a_result_and_is_reparsed(tmp_path):
    client, _, _ = make_client(tmp_path, [VALID])
    first = call(client)
    second = call(client)
    assert first["cached"] is False
    assert second["cached"] is True
    assert second["is_result"] is False
    assert second["verdict"] == parse_response(second["raw_response"])["verdict"]


def test_rows_carry_observability_fields(tmp_path):
    client, _, _ = make_client(tmp_path, [VALID])
    row = call(client)
    for field in (
        "timestamp",
        "http_status",
        "reasoning_tokens",
        "response_model",
        "system_fingerprint",
        "user_template_sha256",
        "cache_key",
    ):
        assert field in row
    assert row["http_status"] == 200
    assert row["reasoning_tokens"] == 3
    assert row["response_model"] == "deepseek-test"


def test_bearer_tokens_are_redacted_in_error_text(tmp_path):
    client, _, _ = make_client(
        tmp_path, [BackendError("bad key Bearer sk-super-secret")]
    )
    row = call(client)
    assert "sk-super-secret" not in row["error"]
    assert "Bearer [REDACTED]" in row["error"]


def test_cache_directory_is_created(tmp_path):
    target = tmp_path / "nested" / "cache"
    client, _, _ = make_client(tmp_path, [VALID])
    custom = vc.VerifierClient(
        client.config, client.backend, client.prompts, tmp_path / "other.jsonl",
        cache_dir=target,
    )
    assert target.is_dir()
