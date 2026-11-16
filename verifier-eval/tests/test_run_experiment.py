"""Tests for scripts/run_experiment.py (no real API calls)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import run_experiment as rx
import verifier_client as vc
from backend import AuthError, BackendResult, BackendTimeout
from prompts import load_prompts

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config.yaml"
VALID = '{"verdict": "SUPPORTED", "confidence": 80, "rationale": "ok", "evidence_concerns": []}'


class FakeBackend:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0

    def complete(self, system, user, **kwargs):
        self.calls += 1
        if not self.replies:
            raise AssertionError("fake backend ran out of replies")
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return BackendResult(
            text=reply, input_tokens=10, output_tokens=5, latency_ms=1.0,
            finish_reason="stop", http_status=200, model="fake",
        )


def model_config(max_retries: int = 0):
    return vc.ModelConfig.from_dict(
        {
            "base_url": "https://api.example.test",
            "model": "fake",
            "api_key_env": "DEEPSEEK_API_KEY",
            "temperature": 0.7,
            "top_p": None,
            "max_tokens": 64,
            "thinking": {"enabled": False, "budget_tokens": None},
            "concurrency": 1,
            "max_retries": max_retries,
            "retry_on": ["api_error"],
            "price_table": None,
        }
    )


def make_client(tmp_path, backend, *, max_retries=0):
    return vc.VerifierClient(
        model_config(max_retries),
        backend,
        load_prompts(ROOT / "prompts"),
        tmp_path / "calls.jsonl",
        cache_dir=tmp_path / "cache",
    )


def small_units(count=4, condition="C0"):
    return [
        {
            "claim_id": f"c_{index}",
            "claim": f"Claim {index}.",
            "evidence_sentences": [{"idx": 0, "text": f"Evidence {index}."}],
            "sample_idx": 0,
            "condition": condition,
        }
        for index in range(count)
    ]


# --------------------------------------------------------------------------- #
# Planning: counts, shards, ordering, naive restriction
# --------------------------------------------------------------------------- #
def test_dry_run_counts():
    """A dry run states the size of the tier and the size of the slice it would run.

    total_calls is the whole tier: 3500 core calls, 1250 extended, 1500 naive.
    shard_calls is what this invocation would execute - the whole tier while no
    order_seed defines an order, one shard of it once a seed does. Neither number
    depends on the seed's value or on the naive prompt file being present; test_r20
    in tests/test_review1_findings.py pins that independence directly.
    """
    config = rx.load_config_for(CONFIG)
    shards = config["run"]["shards"]
    sharded = config["run"]["order_seed"] is not None

    for tier, tier_calls in (("core", 3500), ("extended", 1250), ("naive", 1500)):
        plan = rx.dry_run(ROOT, CONFIG, tier, f"1/{shards}")
        assert plan["shard"] == f"1/{shards}", tier
        assert plan["total_calls"] == tier_calls, tier
        assert plan["shard_calls"] == (tier_calls // shards if sharded else tier_calls), tier


def test_shards_partition_the_order():
    config = rx.load_config_for(CONFIG)
    units = rx.build_units(ROOT, config, "core")
    ordered = rx.ordered_units(units, 7)
    pieces = [rx.shard_slice(ordered, index, 10) for index in range(1, 11)]
    assert sum(len(piece) for piece in pieces) == len(ordered)
    assert all(len(piece) == 350 for piece in pieces)
    flattened = [unit for piece in pieces for unit in piece]
    assert flattened == ordered
    assert len({(u["claim_id"], u["condition"], u["sample_idx"]) for u in flattened}) == len(flattened)


def test_order_seed_null_is_an_error_for_ordering():
    with pytest.raises(rx.ConfigError, match="order_seed"):
        rx.ordered_units([], None)


def test_naive_refuses_other_conditions():
    config = rx.load_config_for(CONFIG)
    assert rx.resolve_conditions(config, "naive") == ["C0", "M2"]
    with pytest.raises(rx.ConfigError, match="C0 and M2"):
        rx.resolve_conditions(config, "naive", ["C0", "C1"])


def test_c0p_units_reuse_c0_rows_with_c0p_condition():
    config = rx.load_config_for(CONFIG)
    units = [u for u in rx.build_units(ROOT, config, "core") if u["condition"] == "C0p"]
    assert len(units) == 750
    c0_ids = {unit["claim_id"] for unit in units}
    rows = {row["id"] for row in __import__("common").read_jsonl(ROOT / "data/conditions/C0.jsonl")}
    assert c0_ids == rows


# --------------------------------------------------------------------------- #
# Execution with a fake backend
# --------------------------------------------------------------------------- #
def test_execute_resumes_with_zero_calls(tmp_path):
    backend = FakeBackend([VALID, VALID])
    client = make_client(tmp_path, backend)
    units = small_units(2)
    first = rx.execute(client, units, order_seed=3, resume_passes=0)
    assert len(first) == 2 and backend.calls == 2
    second = rx.execute(client, units, order_seed=3, resume_passes=0)
    assert len(second) == 2
    assert backend.calls == 2, "resume made an API call"


def test_c0p_rows_carry_condition_c0p(tmp_path):
    config = rx.load_config_for(CONFIG)
    units = [u for u in rx.build_units(ROOT, config, "core") if u["condition"] == "C0p"][:3]
    backend = FakeBackend([VALID] * 3)
    rows = rx.execute(make_client(tmp_path, backend), units, order_seed=1, resume_passes=0)
    assert [row["condition"] for row in rows] == ["C0p"] * 3


def test_auth_abort_stops_the_run(tmp_path):
    backend = FakeBackend([AuthError(403, "forbidden")])
    with pytest.raises(vc.RunAborted):
        rx.execute(make_client(tmp_path, backend), small_units(2), order_seed=1, resume_passes=0)


def test_api_error_window_aborts(tmp_path):
    replies = [BackendTimeout("t")] * 21 + [VALID] * 79
    backend = FakeBackend(replies)
    units = small_units(100)
    with pytest.raises(vc.RunAborted, match="20 api_error"):
        rx.execute(
            make_client(tmp_path, backend), units,
            order_seed=1, resume_passes=0, progress_every=100,
        )


# --------------------------------------------------------------------------- #
# Guards
# --------------------------------------------------------------------------- #
def _patch_guards_ok(monkeypatch):
    monkeypatch.setattr(rx, "git_tree_clean", lambda root: True)
    monkeypatch.setattr(rx, "tag_is_ancestor", lambda root, tag: True)
    monkeypatch.setattr(rx, "spec_hash_matches", lambda root: True)
    monkeypatch.setattr(rx, "hash_files_match", lambda root: True)
    monkeypatch.setattr(rx, "prompt_hashes_match", lambda root: True)


def test_guard_git_refuses(monkeypatch):
    _patch_guards_ok(monkeypatch)
    monkeypatch.setattr(rx, "git_tree_clean", lambda root: False)
    with pytest.raises(rx.GuardError, match="git tree"):
        rx.assert_guards(ROOT, rx.load_config_for(CONFIG))


def test_guard_tag_refuses(monkeypatch):
    _patch_guards_ok(monkeypatch)
    monkeypatch.setattr(rx, "tag_is_ancestor", lambda root, tag: False)
    with pytest.raises(rx.GuardError, match="tag"):
        rx.assert_guards(ROOT, rx.load_config_for(CONFIG))


def test_guard_spec_hash_refuses(monkeypatch):
    _patch_guards_ok(monkeypatch)
    monkeypatch.setattr(rx, "spec_hash_matches", lambda root: False)
    with pytest.raises(rx.GuardError, match="SPEC.md hash"):
        rx.assert_guards(ROOT, rx.load_config_for(CONFIG))


def test_guard_data_hashes_refuse(monkeypatch):
    _patch_guards_ok(monkeypatch)
    monkeypatch.setattr(rx, "hash_files_match", lambda root: False)
    with pytest.raises(rx.GuardError, match="HASHES"):
        rx.assert_guards(ROOT, rx.load_config_for(CONFIG))


def test_guard_prompt_hashes_refuse(monkeypatch):
    _patch_guards_ok(monkeypatch)
    monkeypatch.setattr(rx, "prompt_hashes_match", lambda root: False)
    with pytest.raises(rx.GuardError, match="prompt hashes"):
        rx.assert_guards(ROOT, rx.load_config_for(CONFIG))


def test_real_guards_that_can_pass_do_pass():
    assert rx.spec_hash_matches(ROOT)
    assert rx.hash_files_match(ROOT)
    assert rx.prompt_hashes_match(ROOT)


def test_hash_reader_strips_cr_on_the_real_crlf_file():
    rows = rx.read_hash_file(ROOT / "data" / "HASHES.txt")
    assert rows
    assert all("\r" not in relative for _, relative in rows)
    assert rx.hash_file_matches(ROOT, ROOT / "data" / "HASHES.txt")


# --------------------------------------------------------------------------- #
# Naive prompt bundle
# --------------------------------------------------------------------------- #
def test_naive_prompt_bundle_uses_the_naive_system_file(tmp_path):
    prompts = tmp_path / "prompts"
    prompts.mkdir()
    (prompts / "verifier_user.txt").write_text("<claim>\n{claim}\n</claim>\n\n<evidence>\n{evidence}\n</evidence>\n", encoding="utf-8")
    (prompts / "verifier_system_naive.txt").write_text("NAIVE SYSTEM", encoding="utf-8")
    bundle, missing = rx.prompt_bundle(tmp_path, "naive")
    assert missing is False
    assert bundle.system == "NAIVE SYSTEM"
