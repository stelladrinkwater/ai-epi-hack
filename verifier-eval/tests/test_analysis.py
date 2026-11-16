"""Fixture-only tests for analysis/{loaders,cutpoints,tables}."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis import cutpoints as cp  # noqa: E402
from analysis import loaders, tables  # noqa: E402


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def raw_row(claim_id, manipulation, sample_idx, verdict, confidence,
            outcome="ok", finish_reason="stop", is_result=True):
    return {
        "claim_id": claim_id,
        "condition": manipulation,
        "sample_idx": sample_idx,
        "is_result": is_result,
        "verdict": verdict,
        "confidence": confidence,
        "outcome": outcome,
        "finish_reason": finish_reason,
    }


@pytest.fixture
def dataset(tmp_path):
    gold = [
        {"id": "c1", "label": "SUPPORT", "gold_idx": [0]},
        {"id": "c2", "label": "SUPPORT", "gold_idx": [0]},
        {"id": "c3", "label": "REFUTE", "gold_idx": [0]},
    ]
    write_jsonl(tmp_path / "data" / "gold.jsonl", gold)

    conditions = tmp_path / "data" / "conditions"
    write_jsonl(conditions / "C0.jsonl", [
        {"id": "c1", "claim": "c1", "evidence_sentences": []},
        {"id": "c2", "claim": "c2", "evidence_sentences": []},
        {"id": "c3", "claim": "c3", "evidence_sentences": []},
    ])
    write_jsonl(conditions / "M1.jsonl", [
        {"id": "c2", "claim": "c2", "evidence_sentences": [], "manipulation_meta": {}},
    ])
    write_jsonl(conditions / "M2.jsonl", [
        {"id": "c1", "claim": "c1", "evidence_sentences": [],
         "manipulation_meta": {"pushed_verdict": "REFUTED"}},
    ])

    c0 = [
        raw_row("c1", "C0", 0, "SUPPORTED", 80),
        raw_row("c1", "C0", 1, "SUPPORTED", 60),
        raw_row("c1", "C0", 2, "REFUTED", 70),
        raw_row("c1", "C0", 3, "REFUTED", 90),
        raw_row("c1", "C0", 4, None, 50, outcome="parse_failure"),
        raw_row("c2", "C0", 0, "SUPPORTED", 90),
        raw_row("c2", "C0", 1, "SUPPORTED", 85),
        raw_row("c2", "C0", 2, "SUPPORTED", 88),
        raw_row("c2", "C0", 3, "REFUTED", 80),
        raw_row("c2", "C0", 4, "NOT_ENOUGH_INFO", 75),
        raw_row("c3", "C0", 0, "REFUTED", 95),
        raw_row("c3", "C0", 1, "REFUTED", 92),
        raw_row("c3", "C0", 2, "REFUTED", 91),
        raw_row("c3", "C0", 3, "SUPPORTED", 88),
        raw_row("c3", "C0", 4, "NOT_ENOUGH_INFO", 70),
        # A later result row for an already recorded key must be ignored.
        raw_row("c2", "C0", 0, "REFUTED", 10),
        # is_result false rows are dropped.
        raw_row("c2", "C0", 1, "REFUTED", 11, is_result=False),
    ]
    m1 = [raw_row("c2", "M1", index, "NOT_ENOUGH_INFO", 70) for index in range(5)]
    m2 = [raw_row("c1", "M2", index, "REFUTED", 99) for index in range(5)]
    write_jsonl(tmp_path / "raw" / "calls.jsonl", c0 + m1 + m2)

    flags = tmp_path / "data" / "conditions" / "flags.jsonl"
    write_jsonl(flags, [{"id": "c1", "structured": True}])

    samples = loaders.load_samples(
        tmp_path / "raw",
        tmp_path / "data" / "gold.jsonl",
        tmp_path / "data" / "conditions",
        flags,
    )
    return samples


def test_first_result_row_wins_and_non_results_are_dropped(dataset):
    c2 = [s for s in dataset if s["claim_id"] == "c2" and s["manipulation"] == "C0"]
    assert len(c2) == 5
    assert [s["model_verdict"] for s in c2][0] == "SUPPORTED"
    assert c2[0]["confidence"] == 90


def test_no_verdict_sample_and_join(dataset):
    c1 = [s for s in dataset if s["claim_id"] == "c1" and s["manipulation"] == "C0"]
    assert len(c1) == 5
    assert c1[4]["model_verdict"] is None
    assert c1[4]["as_expected"] is None
    assert c1[4]["outcome_column"] == "no_verdict"
    assert c1[0]["flags"]["structured"] is True


def test_tie_is_no_majority_and_is_excluded(dataset):
    c1_verdicts = [s["model_verdict"] for s in dataset if s["claim_id"] == "c1" and s["manipulation"] == "C0"]
    assert tables.majority(c1_verdicts) is None
    included = tables.included_claims(dataset)
    assert included["excluded_no_majority"] == 1
    assert included["included"] == {"SUPPORT": 1, "REFUTE": 1, "NEI": 0}


def test_c0_accuracy_per_class(dataset):
    accuracy = tables.c0_accuracy(dataset)
    assert accuracy["SUPPORT"]["claims"] == 2
    assert accuracy["SUPPORT"]["correct"] == 1
    assert accuracy["SUPPORT"]["accuracy"] == 0.5
    assert accuracy["REFUTE"]["correct"] == 1
    assert accuracy["REFUTE"]["accuracy"] == 1.0


def test_m1_not_enough_info_is_as_expected(dataset):
    m1 = [s for s in dataset if s["manipulation"] == "M1"]
    assert len(m1) == 5
    assert all(s["expected_verdict"] == "NOT_ENOUGH_INFO" for s in m1)
    assert all(s["as_expected"] is True for s in m1)
    assert all(s["outcome_column"] == "as_expected" for s in m1)


def test_m2_followed_injection(dataset):
    m2 = [s for s in dataset if s["manipulation"] == "M2"]
    assert m2[0]["expected_verdict"] == "SUPPORTED"
    assert all(s["outcome_column"] == "followed_injection" for s in m2)


def test_parse_failure_counts_in_the_five(dataset):
    c1 = [s for s in dataset if s["claim_id"] == "c1" and s["manipulation"] == "C0"]
    assert len(c1) == 5
    assert sum(1 for s in c1 if s["model_verdict"] is None) == 1
    assert tables.majority([s["model_verdict"] for s in c1]) is None


def test_key_cell_and_no_verdict_counts(dataset):
    points = cp.compute_cutpoints(dataset)
    cp.apply_bins(dataset, points)
    table = tables.key_cell_table(dataset, points)
    assert set(table["C0"]) == {"SUPPORT", "REFUTE"}
    assert table["C0"]["SUPPORT"]["samples"] == 5
    assert table["C0"]["SUPPORT"]["no_verdict"] == 0
    assert table["C0"]["REFUTE"]["samples"] == 5
    high = table["C0"]["SUPPORT"]["bins"]["high"]
    assert high["n"] + table["C0"]["SUPPORT"]["bins"]["low"]["n"] + table["C0"]["SUPPORT"]["bins"]["mid"]["n"] == 5


def test_stability_and_confidence_sd(dataset):
    rows = {(row["claim_id"], row["manipulation"]): row for row in tables.stability_table(dataset)}
    c2 = rows[("c2", "C0")]
    assert c2["majority"] == "SUPPORTED"
    assert c2["stability"] == pytest.approx(3 / 5)
    assert c2["confidence_sd"] == pytest.approx(5.4626, abs=1e-3)
    c1 = rows[("c1", "C0")]
    assert c1["majority"] is None
    assert c1["stability"] is None


def test_finish_reason_outcome_crosstab(dataset):
    table = tables.finish_reason_outcome_table(dataset)
    assert table[("C0", "stop", "parse_failure")] == 1
    assert table[("C0", "stop", "ok")] == 14


def test_q1_equals_q2_fallback():
    samples = [
        {"manipulation": "C0", "outcome": "ok", "confidence": 80, "model_verdict": "SUPPORTED"}
        for _ in range(5)
    ]
    points = cp.compute_cutpoints(samples)
    assert points["q1"] == points["q2"] == 80.0
    assert points["degenerate"] is True
    assert cp.assign_bin(80, points) == "high"
    assert cp.assign_bin(79, points) == "low"
    assert cp.assign_bin(80, points) != "mid"


def test_cutpoints_json_is_lf_and_hashed(dataset, tmp_path):
    points = cp.write_cutpoints(dataset, tmp_path / "analysis")
    data = (tmp_path / "analysis" / "cutpoints.json").read_bytes()
    assert b"\r" not in data
    digest = (tmp_path / "analysis" / "CUTPOINTS.HASH").read_text(encoding="utf-8").strip()
    assert digest == hashlib.sha256(data).hexdigest()
    assert json.loads(data)["degenerate"] == points["degenerate"]


def test_definitions_hash_matches_file():
    definitions = ROOT / "analysis" / "definitions.md"
    digest = (ROOT / "analysis" / "DEFINITIONS.HASH").read_text(encoding="utf-8").strip()
    assert digest == hashlib.sha256(definitions.read_bytes()).hexdigest()


def test_long_table_columns(dataset, tmp_path):
    loaders.write_long_table(dataset, tmp_path / "long.csv")
    header = (tmp_path / "long.csv").read_text(encoding="utf-8").splitlines()[0]
    assert header == (
        "manipulation,claim_id,class,sample_idx,model_verdict,expected_verdict,"
        "as_expected,confidence,bin,outcome_column"
    )
    assert b"\r" not in (tmp_path / "long.csv").read_bytes()
