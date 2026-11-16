"""Tests for the M1 / C1 / C2 audit-sheet builder."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import pytest

import make_audit_sheet as mas
from common import read_jsonl

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config.yaml"
SHEETS = ("m1_audit.csv", "m1_audit.md", "c_filler_audit.csv", "c_filler_audit.md")
ALL_FILES = SHEETS + ("key.csv", "README.md")

RULE = (
    "strong = the remaining text alone would make a careful reader say SUPPORTED or REFUTED;\n"
    "weak = there is a hint but not enough to decide; none = nothing relevant.\n"
    "topically_related = yes if the inserted sentence concerns the same disease, gene, intervention or\n"
    "outcome as the claim. Code independently, then compare.\n"
)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    out = tmp_path_factory.mktemp("audit")
    summary = mas.build(CONFIG, out_dir=out)
    return out, summary


def read_csv(path: Path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def key_rows(out: Path):
    return read_csv(out / "key.csv")


# --------------------------------------------------------------------------- #
# Counts and determinism
# --------------------------------------------------------------------------- #
def test_counts(built):
    out, summary = built
    m1 = read_csv(out / "m1_audit.csv")
    c = read_csv(out / "c_filler_audit.csv")
    assert len(m1) == 41
    assert sum(1 for row in c if row["condition"] == "C1") == 12
    assert sum(1 for row in c if row["condition"] == "C2") == 12
    assert len(key_rows(out)) == 65
    assert summary["m1_flagged"] == 29
    assert summary["m1_extra"] == 12


def test_deterministic_rebuild(built, tmp_path):
    out, _ = built
    other = tmp_path / "audit2"
    mas.build(CONFIG, out_dir=other)
    for name in ALL_FILES:
        assert (other / name).read_bytes() == (out / name).read_bytes(), name


def test_lf_and_no_bom(built):
    out, _ = built
    for name in ALL_FILES:
        data = (out / name).read_bytes()
        assert b"\r" not in data, name
        assert not data.startswith(b"\xef\xbb\xbf"), name


# --------------------------------------------------------------------------- #
# Leakage: no id, label, gold text or flag in the sheets
# --------------------------------------------------------------------------- #
def test_no_opaque_ids_in_sheets(built):
    out, _ = built
    pattern = re.compile(r"c_[0-9a-f]{10}")
    for name in SHEETS:
        assert not pattern.search((out / name).read_text(encoding="utf-8")), name


def test_no_labels_in_sheets(built):
    out, _ = built
    pattern = re.compile(r"\b(SUPPORT|REFUTE|NEI|SUPPORTED|REFUTED|NOT_ENOUGH_INFO)\b")
    for name in SHEETS:
        assert not pattern.search((out / name).read_text(encoding="utf-8")), name


def test_no_flag_status_in_sheets(built):
    out, _ = built
    for name in SHEETS:
        assert "flagged" not in (out / name).read_text(encoding="utf-8").lower(), name


def test_removed_gold_sentences_never_appear(built):
    out, _ = built
    c0 = {row["id"]: row for row in read_jsonl(ROOT / "data" / "conditions" / "C0.jsonl")}
    gold = {row["id"]: row for row in read_jsonl(ROOT / "data" / "gold.jsonl")}
    sheet_text = "".join(
        (out / name).read_text(encoding="utf-8") for name in ("m1_audit.csv", "m1_audit.md")
    )
    for entry in key_rows(out):
        if entry["sheet"] != "m1_audit":
            continue
        removed = set(gold[entry["id"]]["gold_idx"])
        for sentence in c0[entry["id"]]["evidence_sentences"]:
            if sentence["idx"] in removed:
                assert sentence["text"].strip() not in sheet_text, entry["id"]


# --------------------------------------------------------------------------- #
# Content shape
# --------------------------------------------------------------------------- #
def test_m1_sheet_shows_exactly_the_remaining_evidence(built):
    out, _ = built
    m1 = {row["id"]: row for row in read_jsonl(ROOT / "data" / "conditions" / "M1.jsonl")}
    for entry in key_rows(out):
        if entry["sheet"] != "m1_audit":
            continue
        expected = mas.numbered(m1[entry["id"]]["evidence_sentences"])
        csv_row = read_csv(out / "m1_audit.csv")[int(entry["row"]) - 1]
        assert csv_row["evidence"] == expected
        assert csv_row["leftover_support"] == ""
        assert csv_row["notes"] == ""


def test_c_sheet_shows_only_inserted_sentences(built):
    out, _ = built
    sources = {
        "C1": {row["id"]: row for row in read_jsonl(ROOT / "data" / "conditions" / "C1.jsonl")},
        "C2": {row["id"]: row for row in read_jsonl(ROOT / "data" / "conditions" / "C2.jsonl")},
    }
    rows = read_csv(out / "c_filler_audit.csv")
    by_key = {(row["condition"], row["row"]): row for row in rows}
    for entry in key_rows(out):
        if entry["sheet"] != "c_filler_audit":
            continue
        source = sources[entry["condition"]][entry["id"]]
        inserted = [s for s in source["evidence_sentences"] if s.get("idx") is None]
        original = [s["text"] for s in source["evidence_sentences"] if s.get("idx") is not None]
        cell = by_key[(entry["condition"], entry["row"])]["inserted_sentences"]
        assert cell == mas.numbered(inserted)
        assert len(inserted) == (6 if entry["condition"] == "C1" else 1)
        for text in original:
            assert text.strip() not in cell
        assert by_key[(entry["condition"], entry["row"])]["topically_related"] == ""


def test_key_maps_rows_and_flags(built):
    out, _ = built
    screen = {row["id"]: row["flagged"] for row in read_jsonl(ROOT / "data" / "conditions" / "m1_overlap_screen.jsonl")}
    rows = key_rows(out)
    assert len({row["id"] for row in rows if row["sheet"] == "m1_audit"}) == 41
    for row in rows:
        if row["sheet"] == "m1_audit":
            assert row["condition"] == "M1"
            assert (row["flagged"] == "True") is screen[row["id"]]
        else:
            assert row["condition"] in ("C1", "C2")
            assert row["flagged"] == "na"


def test_readme_rule_is_verbatim(built):
    out, _ = built
    assert RULE in (out / "README.md").read_text(encoding="utf-8")
