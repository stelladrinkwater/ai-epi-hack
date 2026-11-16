"""Exploratory strata and descriptive checks (SPEC v1.4 §6/§8).

No p-values are produced here. Every table is descriptive; the Spearman result
is flagged degenerate when stability is 1.0 in more than 80% of cells.
"""

from __future__ import annotations

import csv
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from analysis import tables

M1_KEYS = ("structured", "lost_headers", "dangling_anaphora", "overlap_flag")
M2_KEYS = ("nearest_gold_distance", "states_confidence_number")
POINT_RE = re.compile(r'"verdict"\s*:\s*"')


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not Path(path).is_file():
        return []
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def audit_codes(audit_dir: Path | None) -> dict[str, str] | None:
    """id -> leftover_support, or None when audit/coded.csv is missing.

    None means the M1 audit stratum cannot be produced. Callers omit it and say
    so; an uncoded audit must never look like a coded stratum with no rows.
    """
    if audit_dir is None:
        return None
    coded_path = Path(audit_dir) / "coded.csv"
    if not coded_path.is_file():
        return None
    key_rows = _read_csv(Path(audit_dir) / "key.csv")
    coded_rows = _read_csv(coded_path)
    if not key_rows or not coded_rows:
        return None
    id_by_row = {
        row["row"]: row["id"]
        for row in key_rows
        if row.get("sheet") == "m1_audit"
    }
    codes: dict[str, str] = {}
    for row in coded_rows:
        row_number = row.get("row")
        if row_number in id_by_row:
            codes[id_by_row[row_number]] = row.get("leftover_support", "").strip()
    return codes


def _rate(group: list[dict[str, Any]]) -> dict[str, Any]:
    verdicts = [sample for sample in group if sample["model_verdict"] is not None]
    not_expected = sum(1 for sample in verdicts if sample["as_expected"] is False)
    return {
        "claims": len({sample["claim_id"] for sample in group}),
        "samples": len(group),
        "with_verdict": len(verdicts),
        "not_as_expected": not_expected,
        "rate": (not_expected / len(verdicts)) if verdicts else None,
        "no_verdict": len(group) - len(verdicts),
    }


def _strata_rows(
    samples: list[dict[str, Any]],
    descriptors: dict[str, str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for descriptor, source in descriptors.items():
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for sample in samples:
            value = source(sample)
            groups["unknown" if value is None or value == "" else str(value)].append(sample)
        for value in sorted(groups):
            stats = _rate(groups[value])
            rows.append({"descriptor": descriptor, "value": value, **stats})
    return rows


def m1_strata(samples: list[dict[str, Any]], audit_dir: Path | None = None) -> list[dict[str, Any]]:
    m1 = [sample for sample in samples if sample["manipulation"] == "M1"]
    codes = audit_codes(audit_dir)
    descriptors: dict[str, Any] = {
        key: (lambda sample, key=key: sample["flags"].get(key)) for key in M1_KEYS
    }
    if codes is not None:
        descriptors["audit_leftover_support"] = lambda sample: codes.get(sample["claim_id"])
    return _strata_rows(m1, descriptors)


def m2_strata(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    m2 = [sample for sample in samples if sample["manipulation"] == "M2"]

    def meta_value(sample: dict[str, Any], key: str) -> Any:
        meta = sample.get("manipulation_meta") or {}
        return meta.get(key, sample["flags"].get(key))

    descriptors = {
        "family": lambda sample: meta_value(sample, "family"),
        "position": lambda sample: meta_value(sample, "position"),
        **{key: (lambda sample, key=key: sample["flags"].get(key)) for key in M2_KEYS},
    }
    return _strata_rows(m2, descriptors)


def _stated_number(flags: dict[str, Any]) -> Any:
    """The template's stated confidence number (SPEC v1.6 §6).

    The descriptor file names it ``stated_number``; the earlier fixture name
    ``template_confidence_number`` is still accepted.
    """
    if "stated_number" in flags:
        return flags.get("stated_number")
    return flags.get("template_confidence_number")


def injection_confidence(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    followed = [sample for sample in samples if sample["outcome_column"] == "followed_injection"]
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in followed:
        stated = _stated_number(sample["flags"])
        groups["stated_number" if stated is not None else "no_number"].append(sample)
    rows: list[dict[str, Any]] = []
    for name in ("stated_number", "no_number"):
        group = groups.get(name, [])
        matched = 0
        for sample in group:
            stated = _stated_number(sample["flags"])
            if stated is not None and sample["confidence"] == stated:
                matched += 1
        rows.append(
            {
                "group": name,
                "followed": len(group),
                "matched_stated_confidence": matched,
                "share": (matched / len(group)) if group else None,
            }
        )
    return rows


def numberless_confidence(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Repeated confidence analysis on templates that state no number."""
    numberless = [
        sample for sample in samples
        if sample["manipulation"] == "M2"
        and not sample["flags"].get("states_confidence_number", False)
        and sample["confidence"] is not None
    ]
    groups: dict[tuple[str, bool], list[int]] = defaultdict(list)
    for sample in numberless:
        groups[(sample["outcome_column"], bool(sample["as_expected"]))].append(sample["confidence"])
    rows = []
    for (outcome_column, as_expected), values in sorted(groups.items()):
        rows.append(
            {
                "outcome_column": outcome_column,
                "as_expected": as_expected,
                "n": len(values),
                "mean_confidence": mean(values),
            }
        )
    return rows


def pushed_regex_counts(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: Counter[tuple[str, bool]] = Counter()
    for sample in samples:
        if sample["manipulation"] != "M2" or sample.get("outcome") == "ok":
            continue
        pushed = (sample.get("manipulation_meta") or {}).get("pushed_verdict")
        raw = sample.get("raw_response") or ""
        matched = bool(pushed) and bool(
            re.search(r'"verdict"\s*:\s*"' + re.escape(str(pushed)) + r'"', str(raw))
        )
        counts[(sample["manipulation"], matched)] += 1
    return [
        {"manipulation": manip, "matched_pushed_regex": matched, "n": count}
        for (manip, matched), count in sorted(counts.items())
    ]


def _ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(order):
        end = index
        while end + 1 < len(order) and values[order[end + 1]] == values[order[index]]:
            end += 1
        average = (index + end) / 2.0 + 1.0
        for position in range(index, end + 1):
            ranks[order[position]] = average
        index = end + 1
    return ranks


def spearman(x: list[float], y: list[float]) -> float | None:
    if len(x) < 3 or len(x) != len(y):
        return None
    rank_x, rank_y = _ranks(x), _ranks(y)
    mean_x, mean_y = mean(rank_x), mean(rank_y)
    numerator = sum((a - mean_x) * (b - mean_y) for a, b in zip(rank_x, rank_y))
    denominator = math.sqrt(
        sum((a - mean_x) ** 2 for a in rank_x) * sum((b - mean_y) ** 2 for b in rank_y)
    )
    return None if denominator == 0 else numerator / denominator


def stability_spearman(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    stability_rows = tables.stability_table(samples)
    group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in stability_rows:
        group[row["manipulation"]].append(row)
    mean_confidence: dict[tuple[str, str], float] = {}
    grouped: dict[tuple[str, str], list[int]] = defaultdict(list)
    for sample in samples:
        if sample["confidence"] is not None:
            grouped[(sample["claim_id"], sample["manipulation"])].append(sample["confidence"])
    for key, values in grouped.items():
        mean_confidence[key] = mean(values)

    rows = []
    # Pooled degeneracy over every cell, no-majority cells included (SPEC v1.6 §6):
    # when it exceeds 80% the correlation has no usable variation and is dropped.
    pooled_share = (
        sum(1 for row in stability_rows if row["stability"] == 1.0) / len(stability_rows)
        if stability_rows
        else None
    )
    pooled_degenerate = bool(pooled_share is not None and pooled_share > 0.8)
    for condition in sorted(group):
        # Degeneracy is per condition (SPEC v1.5 §6); a degenerate condition
        # has no usable stability variation, so its rho is dropped.
        condition_cells = [
            row for row in group[condition] if row["stability"] is not None
        ]
        degenerate_share = (
            sum(1 for row in condition_cells if row["stability"] == 1.0)
            / len(condition_cells)
            if condition_cells
            else None
        )
        degenerate = bool(degenerate_share is not None and degenerate_share > 0.8)
        values = [
            (mean_confidence[(row["claim_id"], condition)], row["stability"])
            for row in condition_cells
            if (row["claim_id"], condition) in mean_confidence
        ]
        correlation = (
            None
            if degenerate or pooled_degenerate
            else spearman([a for a, _ in values], [b for _, b in values])
            if values
            else None
        )
        rows.append(
            {
                "manipulation": condition,
                "n_cells": len(values),
                "spearman": correlation,
                "degenerate_stability_share": degenerate_share,
                "degenerate": degenerate,
                "pooled_degenerate_share": pooled_share,
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in columns})


def write_strata(
    samples: list[dict[str, Any]],
    out_dir: Path,
    *,
    audit_dir: Path | None = None,
) -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    m1 = m1_strata(samples, audit_dir)
    m2 = m2_strata(samples)
    injection = injection_confidence(samples)
    numberless = numberless_confidence(samples)
    pushed = pushed_regex_counts(samples)
    spearman_rows = stability_spearman(samples)

    write_csv(out / "strata_m1.csv", m1, ["descriptor", "value", "claims", "samples", "with_verdict", "not_as_expected", "rate", "no_verdict"])
    write_csv(out / "strata_m2.csv", m2, ["descriptor", "value", "claims", "samples", "with_verdict", "not_as_expected", "rate", "no_verdict"])
    write_csv(out / "injection_confidence.csv", injection, ["group", "followed", "matched_stated_confidence", "share"])
    write_csv(out / "numberless_confidence.csv", numberless, ["outcome_column", "as_expected", "n", "mean_confidence"])
    write_csv(out / "pushed_regex.csv", pushed, ["manipulation", "matched_pushed_regex", "n"])
    write_csv(out / "spearman.csv", spearman_rows, ["manipulation", "n_cells", "spearman", "degenerate_stability_share", "degenerate"])
    return {
        "m1": m1, "m2": m2, "injection": injection, "numberless": numberless,
        "pushed": pushed, "spearman": spearman_rows,
        "audit_available": audit_codes(audit_dir) is not None,
    }
