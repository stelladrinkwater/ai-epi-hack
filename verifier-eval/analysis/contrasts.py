"""Paired bootstrap contrasts (SPEC v1.5 §6).

The two primary contrasts are pooled across classes and carry the only p-values;
per-class rows are exploratory and carry none. The C0 filter is a fixed
per-claim property read off the observed C0 samples: a resample draws only
filtered claims, so the filter's selection noise is not inside the interval.
Cells with fewer than 10 contributing claims in either arm report raw counts
and rates only, with no interval and no p-value.
"""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from analysis import tables

MIN_CONTRIBUTING_CLAIMS = 10
N_RESAMPLES = 10000

# name, manipulation, control, classes the primary contrast is pooled over
PRIMARY = (
    ("M1_vs_C0p", "M1", "C0p", ("SUPPORT", "REFUTE")),
    ("M2_vs_C2", "M2", "C2", tables.LABELS),
)
SECONDARY = (("M2_vs_C0p", "M2", "C0p"),)


def _gold_verdict(label: str) -> str:
    return {"SUPPORT": "SUPPORTED", "REFUTE": "REFUTED", "NEI": "NOT_ENOUGH_INFO"}[label]


def _class_of(samples: list[dict[str, Any]]) -> dict[str, str]:
    return {
        sample["claim_id"]: sample["class"]
        for sample in samples
        if sample["manipulation"] == "C0"
    }


def c0_inclusion(samples: list[dict[str, Any]]) -> dict[str, bool]:
    gold = _class_of(samples)
    verdicts: dict[str, list[str | None]] = defaultdict(list)
    for sample in samples:
        if sample["manipulation"] == "C0":
            verdicts[sample["claim_id"]].append(sample["model_verdict"])
    return {
        claim_id: tables.majority(verdicts.get(claim_id, [])) == _gold_verdict(label)
        for claim_id, label in gold.items()
    }


def _has_rows(
    samples: list[dict[str, Any]], manipulation: str, control: str, classes: tuple[str, ...]
) -> bool:
    return any(
        sample["manipulation"] in (manipulation, control) and sample["class"] in classes
        for sample in samples
    )


def _arm_counts(
    samples: list[dict[str, Any]],
    included: dict[str, bool],
    condition: str,
    classes: tuple[str, ...],
) -> tuple[int, int]:
    numerator = denominator = 0
    for sample in samples:
        if sample["manipulation"] != condition or sample["class"] not in classes:
            continue
        if not included.get(sample["claim_id"], False):
            continue
        if sample.get("bin") != "high" or sample["model_verdict"] is None:
            continue
        denominator += 1
        if sample["as_expected"] is False:
            numerator += 1
    return numerator, denominator


def cell_counts(
    samples: list[dict[str, Any]],
    included: dict[str, bool],
    condition: str,
    label: str,
) -> tuple[int, int]:
    """Key-cell numerator and denominator for one condition and class."""
    return _arm_counts(samples, included, condition, (label,))


def _per_claim_counts(
    samples: list[dict[str, Any]],
    included: dict[str, bool],
    condition: str,
    classes: tuple[str, ...],
) -> dict[str, tuple[int, int]]:
    per_claim: dict[str, tuple[int, int]] = defaultdict(lambda: (0, 0))
    for sample in samples:
        if sample["manipulation"] != condition or sample["class"] not in classes:
            continue
        if not included.get(sample["claim_id"], False):
            continue
        if sample.get("bin") != "high" or sample["model_verdict"] is None:
            continue
        num, den = per_claim[sample["claim_id"]]
        per_claim[sample["claim_id"]] = (
            num + (1 if sample["as_expected"] is False else 0),
            den + 1,
        )
    return per_claim


def _bootstrap_delta(
    claims: list[str],
    per_m: dict[str, tuple[int, int]],
    per_c: dict[str, tuple[int, int]],
    seed: int,
    n_resamples: int,
) -> list[float]:
    """Resample the filtered claims with replacement; return the rate differences."""
    rng = random.Random(seed)
    deltas: list[float] = []
    for _ in range(n_resamples):
        num_m = den_m = num_c = den_c = 0
        for _ in range(len(claims)):
            claim = claims[rng.randrange(len(claims))]
            a, b = per_m.get(claim, (0, 0))
            num_m += a
            den_m += b
            a, b = per_c.get(claim, (0, 0))
            num_c += a
            den_c += b
        if den_m and den_c:
            deltas.append(num_m / den_m - num_c / den_c)
    return deltas


def _holm(primary_rows: list[dict[str, Any]]) -> None:
    """Holm over exactly the two primary contrasts (SPEC v1.6 §6).

    m is always 2: a missing or undefined primary p-value enters the step-down
    procedure as 1.0, so the remaining contrast is never adjusted with m = 1.
    """
    m = 2
    entries = [
        (1.0 if row.get("p_value") is None else float(row["p_value"]), row)
        for row in primary_rows
    ]
    entries.sort(key=lambda item: item[0])
    previous = 0.0
    for index, (p_value, row) in enumerate(entries):
        adjusted = min(1.0, max(previous, (m - index) * p_value))
        # A row without a p-value takes part as 1.0 in the ordering but reports
        # no adjusted p-value of its own.
        if row.get("p_value") is not None:
            row["p_holm"] = adjusted
        previous = adjusted


def _contrast_row(
    samples: list[dict[str, Any]],
    included: dict[str, bool],
    pool: list[str],
    *,
    name: str,
    manipulation: str,
    control: str,
    classes: tuple[str, ...],
    class_name: str,
    with_p_value: bool,
    seed: int,
    n_resamples: int,
) -> dict[str, Any]:
    num_m, den_m = _arm_counts(samples, included, manipulation, classes)
    num_c, den_c = _arm_counts(samples, included, control, classes)
    per_m = _per_claim_counts(samples, included, manipulation, classes)
    per_c = _per_claim_counts(samples, included, control, classes)
    contributing = sorted(set(per_m) | set(per_c))
    row: dict[str, Any] = {
        "contrast": name,
        "class": class_name,
        "manipulation": manipulation,
        "control": control,
        "n_claims": len(contributing),
        "n_manipulation": den_m,
        "not_as_expected_manipulation": num_m,
        "n_control": den_c,
        "not_as_expected_control": num_c,
        "estimate": None,
        "method": "counts_only",
        "ci_low": None,
        "ci_high": None,
        "p_value": None,
        "p_holm": None,
    }
    if den_m and den_c:
        row["estimate"] = num_m / den_m - num_c / den_c
    enough = (
        len(per_m) >= MIN_CONTRIBUTING_CLAIMS
        and len(per_c) >= MIN_CONTRIBUTING_CLAIMS
        and den_m > 0
        and den_c > 0
    )
    if enough and pool:
        deltas = _bootstrap_delta(pool, per_m, per_c, seed, n_resamples)
        if deltas:
            row["method"] = "bootstrap"
            row["ci_low"] = float(np.percentile(deltas, 2.5))
            row["ci_high"] = float(np.percentile(deltas, 97.5))
            if with_p_value:
                below = sum(1 for delta in deltas if delta <= 0.0) / len(deltas)
                above = sum(1 for delta in deltas if delta >= 0.0) / len(deltas)
                row["p_value"] = min(1.0, 2.0 * min(below, above))
    return row


def run_contrasts(
    samples: list[dict[str, Any]],
    seed: int,
    *,
    n_resamples: int = N_RESAMPLES,
) -> list[dict[str, Any]]:
    """Pooled primary rows first (with p-values), then per-class exploratory rows."""
    included = c0_inclusion(samples)
    classes_of = _class_of(samples)
    filtered = [claim for claim, keep in included.items() if keep]
    rows: list[dict[str, Any]] = []
    primary_rows: list[dict[str, Any]] = []
    for name, manipulation, control, classes in PRIMARY:
        if not _has_rows(samples, manipulation, control, classes):
            continue
        pool = [claim for claim in filtered if classes_of.get(claim) in classes]
        row = _contrast_row(
            samples, included, pool, name=name, manipulation=manipulation,
            control=control, classes=classes, class_name="pooled",
            with_p_value=True, seed=seed, n_resamples=n_resamples,
        )
        rows.append(row)
        primary_rows.append(row)
    _holm(primary_rows)

    for class_name in tables.LABELS:
        pool = [claim for claim in filtered if classes_of.get(claim) == class_name]
        for name, manipulation, control, classes in PRIMARY:
            if class_name not in classes:
                continue
            if not _has_rows(samples, manipulation, control, (class_name,)):
                continue
            rows.append(
                _contrast_row(
                    samples, included, pool, name=name, manipulation=manipulation,
                    control=control, classes=(class_name,), class_name=class_name,
                    with_p_value=False, seed=seed, n_resamples=n_resamples,
                )
            )
        for name, manipulation, control in SECONDARY:
            if not _has_rows(samples, manipulation, control, (class_name,)):
                continue
            rows.append(
                _contrast_row(
                    samples, included, pool, name=name, manipulation=manipulation,
                    control=control, classes=(class_name,), class_name=class_name,
                    with_p_value=False, seed=seed, n_resamples=n_resamples,
                )
            )
    return rows


def write_contrasts(
    rows: list[dict[str, Any]],
    out_dir: Path,
    *,
    seed: int | None = None,
    n_resamples: int | None = None,
) -> Path:
    import csv

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "contrasts.csv"
    columns = [
        "contrast", "class", "manipulation", "control", "n_claims",
        "n_manipulation", "not_as_expected_manipulation", "n_control",
        "not_as_expected_control", "estimate", "method", "ci_low", "ci_high",
        "p_value", "p_holm", "seed", "n_resamples",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            record = {column: row.get(column) for column in columns}
            record["seed"] = seed
            record["n_resamples"] = f"{n_resamples:,}" if n_resamples is not None else None
            writer.writerow(record)
    return path
