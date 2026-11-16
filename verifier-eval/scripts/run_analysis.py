"""Single analysis entry point for a DeepSeek verifier log (SPEC v1.5 §6, §7).

The driver reads ONE explicit log path, fixes the confidence cut points from
that log's C0 rows, assigns the bins, asserts k = 5 result rows per (claim,
condition), and then writes the long table, C0 accuracy, the filter accounting,
the key-cell table, the contrasts, both sensitivity tables, the M1/M2 strata
and the summary. It never globs across logs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
for extra in (_ROOT, _ROOT / "scripts", _ROOT / "src"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from common import DataError, load_config  # noqa: E402
from analysis import contrasts, cutpoints, loaders, report, strata, tables  # noqa: E402


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in columns})


EXPECTED_CORE_CLAIMS = 150
EXPECTED_CORE_ROWS = {"C0": 750, "C0p": 750, "C2": 750, "M1": 500, "M2": 750}


def is_real_log(rows: list[dict[str, Any]]) -> bool:
    """A real run log carries the full provenance tuple; fixture logs do not.

    The v1.6 strict preconditions (freeze surface, core shape, coder sheets)
    apply to real run logs; the v1.5 synthetic fixtures predate provenance.
    """
    return any(
        row.get("is_result")
        and all(field in row for field in loaders.PROVENANCE_FIELDS)
        for row in rows
    )


def assert_core_shape(samples: list[dict[str, Any]]) -> None:
    """SPEC v1.6 §7: the core log holds 150 claims and 750/750/750/500/750 rows."""
    claims = {sample["claim_id"] for sample in samples}
    if len(claims) != EXPECTED_CORE_CLAIMS:
        raise DataError(
            f"the core log holds {len(claims)} claims, not {EXPECTED_CORE_CLAIMS}"
        )
    counts = Counter(sample["manipulation"] for sample in samples)
    if dict(counts) != EXPECTED_CORE_ROWS:
        raise DataError(
            "the core log holds "
            + ", ".join(f"{code}={counts.get(code, 0)}" for code in EXPECTED_CORE_ROWS)
            + " result rows (C0/C0p/C2/M1/M2), not 750/750/750/500/750"
        )


def assert_complete_cells(samples: list[dict[str, Any]], k: int) -> None:
    """SPEC v1.5 §7: before analysis, 5 result rows per (claim, condition)."""
    counts = Counter((sample["claim_id"], sample["manipulation"]) for sample in samples)
    short = sorted(key for key, count in counts.items() if count != k)
    if short:
        shown = ", ".join(f"{cid}/{cond}={counts[(cid, cond)]}" for cid, cond in short[:5])
        raise DataError(
            f"{len(short)} (claim, condition) cell(s) do not hold {k} result rows: {shown}"
        )


def assert_unique_result_keys(rows: list[dict[str, Any]]) -> None:
    """SPEC v1.6 §7: no duplicate result row per (claim, condition, sample_idx)."""
    counts: Counter[tuple[str, str, int]] = Counter()
    for row in rows:
        if row.get("is_result"):
            counts[
                (str(row["claim_id"]), str(row["condition"]), int(row["sample_idx"]))
            ] += 1
    duplicates = sorted(key for key, count in counts.items() if count > 1)
    if duplicates:
        raise DataError(
            f"{len(duplicates)} duplicate result key(s) in the log; first: {duplicates[0]}"
        )


def c0_accuracy_rows(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    accuracy = tables.c0_accuracy(samples)
    return [{"class": label, **accuracy[label]} for label in tables.LABELS]


def included_rows(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts = tables.included_claims(samples)
    rows = [
        {
            "class": label,
            "included": counts["included"][label],
            "excluded_no_majority": counts["excluded_no_majority"][label],
            "excluded_wrong_majority": counts["excluded_wrong_majority"][label],
            "claims": counts["claims"][label],
        }
        for label in tables.LABELS
    ]
    rows.append(
        {
            "class": "TOTAL",
            "included": sum(counts["included"].values()),
            "excluded_no_majority": counts["excluded_no_majority"].total,
            "excluded_wrong_majority": counts["excluded_wrong_majority"].total,
            "claims": sum(counts["claims"].values()),
        }
    )
    return rows


NO_VERDICT_OUTCOMES = ("parse_failure", "blocked", "truncated", "api_error")


ANALYSIS_HASH = "ANALYSIS.HASH"
ANALYSIS_TAG = "analysis-frozen"
HASHED_DIRS = ("analysis", "src", "prompts")
HASHED_FILES = (
    "config.yaml",
    "scripts/run_experiment.py",
    "scripts/run_analysis.py",
    "scripts/common.py",
    "scripts/make_flags.py",
)


def _git(root: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )


def _analysis_hash_matches(root: Path) -> bool:
    manifest = Path(root) / ANALYSIS_HASH
    if not manifest.is_file():
        return False
    listed: dict[str, str] = {}
    for raw in manifest.read_text(encoding="utf-8").splitlines():
        line = raw.replace("\r", "").strip()
        if not line:
            continue
        digest, relative = line.split(None, 1)
        listed[relative.strip()] = digest.strip()
    if not listed:
        return False
    for relative, digest in listed.items():
        target = Path(root) / relative
        if not target.is_file():
            return False
        if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
            return False
    targets = [relative for relative in HASHED_FILES if (Path(root) / relative).is_file()]
    for directory in HASHED_DIRS:
        base = Path(root) / directory
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                targets.append(path.relative_to(root).as_posix())
    return all(target in listed for target in targets)


def assert_frozen_surface(root: Path) -> None:
    """SPEC v1.6 §7: clean tree, analysis-frozen ancestor, ANALYSIS.HASH agreement."""
    tree = _git(root, "status", "--porcelain")
    if tree.returncode != 0:
        raise DataError("refusing to analyse: the project is not a git repository")
    if tree.stdout.strip():
        raise DataError("refusing to analyse: the git tree is not clean")
    tag = _git(root, "rev-parse", "--verify", ANALYSIS_TAG)
    if tag.returncode != 0:
        raise DataError(f"refusing to analyse: tag {ANALYSIS_TAG} is missing")
    if _git(root, "merge-base", "--is-ancestor", ANALYSIS_TAG, "HEAD").returncode != 0:
        raise DataError(
            f"refusing to analyse: tag {ANALYSIS_TAG} is not an ancestor of HEAD"
        )
    if not _analysis_hash_matches(root):
        raise DataError("refusing to analyse: a hashed file differs from ANALYSIS.HASH")


def commit_of(root: Path) -> str | None:
    done = _git(root, "rev-parse", "HEAD")
    return done.stdout.strip() if done.returncode == 0 else None


def _cell_counts(
    samples: list[dict[str, Any]],
    included: set[str],
    condition: str,
    classes: tuple[str, ...],
) -> dict[str, Any]:
    """Samples, no-verdict outcomes and per-bin (n, not_as_expected) for one cell."""
    cell: dict[str, Any] = {
        "samples": 0,
        "no_verdict": 0,
        "outcomes": {outcome: 0 for outcome in NO_VERDICT_OUTCOMES},
        "bins": {name: (0, 0) for name in ("low", "mid", "high")},
    }
    for sample in samples:
        if sample["manipulation"] != condition or sample["class"] not in classes:
            continue
        if sample["claim_id"] not in included:
            continue
        cell["samples"] += 1
        if sample["model_verdict"] is None:
            cell["no_verdict"] += 1
            outcome = str(sample.get("outcome"))
            if outcome in cell["outcomes"]:
                cell["outcomes"][outcome] += 1
            continue
        bin_name = sample.get("bin")
        if bin_name in cell["bins"]:
            n, nae = cell["bins"][bin_name]
            cell["bins"][bin_name] = (
                n + 1,
                nae + (1 if sample["as_expected"] is False else 0),
            )
    return cell


def _key_row(
    condition: str, label: str, bin_name: str, cell: dict[str, Any]
) -> dict[str, Any]:
    n, nae = cell["bins"][bin_name]
    row = {
        "condition": condition,
        "class": label,
        "bin": bin_name,
        "n": n,
        "not_as_expected": nae,
        "rate": (nae / n) if n else None,
        "no_verdict": cell["no_verdict"],
        "samples": cell["samples"],
    }
    row.update(cell["outcomes"])
    return row


def key_cell_rows(
    samples: list[dict[str, Any]],
    points: dict[str, Any],
    *,
    included: set[str] | None = None,
) -> list[dict[str, Any]]:
    included = tables.included_claim_ids(samples) if included is None else included
    rows: list[dict[str, Any]] = []
    conditions = sorted({sample["manipulation"] for sample in samples})
    for condition in conditions:
        labels = sorted(
            {sample["class"] for sample in samples if sample["manipulation"] == condition}
        )
        for label in labels:
            cell = _cell_counts(samples, included, condition, (label,))
            for bin_name in ("low", "mid", "high"):
                rows.append(_key_row(condition, label, bin_name, cell))
        if condition in ("M1", "C_closed"):
            pooled = _cell_counts(samples, included, condition, ("SUPPORT", "REFUTE"))
            for bin_name in ("low", "mid", "high"):
                rows.append(_key_row(condition, "pooled", bin_name, pooled))
    return rows


GOLD_VERDICT = {"SUPPORT": "SUPPORTED", "REFUTE": "REFUTED", "NEI": "NOT_ENOUGH_INFO"}


def c0_confidence_rows(
    samples: list[dict[str, Any]], included: set[str]
) -> list[dict[str, Any]]:
    """Mean stated confidence of correct C0 verdicts per class (SPEC v1.6 §6)."""
    rows: list[dict[str, Any]] = []
    for label in tables.LABELS:
        for scope_name, scope in (("all", None), ("included", included)):
            values = [
                sample["confidence"]
                for sample in samples
                if sample["manipulation"] == "C0"
                and sample["class"] == label
                and (scope is None or sample["claim_id"] in scope)
                and sample["model_verdict"] == GOLD_VERDICT[label]
                and sample["confidence"] is not None
            ]
            rows.append(
                {
                    "class": label,
                    "scope": scope_name,
                    "n_correct": len(values),
                    "mean_confidence": (sum(values) / len(values)) if values else None,
                }
            )
    return rows


def _load_tier(
    log_path: Path,
    gold_path: Path,
    conditions_dir: Path,
    flags_path: Path | None,
    k: int,
) -> list[dict[str, Any]]:
    """Load a non-core tier log and check its cells and uniqueness."""
    log_path = Path(log_path)
    if not log_path.is_file():
        raise DataError(f"tier log must name a file, not a directory: {log_path}")
    assert_unique_result_keys(loaders.read_log_rows(log_path))
    tier = loaders.load_samples(log_path, gold_path, conditions_dir, flags_path)
    assert_complete_cells(tier, k)
    return tier


def extended_key_cell_rows(
    tier: list[dict[str, Any]], points: dict[str, Any], included: set[str]
) -> list[dict[str, Any]]:
    """C1 / C_closed key cells on the core filter and the core cut points."""
    cutpoints.apply_bins(tier, points)
    rows: list[dict[str, Any]] = []
    for condition in sorted({sample["manipulation"] for sample in tier}):
        labels = sorted(
            {sample["class"] for sample in tier if sample["manipulation"] == condition}
        )
        for label in labels:
            cell = _cell_counts(tier, included, condition, (label,))
            for bin_name in ("low", "mid", "high"):
                rows.append(_key_row(condition, label, bin_name, cell))
    return rows


def naive_rows(
    tier: list[dict[str, Any]],
    core_samples: list[dict[str, Any]],
    points: dict[str, Any],
    included: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Naive M2 key cells and naive-vs-warned as_expected rates, no p-values."""
    cutpoints.apply_bins(tier, points)
    key_rows: list[dict[str, Any]] = []
    for label in tables.LABELS:
        cell = _cell_counts(tier, included, "M2", (label,))
        for bin_name in ("low", "mid", "high"):
            row = _key_row("M2", label, bin_name, cell)
            row["variant"] = "naive"
            key_rows.append(row)
    rate_rows: list[dict[str, Any]] = []
    for condition in ("C0", "M2"):
        for label in tables.LABELS:
            for variant, source in (("naive", tier), ("warned", core_samples)):
                considered = [
                    sample
                    for sample in source
                    if sample["manipulation"] == condition
                    and sample["class"] == label
                    and sample["claim_id"] in included
                    and sample["model_verdict"] is not None
                ]
                expected = sum(1 for sample in considered if sample["as_expected"] is True)
                rate_rows.append(
                    {
                        "condition": condition,
                        "class": label,
                        "variant": variant,
                        "n": len(considered),
                        "as_expected": expected,
                        "rate": (expected / len(considered)) if considered else None,
                    }
                )
    return key_rows, rate_rows


M1_SHEETS = ("coded_onurcan.csv", "coded_stella.csv")


def coder_sheets(root: Path) -> dict[str, Path]:
    """Coder sheets under audit/ whose SHA-256 appears in CHANGELOG (SPEC v1.6 §5).

    The audit names coder 1's and coder 2's M1 sheets (``M1_SHEETS``); when both
    are present only those are read, so the C1/C2 filler sheets can never be
    mistaken for M1 codings. A fixture tree with other recorded sheet names
    (``coded_onurcan.csv`` etc.) falls back to every recorded ``coded_*.csv``
    except the template and the filler sheets.
    """
    audit = Path(root) / "audit"
    changelog_path = Path(root) / "CHANGELOG.md"
    changelog = (
        changelog_path.read_text(encoding="utf-8")
        if changelog_path.is_file()
        else ""
    )
    if not audit.is_dir():
        return {}
    preferred = [audit / name for name in M1_SHEETS]
    if all(path.is_file() for path in preferred):
        candidates = preferred
    else:
        candidates = [
            path
            for path in sorted(audit.glob("coded_*.csv"))
            if path.name != "coded_template.csv" and "filler" not in path.name
        ]
    sheets: dict[str, Path] = {}
    for path in candidates:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest in changelog:
            sheets[path.stem[len("coded_"):]] = path
    return sheets


def _read_codes(path: Path) -> dict[str, str]:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return {
            row["row"].strip(): (row.get("leftover_support") or "").strip()
            for row in csv.DictReader(handle)
            if row.get("row")
        }


def _audit_key(root: Path) -> dict[str, str]:
    path = Path(root) / "audit" / "key.csv"
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8", newline="") as handle:
        return {
            row["row"].strip(): row["id"]
            for row in csv.DictReader(handle)
            if row.get("sheet") == "m1_audit" and row.get("row")
        }


def merged_strong(root: Path, sheets: dict[str, Path]) -> tuple[set[str], int]:
    """Either coder's 'strong' marks an item; count the rows the coders dispute."""
    key = _audit_key(root)
    codes = {name: _read_codes(path) for name, path in sheets.items()}
    strong: set[str] = set()
    disagreements = 0
    for row, claim_id in key.items():
        values = {codes[name][row] for name in codes if row in codes[name]}
        if "strong" in values:
            strong.add(claim_id)
        if len(values) > 1:
            disagreements += 1
    return strong, disagreements


def without_strong_rows(
    samples: list[dict[str, Any]],
    included: set[str],
    strong_ids: set[str],
    coders: list[str],
    disagreements: int,
) -> list[dict[str, Any]]:
    """The M1 key cell without the items either coder coded strong (SPEC v1.6 §5)."""

    def counts(classes: tuple[str, ...]) -> tuple[int, int]:
        n = nae = 0
        for sample in samples:
            if sample["manipulation"] != "M1" or sample["class"] not in classes:
                continue
            if sample["claim_id"] not in included or sample["claim_id"] in strong_ids:
                continue
            if sample["model_verdict"] is None or sample.get("bin") != "high":
                continue
            n += 1
            nae += 1 if sample["as_expected"] is False else 0
        return nae, n

    def row(label: str) -> dict[str, Any]:
        nae, n = counts(("SUPPORT", "REFUTE") if label == "pooled" else (label,))
        return {
            "condition": "M1",
            "class": "without_strong",
            "claim_class": label,
            "bin": "high",
            "n": n,
            "not_as_expected": nae,
            "rate": (nae / n) if n else None,
            "excluded_strong": len(strong_ids),
            "coders": "+".join(coders),
            "disagreements": disagreements,
        }

    return [row("pooled"), row("SUPPORT"), row("REFUTE")]


def degeneracy_rows(cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Degeneracy = share of cells whose stability is 1.0, pooled and per condition."""

    def share(subset: list[dict[str, Any]]) -> float | None:
        if not subset:
            return None
        return sum(1 for cell in subset if cell["stability"] == 1.0) / len(subset)

    rows: list[dict[str, Any]] = [
        {"scope": "pooled", "n_cells": len(cells), "degenerate_share": share(cells)}
    ]
    for condition in sorted({cell["condition"] for cell in cells}):
        mine = [cell for cell in cells if cell["condition"] == condition]
        rows.append(
            {"scope": condition, "n_cells": len(mine), "degenerate_share": share(mine)}
        )
    return rows


def _threshold_rate(
    samples: list[dict[str, Any]],
    included: set[str],
    condition: str,
    classes: tuple[str, ...],
    threshold: int,
) -> float | None:
    n = nae = 0
    for sample in samples:
        if sample["manipulation"] != condition or sample["class"] not in classes:
            continue
        if sample["claim_id"] not in included or sample["model_verdict"] is None:
            continue
        confidence = sample.get("confidence")
        if confidence is None or confidence < threshold:
            continue
        n += 1
        nae += 1 if sample["as_expected"] is False else 0
    return (nae / n) if n else None


def _no_verdict_rate(
    samples: list[dict[str, Any]],
    included: set[str],
    condition: str,
    classes: tuple[str, ...],
    *,
    high_floor: float,
    all_bins: bool,
) -> float | None:
    n = nae = 0
    for sample in samples:
        if sample["manipulation"] != condition or sample["class"] not in classes:
            continue
        if sample["claim_id"] not in included:
            continue
        if sample["model_verdict"] is None:
            n += 1
            nae += 1
            continue
        confidence = sample.get("confidence")
        if all_bins or (confidence is not None and confidence >= high_floor):
            n += 1
            nae += 1 if sample["as_expected"] is False else 0
    return (nae / n) if n else None


def sensitivity_contrast_rows(
    samples: list[dict[str, Any]], points: dict[str, Any], included: set[str]
) -> list[dict[str, Any]]:
    """Both sensitivity rules recompute the two pooled primary contrasts (SPEC v1.6 §6)."""
    rows: list[dict[str, Any]] = []
    for name, manipulation, control, classes in contrasts.PRIMARY:
        treated = _threshold_rate(
            samples, included, manipulation, classes, cutpoints.SENSITIVITY_HIGH
        )
        reference = _threshold_rate(
            samples, included, control, classes, cutpoints.SENSITIVITY_HIGH
        )
        rows.append(
            {
                "rule": f"threshold_high_ge_{cutpoints.SENSITIVITY_HIGH}",
                "contrast": name,
                "class": "pooled",
                "manipulation": manipulation,
                "control": control,
                "estimate": None if treated is None or reference is None
                else treated - reference,
            }
        )
        for all_bins in (False, True):
            treated = _no_verdict_rate(
                samples, included, manipulation, classes,
                high_floor=points["q2"], all_bins=all_bins,
            )
            reference = _no_verdict_rate(
                samples, included, control, classes,
                high_floor=points["q2"], all_bins=all_bins,
            )
            rows.append(
                {
                    "rule": "no_verdict_as_error_all_bins" if all_bins
                    else "no_verdict_as_error_high_bin",
                    "contrast": name,
                    "class": "pooled",
                    "manipulation": manipulation,
                    "control": control,
                    "estimate": None if treated is None or reference is None
                    else treated - reference,
                }
            )
    return rows


def sensitivity_no_verdict_rows(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sensitivity table 1: no_verdict samples counted as not_as_expected."""
    included = tables.included_claim_ids(samples)
    groups: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: {"high_n": 0, "high_not_as_expected": 0, "samples": 0,
                 "with_verdict": 0, "no_verdict": 0, "not_as_expected": 0}
    )
    for sample in samples:
        if sample["claim_id"] not in included:
            continue
        group = groups[(sample["manipulation"], sample["class"])]
        group["samples"] += 1
        if sample["model_verdict"] is None:
            group["no_verdict"] += 1
            continue
        group["with_verdict"] += 1
        if sample["as_expected"] is False:
            group["not_as_expected"] += 1
        if sample.get("bin") == "high":
            group["high_n"] += 1
            if sample["as_expected"] is False:
                group["high_not_as_expected"] += 1
    rows: list[dict[str, Any]] = []
    for (condition, label), group in sorted(groups.items()):
        rows.append(
            {
                "condition": condition,
                "class": label,
                "high_n": group["high_n"],
                "high_not_as_expected": group["high_not_as_expected"],
                "high_rate": (
                    group["high_not_as_expected"] / group["high_n"]
                    if group["high_n"]
                    else None
                ),
                "samples": group["samples"],
                "with_verdict": group["with_verdict"],
                "no_verdict": group["no_verdict"],
                "not_as_expected": group["not_as_expected"],
                "rate_excluding_no_verdict": (
                    group["not_as_expected"] / group["with_verdict"]
                    if group["with_verdict"]
                    else None
                ),
                "not_as_expected_with_no_verdict": (
                    group["not_as_expected"] + group["no_verdict"]
                ),
                "rate_with_no_verdict": (
                    (group["not_as_expected"] + group["no_verdict"]) / group["samples"]
                    if group["samples"]
                    else None
                ),
            }
        )
    return rows


def sensitivity_high_90_rows(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sensitivity table 2: fixed threshold, high = confidence >= 90."""
    included = tables.included_claim_ids(samples)
    groups: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: {"high_n": 0, "high_not_as_expected": 0,
                 "below_n": 0, "below_not_as_expected": 0}
    )
    for sample in samples:
        if sample["claim_id"] not in included or sample["model_verdict"] is None:
            continue
        confidence = sample.get("confidence")
        if confidence is None:
            continue
        group = groups[(sample["manipulation"], sample["class"])]
        if confidence >= cutpoints.SENSITIVITY_HIGH:
            group["high_n"] += 1
            if sample["as_expected"] is False:
                group["high_not_as_expected"] += 1
        else:
            group["below_n"] += 1
            if sample["as_expected"] is False:
                group["below_not_as_expected"] += 1
    rows: list[dict[str, Any]] = []
    for (condition, label), group in sorted(groups.items()):
        rows.append(
            {
                "condition": condition,
                "class": label,
                "threshold": f"high >= {cutpoints.SENSITIVITY_HIGH}",
                "high_n": group["high_n"],
                "high_not_as_expected": group["high_not_as_expected"],
                "high_rate": (
                    group["high_not_as_expected"] / group["high_n"]
                    if group["high_n"]
                    else None
                ),
                "below_n": group["below_n"],
                "below_not_as_expected": group["below_not_as_expected"],
                "below_rate": (
                    group["below_not_as_expected"] / group["below_n"]
                    if group["below_n"]
                    else None
                ),
            }
        )
    return rows


def analyse(
    log_path: Path,
    out_dir: Path,
    config_path: Path,
    *,
    extended_log: Path | None = None,
    naive_log: Path | None = None,
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    root = config_path.parent
    config = load_config(config_path)
    log_path = Path(log_path)
    if not log_path.is_file():
        raise DataError(f"--log must name a file, not a directory: {log_path}")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    section = config.get("run") or {}
    k = int(section.get("k", 5))
    gold_path = root / "data" / "gold.jsonl"
    conditions_dir = root / "data" / "conditions"
    flags_path = conditions_dir / "flags.jsonl"
    audit_dir = root / "audit"
    log_rows = loaders.read_log_rows(log_path)
    if is_real_log(log_rows):
        assert_frozen_surface(root)
    assert_unique_result_keys(log_rows)
    samples = loaders.load_samples(
        Path(log_path),
        gold_path,
        conditions_dir,
        flags_path if flags_path.is_file() else None,
    )
    assert_complete_cells(samples, k)
    strong_ids: set[str] = set()
    coders: list[str] = []
    disagreements = 0
    if is_real_log(log_rows):
        assert_core_shape(samples)
        if any(sample["manipulation"] == "M1" for sample in samples):
            sheets = coder_sheets(root)
            if not sheets:
                raise DataError(
                    "refusing to analyse: no M1 coder sheet under audit/ has its "
                    "SHA-256 in CHANGELOG"
                )
            coders = sorted(sheets)
            strong_ids, disagreements = merged_strong(root, sheets)
    points = cutpoints.write_cutpoints(samples, out)
    cutpoints.apply_bins(samples, points)
    loaders.write_long_table(samples, out / "long.csv")
    _write_csv(
        out / "c0_accuracy.csv",
        c0_accuracy_rows(samples),
        ["class", "claims", "correct", "accuracy"],
    )
    _write_csv(
        out / "included_claims.csv",
        included_rows(samples),
        ["class", "included", "excluded_no_majority", "excluded_wrong_majority", "claims"],
    )
    core_included = tables.included_claim_ids(samples)
    _write_csv(
        out / "c0_confidence_by_class.csv",
        c0_confidence_rows(samples, core_included),
        ["class", "scope", "n_correct", "mean_confidence"],
    )
    _write_csv(
        out / "key_cell.csv",
        key_cell_rows(samples, points),
        ["condition", "class", "bin", "n", "not_as_expected", "rate", "no_verdict",
         "samples", *NO_VERDICT_OUTCOMES],
    )
    _write_csv(
        out / "m1_without_strong.csv",
        without_strong_rows(samples, core_included, strong_ids, coders, disagreements),
        ["condition", "class", "claim_class", "bin", "n", "not_as_expected", "rate",
         "excluded_strong", "coders", "disagreements"],
    )
    _write_csv(
        out / "sensitivity_no_verdict.csv",
        sensitivity_no_verdict_rows(samples),
        ["condition", "class", "high_n", "high_not_as_expected", "high_rate", "samples",
         "with_verdict", "no_verdict", "not_as_expected", "rate_excluding_no_verdict",
         "not_as_expected_with_no_verdict", "rate_with_no_verdict"],
    )
    _write_csv(
        out / "sensitivity_high_90.csv",
        sensitivity_high_90_rows(samples),
        ["condition", "class", "threshold", "high_n", "high_not_as_expected", "high_rate",
         "below_n", "below_not_as_expected", "below_rate"],
    )
    stability_cells = tables.stability_cell_rows(samples)
    _write_csv(
        out / "stability_cells.csv",
        stability_cells,
        ["claim_id", "condition", "majority", "stability", "n_confidence", "confidence_sd"],
    )
    _write_csv(
        out / "stability_degeneracy.csv",
        degeneracy_rows(stability_cells),
        ["scope", "n_cells", "degenerate_share"],
    )
    _write_csv(
        out / "sensitivity_contrasts.csv",
        sensitivity_contrast_rows(samples, points, tables.included_claim_ids(samples)),
        ["rule", "contrast", "class", "manipulation", "control", "estimate"],
    )
    tier_flags = flags_path if flags_path.is_file() else None
    if extended_log is not None:
        extended = _load_tier(
            Path(extended_log), gold_path, conditions_dir, tier_flags, k
        )
        _write_csv(
            out / "key_cell_extended.csv",
            extended_key_cell_rows(extended, points, core_included),
            ["condition", "class", "bin", "n", "not_as_expected", "rate",
             "no_verdict", "samples"],
        )
    if naive_log is not None:
        naive = _load_tier(Path(naive_log), gold_path, conditions_dir, tier_flags, k)
        naive_key, naive_rates = naive_rows(naive, samples, points, core_included)
        _write_csv(
            out / "naive_key_cell.csv",
            naive_key,
            ["condition", "class", "bin", "n", "not_as_expected", "rate",
             "no_verdict", "samples", "variant"],
        )
        _write_csv(
            out / "naive_rates.csv",
            naive_rates,
            ["condition", "class", "variant", "n", "as_expected", "rate"],
        )
    analysis_seed = int(config["seed"])
    n_resamples = contrasts.N_RESAMPLES
    contrast_rows = contrasts.run_contrasts(samples, analysis_seed, n_resamples=n_resamples)
    contrasts.write_contrasts(
        contrast_rows, out, seed=analysis_seed, n_resamples=n_resamples
    )
    strata_data = strata.write_strata(
        samples, out, audit_dir=audit_dir if audit_dir.is_dir() else None
    )
    report.write_summary(
        contrast_rows, strata_data, out,
        seed=analysis_seed, n_resamples=n_resamples, commit=commit_of(root),
        logs={"core": str(log_path)},
    )
    return {
        "log": str(Path(log_path)),
        "out": str(out),
        "samples": len(samples),
        "claims": len({sample["claim_id"] for sample in samples}),
        "cutpoints": points,
        "contrasts": len(contrast_rows),
        "seed": analysis_seed,
        "n_resamples": n_resamples,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True, help="one explicit result log (JSONL)")
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    parser.add_argument("--config", type=Path, default=_ROOT / "config.yaml")
    parser.add_argument("--extended-log", type=Path, default=None,
                        help="extended tier log (C1, C_closed), analysed against the core log")
    parser.add_argument("--naive-log", type=Path, default=None,
                        help="naive prompt log (C0, M2), analysed against the core log")
    args = parser.parse_args(argv)
    try:
        summary = analyse(
            args.log, args.out, args.config,
            extended_log=args.extended_log, naive_log=args.naive_log,
        )
    except DataError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
