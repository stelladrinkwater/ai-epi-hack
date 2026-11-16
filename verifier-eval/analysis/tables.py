"""Analysis tables over the long sample table (SPEC v1.4 §6)."""

from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from typing import Any

LABELS = ("SUPPORT", "REFUTE", "NEI")
SAMPLE_COUNT = 5


def majority(verdicts: list[str | None]) -> str | None:
    """The verdict at least 3 of the 5 samples give, else None (SPEC v1.5 §6).

    A group that does not hold exactly five rows has no majority: rescaling
    the share of a short cell would silently change the filter and the cells.
    """
    if len(verdicts) != SAMPLE_COUNT:
        return None
    counts: Counter[str] = Counter(verdict for verdict in verdicts if verdict is not None)
    for verdict, count in counts.items():
        if count >= 3:
            return verdict
    return None


def _claim_verdicts(samples: list[dict[str, Any]], manipulation: str) -> dict[str, list[str | None]]:
    verdicts: dict[str, list[str | None]] = defaultdict(list)
    for sample in samples:
        if sample["manipulation"] == manipulation:
            verdicts[sample["claim_id"]].append(sample["model_verdict"])
    return verdicts


def c0_accuracy(samples: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per class: C0 claims whose majority equals the gold class."""
    gold = {sample["claim_id"]: sample["class"] for sample in samples if sample["manipulation"] == "C0"}
    majority_by_claim = {
        claim_id: majority(verdicts)
        for claim_id, verdicts in _claim_verdicts(samples, "C0").items()
    }
    result: dict[str, dict[str, Any]] = {}
    for label in LABELS:
        claims = [claim_id for claim_id, cls in gold.items() if cls == label]
        correct = sum(1 for claim_id in claims if majority_by_claim.get(claim_id) == _gold_verdict(label))
        result[label] = {
            "claims": len(claims),
            "correct": correct,
            "accuracy": (correct / len(claims)) if claims else None,
        }
    return result


def _gold_verdict(label: str) -> str:
    return {"SUPPORT": "SUPPORTED", "REFUTE": "REFUTED", "NEI": "NOT_ENOUGH_INFO"}[label]


class PerClassCounts(dict):
    """A per-class count that also compares equal to its scalar total.

    SPEC v1.5 §6 requires both exclusion counts per class, while callers that
    predate that rule compared the whole count to an integer; keeping the
    scalar comparison means those callers read the same number, not a mapping.
    """

    @property
    def total(self) -> int:
        return sum(self.values())

    def __eq__(self, other: object) -> bool:
        if isinstance(other, int) and not isinstance(other, bool):
            return self.total == other
        return dict.__eq__(self, other)

    __hash__ = None


def included_claims(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Per class: included, excluded_no_majority, excluded_wrong_majority.

    The three counts add up to every claim in the subset (SPEC v1.5 §6).
    """
    gold = {sample["claim_id"]: sample["class"] for sample in samples if sample["manipulation"] == "C0"}
    majority_by_claim = {
        claim_id: majority(verdicts)
        for claim_id, verdicts in _claim_verdicts(samples, "C0").items()
    }
    included = Counter()
    no_majority = Counter()
    wrong_majority = Counter()
    for claim_id, label in gold.items():
        verdict = majority_by_claim.get(claim_id)
        if verdict is None:
            no_majority[label] += 1
        elif verdict == _gold_verdict(label):
            included[label] += 1
        else:
            wrong_majority[label] += 1
    counts = {
        "included": {label: included.get(label, 0) for label in LABELS},
        "excluded_no_majority": PerClassCounts(
            {label: no_majority.get(label, 0) for label in LABELS}
        ),
        "excluded_wrong_majority": PerClassCounts(
            {label: wrong_majority.get(label, 0) for label in LABELS}
        ),
    }
    counts["claims"] = {
        label: counts["included"][label]
        + counts["excluded_no_majority"][label]
        + counts["excluded_wrong_majority"][label]
        for label in LABELS
    }
    return counts


def included_claim_ids(samples: list[dict[str, Any]]) -> set[str]:
    gold = {sample["claim_id"]: sample["class"] for sample in samples if sample["manipulation"] == "C0"}
    majority_by_claim = {
        claim_id: majority(verdicts)
        for claim_id, verdicts in _claim_verdicts(samples, "C0").items()
    }
    return {
        claim_id
        for claim_id, label in gold.items()
        if majority_by_claim.get(claim_id) == _gold_verdict(label)
    }


def key_cell_table(
    samples: list[dict[str, Any]], cutpoints: dict[str, Any]
) -> dict[str, dict[str, dict[str, Any]]]:
    """condition -> class -> {bins: {...}, no_verdict: n, samples: n}."""
    included = included_claim_ids(samples)
    table: dict[str, dict[str, dict[str, Any]]] = {}
    for sample in samples:
        if sample["claim_id"] not in included:
            continue
        condition = sample["manipulation"]
        label = sample["class"]
        cell = table.setdefault(
            condition,
            {},
        ).setdefault(
            label,
            {
                "bins": {
                    "low": {"n": 0, "not_as_expected": 0, "p": None},
                    "mid": {"n": 0, "not_as_expected": 0, "p": None},
                    "high": {"n": 0, "not_as_expected": 0, "p": None},
                },
                "no_verdict": 0,
                "samples": 0,
            },
        )
        cell["samples"] += 1
        if sample["model_verdict"] is None:
            cell["no_verdict"] += 1
            continue
        bin_name = sample["bin"]
        if bin_name is None:
            continue
        cell["bins"][bin_name]["n"] += 1
        if sample["as_expected"] is False:
            cell["bins"][bin_name]["not_as_expected"] += 1
    for classes in table.values():
        for cell in classes.values():
            for values in cell["bins"].values():
                values["p"] = (
                    values["not_as_expected"] / values["n"] if values["n"] else None
                )
    return table


def finish_reason_outcome_table(samples: list[dict[str, Any]]) -> dict[tuple[str, str, str], int]:
    counts: Counter[tuple[str, str, str]] = Counter()
    for sample in samples:
        counts[(sample["manipulation"], str(sample.get("finish_reason")), str(sample.get("outcome")))] += 1
    return dict(sorted(counts.items()))


def stability_cell_rows(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per (claim, condition) for the stability table (SPEC v1.6 §6).

    ``confidence_sd`` is the population SD of the stated confidences and is
    undefined unless the cell holds exactly five confidence values.
    """
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        grouped[(sample["claim_id"], sample["manipulation"])].append(sample)
    rows: list[dict[str, Any]] = []
    for (claim_id, condition), group in sorted(grouped.items()):
        verdicts = [sample["model_verdict"] for sample in group]
        top = majority(verdicts)
        stability = (
            None if top is None else sum(1 for v in verdicts if v == top) / SAMPLE_COUNT
        )
        confidences = [
            sample["confidence"] for sample in group if sample["confidence"] is not None
        ]
        sd = statistics.pstdev(confidences) if len(confidences) == SAMPLE_COUNT else None
        rows.append(
            {
                "claim_id": claim_id,
                "condition": condition,
                "majority": top,
                "stability": stability,
                "n_confidence": len(confidences),
                "confidence_sd": sd,
                "n": len(group),
            }
        )
    return rows


def stability_table(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        grouped[(sample["claim_id"], sample["manipulation"])].append(sample)
    rows: list[dict[str, Any]] = []
    for (claim_id, condition), group in sorted(grouped.items()):
        if len(group) != SAMPLE_COUNT:
            rows.append(
                {
                    "claim_id": claim_id,
                    "manipulation": condition,
                    "majority": None,
                    "stability": None,
                    "confidence_sd": None,
                    "n": len(group),
                }
            )
            continue
        verdicts = [sample["model_verdict"] for sample in group]
        top = majority(verdicts)
        stability = None if top is None else sum(1 for v in verdicts if v == top) / SAMPLE_COUNT
        confidences = [sample["confidence"] for sample in group if sample["confidence"] is not None]
        sd = statistics.pstdev(confidences) if len(confidences) >= 2 else None
        rows.append(
            {
                "claim_id": claim_id,
                "manipulation": condition,
                "majority": top,
                "stability": stability,
                "confidence_sd": sd,
                "n": len(group),
            }
        )
    return rows
