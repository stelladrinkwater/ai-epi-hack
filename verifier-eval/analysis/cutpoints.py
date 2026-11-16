"""Tertile cut points for the confidence bins (SPEC v1.4 §6).

Reads ONLY C0 sample-level confidence of ok rows. Writes analysis/cutpoints.json
(LF) and its SHA-256.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from common import DataError

Q1 = 33.33
Q2 = 66.67
SENSITIVITY_HIGH = 90


def c0_ok_confidences(samples: list[dict[str, Any]]) -> list[int]:
    return [
        int(sample["confidence"])
        for sample in samples
        if sample["manipulation"] == "C0"
        and sample.get("outcome") == "ok"
        and sample.get("confidence") is not None
    ]


def compute_cutpoints(samples: list[dict[str, Any]]) -> dict[str, Any]:
    values = c0_ok_confidences(samples)
    if not values:
        raise DataError("no C0 ok samples with confidence; cannot compute cut points")
    q1, q2 = np.percentile(np.asarray(values), [Q1, Q2], method="inverted_cdf")
    q1, q2 = float(q1), float(q2)
    return {
        "q1": q1,
        "q2": q2,
        "n": len(values),
        "degenerate": q1 == q2,
        "sensitivity_high": SENSITIVITY_HIGH,
    }


def assign_bin(confidence: int | None, cutpoints: dict[str, Any]) -> str | None:
    if confidence is None:
        return None
    if confidence < cutpoints["q1"]:
        return "low"
    if confidence < cutpoints["q2"]:
        return "mid"
    return "high"


def apply_bins(samples: list[dict[str, Any]], cutpoints: dict[str, Any]) -> None:
    for sample in samples:
        sample["bin"] = assign_bin(sample.get("confidence"), cutpoints)


def write_cutpoints(samples: list[dict[str, Any]], out_dir: Path) -> dict[str, Any]:
    cutpoints = compute_cutpoints(samples)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "cutpoints.json"
    payload = json.dumps(cutpoints, indent=2, sort_keys=True) + "\n"
    with json_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    with (out / "CUTPOINTS.HASH").open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(digest + "\n")
    return cutpoints
