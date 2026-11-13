#!/usr/bin/env python3
"""External support-evidence scoring for confidence claims.

Input is JSONL. Each row has: id, claim (or question), confidence (0..1),
and similar_examples (list of strings or objects with a `text` field).
The scorer compares model-cited situations to the target using a corpus-aware
TF-IDF cosine baseline. It does not treat confidence as evidence.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


def tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def cosine_tfidf(query: str, documents: list[str]) -> list[float]:
    """Cosine similarity with IDF computed across this evaluation's examples."""
    docs = [tokens(d) for d in documents]
    q = tokens(query)
    if not q or not any(docs):
        return [0.0] * len(documents)
    df = Counter(term for doc in docs for term in set(doc))
    n = max(1, len(docs))

    def vector(words: list[str]) -> dict[str, float]:
        counts = Counter(words)
        return {t: (1 + math.log(c)) * (math.log((n + 1) / (df[t] + 1)) + 1)
                for t, c in counts.items()}

    qv = vector(q)
    qnorm = math.sqrt(sum(x * x for x in qv.values()))
    scores = []
    for words in docs:
        dv = vector(words)
        dnorm = math.sqrt(sum(x * x for x in dv.values()))
        dot = sum(value * dv.get(term, 0.0) for term, value in qv.items())
        scores.append(dot / (qnorm * dnorm) if qnorm and dnorm else 0.0)
    return scores


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2:
        return None
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    dx, dy = [x - mx for x in xs], [y - my for y in ys]
    denom = math.sqrt(sum(x*x for x in dx) * sum(y*y for y in dy))
    return sum(x*y for x, y in zip(dx, dy)) / denom if denom else None


def ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    out = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        rank = (i + 1 + j) / 2
        for k in order[i:j]:
            out[k] = rank
        i = j
    return out


def analyze(rows: list[dict[str, Any]], top_k: int = 5) -> dict[str, Any]:
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    cases = []
    for i, row in enumerate(rows):
        target = row.get("claim") or row.get("question")
        if not isinstance(target, str) or not target.strip():
            raise ValueError(f"row {i + 1}: provide a non-empty claim or question")
        confidence = float(row["confidence"])
        if not 0 <= confidence <= 1:
            raise ValueError(f"row {i + 1}: confidence must be between 0 and 1")
        examples = row.get("similar_examples", [])
        if not isinstance(examples, list):
            raise ValueError(f"row {i + 1}: similar_examples must be a list")
        normalized = []
        for ex in examples:
            text = ex if isinstance(ex, str) else ex.get("text", "") if isinstance(ex, dict) else ""
            if isinstance(text, str) and text.strip():
                normalized.append(text.strip())
        scores = cosine_tfidf(target, normalized)
        ranked = sorted(zip(normalized, scores), key=lambda pair: pair[1], reverse=True)[:top_k]
        # Sum the strongest matches, normalized to 0..1 for easier comparison.
        support = sum(score for _, score in ranked) / top_k
        cases.append({
            "id": row.get("id", str(i + 1)), "claim": target,
            "confidence": confidence,
            "support_score": round(support, 6),
            "scored_examples": [
                {"text": text, "similarity": round(score, 6)} for text, score in ranked
            ],
        })
    confidences = [case["confidence"] for case in cases]
    supports = [case["support_score"] for case in cases]
    return {
        "method": "tf-idf cosine over model-cited examples; support = sum(top-k similarity) / k",
        "top_k": top_k,
        "n": len(cases),
        "correlation": {
            "pearson_confidence_vs_support": pearson(confidences, supports),
            "spearman_confidence_vs_support": pearson(ranks(confidences), ranks(supports)),
        },
        "cases": cases,
    }


def read_jsonl(path: str) -> list[dict[str, Any]]:
    rows = []
    with (sys.stdin if path == "-" else Path(path).open(encoding="utf-8")) as handle:
        for line_no, line in enumerate(handle, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"line {line_no}: invalid JSON: {exc}") from exc
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="JSONL file, or - for stdin")
    parser.add_argument("--top-k", type=int, default=5, help="number of strongest examples to aggregate (default: 5)")
    parser.add_argument("--output", help="write JSON report to this file (default: stdout)")
    args = parser.parse_args()
    try:
        report = analyze(read_jsonl(args.input), args.top_k)
    except (ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    rendered = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
