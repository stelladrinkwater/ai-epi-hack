"""Descriptor file for stratification (SPEC v1.5 section 5).

Reads the seven condition files in ``data/conditions/``, ``data/key.jsonl``,
``data/gold.jsonl``, the SciFact corpus (the same file ``build_conditions.py``
reads) and ``prompts/m2_templates.json``. Writes
``data/conditions/flags.jsonl``: one row per ``(id, condition)``, sorted,
UTF-8 without BOM, LF only, byte-identical across runs. No condition file is
written and no model is called.

Definitions (deterministic; every value is computed from the frozen inputs):

- ``n_sentences`` / ``evidence_chars``: sentence count and the length of the
  rendered evidence (each sentence stripped, joined with one space, exactly as
  the call layer renders it). C_closed has no evidence, so both are 0.
- ``structured`` (M1): the corpus ``structured`` annotation for the claim's
  document; when the corpus carries no such field, a leading heading token
  matching ``^[A-Z][A-Z ]{3,}:?`` on any abstract sentence.
- ``lost_headers`` (M1): number of distinct heading tokens (the same regex) present
  in C0 and gone in M1. With the corpus annotation this reproduces "28 of the 39
  structured abstracts" in SPEC section 8.
- ``dangling_anaphora`` (M1): original indices of kept M1 sentences that begin
  with These|Those|This|Such|Neither|The former|The latter while their
  predecessor sentence was removed.
- ``n_remaining`` (M1): sentences left in M1.
- ``overlap_flag`` (M1): the flag from ``m1_overlap_screen.jsonl``.
- ``position``, ``nearest_gold_distance`` (M2, C2): the seeded insertion
  position and the distance in evidence positions from the inserted sentence to
  the nearest gold sentence; null when the claim has no gold sentence (NEI).
- ``instruction_words`` / ``instruction_chars`` (M2, C2): length of the
  inserted sentence, i.e. the M2 instruction or the word-count-matched C2
  filler.
- ``family``, ``template_index``, ``states_confidence_number``,
  ``stated_number`` (M2): the seeded template and the confidence number it
  states (null when it states none).
- ``prefix5_overlap``, ``entity_overlap`` (C1, C2): whether the inserted filler
  shares a 5-character alphabetic word prefix, or an entity-like token (an
  all-caps acronym of length >= 2, or a token containing a digit such as
  ``DENV-4`` or ``p53``), with the claim.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
for extra in (_ROOT, _ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from common import DataError, load_config, read_jsonl, resolve  # noqa: E402

CODES = ("C0", "C0p", "C1", "C2", "M1", "M2", "C_closed")
FULL_TIER = ("C0", "C0p", "C1", "C2", "M2")
DIRECTIONAL_TIER = ("M1", "C_closed")

HEADING = re.compile(r"^([A-Z][A-Z ]{3,}):?")
ANAPHOR = re.compile(r"^(?:These|Those|This|Such|Neither|The former|The latter)\b")
CONFIDENCE_NUMBER = re.compile(r"confidence[^0-9]{0,20}(\d{1,3})", re.IGNORECASE)
PREFIX_WORD = re.compile(r"[A-Za-z]{5,}")
ENTITY_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9-]*")


# --------------------------------------------------------------------------- #
# Small text helpers
# --------------------------------------------------------------------------- #
def heading_token(text: str) -> str | None:
    """The leading heading token of a sentence, if it starts with one.

    The regex from SPEC section 5 is greedy over ``[A-Z ]`` and would swallow the
    first capital letter of the following word ("OBJECTIVE To" -> "OBJECTIVE T");
    the run is therefore trimmed to its all-caps words. A single short acronym
    such as "DNA " is not a heading (fewer than 4 characters).
    """
    stripped = str(text).strip()
    match = HEADING.match(stripped)
    if not match:
        return None
    kept: list[str] = []
    for word in match.group(1).split():
        if len(word) >= 2 and word.isalpha() and word.isupper():
            kept.append(word)
        else:
            break
    token = " ".join(kept).strip()
    return token if len(token) >= 4 else None


def heading_tokens(texts: list[str]) -> set[str]:
    return {token for token in (heading_token(text) for text in texts) if token}


def render_evidence(sentences: list[dict[str, Any]]) -> str:
    return " ".join(str(sentence.get("text", "")).strip() for sentence in sentences)


def prefix5_tokens(text: str) -> set[str]:
    return {match.group(0).lower()[:5] for match in PREFIX_WORD.finditer(str(text))}


def entity_tokens(text: str) -> set[str]:
    tokens: set[str] = set()
    for match in ENTITY_TOKEN.finditer(str(text)):
        token = match.group(0)
        if any(char.isdigit() for char in token) or (token.isupper() and len(token) >= 2):
            tokens.add(token)
    return tokens


def nearest_gold_distance(
    evidence: list[dict[str, Any]], gold_idx: list[int]
) -> int | None:
    """Distance in evidence positions from the inserted sentence to nearest gold."""
    if not gold_idx:
        return None
    gold = set(gold_idx)
    inserted = [
        index for index, sentence in enumerate(evidence) if sentence.get("idx") is None
    ]
    if not inserted:
        return None
    gold_positions = [
        index for index, sentence in enumerate(evidence) if sentence.get("idx") in gold
    ]
    if not gold_positions:
        return None
    return min(abs(inserted[0] - position) for position in gold_positions)


def stated_confidence(template: str) -> int | None:
    match = CONFIDENCE_NUMBER.search(template)
    if not match:
        return None
    value = int(match.group(1))
    return value if 0 <= value <= 100 else None


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #
def corpus_path_for(root: Path) -> Path:
    """The corpus file, resolved as ``build_conditions`` resolves it."""
    root = Path(root)
    config_path = root / "config.yaml"
    if config_path.is_file():
        config = load_config(config_path)
        data = config["data"]
        return resolve(root, data["raw_dir"]) / data["files"]["corpus"]
    return root / "data" / "raw" / "scifact" / "corpus.jsonl"


def load_abstracts(
    corpus_path: Path, wanted_docs: set[Any]
) -> dict[Any, dict[str, Any]]:
    """doc_id -> {"sentences": [...], "structured": bool | None}."""
    abstracts: dict[Any, dict[str, Any]] = {}
    for row in read_jsonl(corpus_path):
        doc_id = row.get("doc_id")
        if doc_id in wanted_docs:
            marked = row.get("structured")
            abstracts[doc_id] = {
                "sentences": [str(sentence) for sentence in row.get("abstract", [])],
                "structured": None if marked is None else bool(marked),
            }
    missing = sorted(str(doc) for doc in wanted_docs if doc not in abstracts)
    if missing:
        raise DataError(
            f"corpus has no abstract for {len(missing)} document(s): {', '.join(missing[:5])}"
        )
    return abstracts


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #
def _index(rows: list[dict[str, Any]], code: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        claim_id = str(row["id"])
        if claim_id in indexed:
            raise DataError(f"{code} has two rows for id {claim_id}")
        indexed[claim_id] = row
    return indexed


def build(root: Path, *, out_path: Path | None = None) -> dict[str, Any]:
    root = Path(root).resolve()
    conditions_dir = root / "data" / "conditions"
    by_code = {
        code: _index(read_jsonl(conditions_dir / f"{code}.jsonl"), code) for code in CODES
    }
    gold = {str(row["id"]): row for row in read_jsonl(root / "data" / "gold.jsonl")}
    key = {str(row["id"]): row for row in read_jsonl(root / "data" / "key.jsonl")}
    screen = {
        str(row["id"]): row
        for row in read_jsonl(conditions_dir / "m1_overlap_screen.jsonl")
    }
    templates = json.loads(
        (root / "prompts" / "m2_templates.json").read_text(encoding="utf-8")
    )

    all_ids = set(gold)
    if set(key) != all_ids:
        raise DataError("data/key.jsonl and data/gold.jsonl do not cover the same ids")
    for code in FULL_TIER:
        if set(by_code[code]) != all_ids:
            raise DataError(f"{code} does not cover every claim id")
    directional = {cid for cid, row in gold.items() if row["label"] != "NEI"}
    for code in DIRECTIONAL_TIER:
        if set(by_code[code]) != directional:
            raise DataError(f"{code} must cover exactly the SUPPORT/REFUTE claims")
    if set(by_code["C0p"]) != set(by_code["C0"]):
        raise DataError("C0p does not mirror C0")

    wanted_docs = {
        key[cid]["doc_id"] for cid in all_ids if cid in key and "doc_id" in key[cid]
    }
    abstracts = load_abstracts(corpus_path_for(root), wanted_docs)
    missing_screen = directional - set(screen)
    if missing_screen:
        raise DataError(
            f"m1_overlap_screen.jsonl misses {len(missing_screen)} directional claim(s)"
        )

    rows: list[dict[str, Any]] = []

    def base_row(claim_id: str, code: str, evidence: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "id": claim_id,
            "condition": code,
            "n_sentences": len(evidence),
            "evidence_chars": len(render_evidence(evidence)),
        }

    for claim_id in all_ids:
        gold_idx = [int(index) for index in gold[claim_id].get("gold_idx", [])]

        for code in CODES:
            if code in DIRECTIONAL_TIER and claim_id not in directional:
                continue
            evidence = by_code[code][claim_id].get("evidence_sentences") or []
            row = base_row(claim_id, code, evidence)

            if code in ("C0", "C0p", "C_closed"):
                pass
            elif code == "C1":
                claim = str(by_code[code][claim_id]["claim"])
                filler = " ".join(
                    str(sentence["text"]) for sentence in evidence if sentence.get("idx") is None
                )
                row["prefix5_overlap"] = bool(prefix5_tokens(filler) & prefix5_tokens(claim))
                row["entity_overlap"] = bool(entity_tokens(filler) & entity_tokens(claim))
            elif code in ("C2", "M2"):
                meta = by_code[code][claim_id].get("manipulation_meta") or {}
                inserted = [
                    sentence for sentence in evidence if sentence.get("idx") is None
                ]
                inserted_text = " ".join(str(s["text"]) for s in inserted)
                row["position"] = meta.get("position") or meta.get("m2_position")
                row["nearest_gold_distance"] = nearest_gold_distance(evidence, gold_idx)
                row["instruction_words"] = len(inserted_text.split())
                row["instruction_chars"] = len(inserted_text)
                if code == "C2":
                    claim = str(by_code[code][claim_id]["claim"])
                    row["prefix5_overlap"] = bool(
                        prefix5_tokens(inserted_text) & prefix5_tokens(claim)
                    )
                    row["entity_overlap"] = bool(
                        entity_tokens(inserted_text) & entity_tokens(claim)
                    )
                else:
                    family = meta.get("family")
                    template_index = meta.get("template_index")
                    if family is None or template_index is None:
                        raise DataError(f"M2 meta incomplete for {claim_id}")
                    if family not in templates or not 0 <= int(template_index) < len(
                        templates[family]
                    ):
                        raise DataError(f"M2 template {family}[{template_index}] does not exist")
                    template = str(templates[family][int(template_index)])
                    number = stated_confidence(template)
                    row["family"] = family
                    row["template_index"] = int(template_index)
                    row["states_confidence_number"] = number is not None
                    row["stated_number"] = number
            elif code == "M1":
                c0_texts = [
                    str(sentence["text"])
                    for sentence in (by_code["C0"][claim_id].get("evidence_sentences") or [])
                ]
                m1_texts = [str(sentence["text"]) for sentence in evidence]
                doc_id = key[claim_id]["doc_id"]
                abstract = abstracts[doc_id]
                annotated = abstract["structured"]
                row["structured"] = (
                    bool(annotated)
                    if annotated is not None
                    else any(
                        heading_token(sentence) for sentence in abstract["sentences"]
                    )
                )
                row["lost_headers"] = len(
                    heading_tokens(c0_texts) - heading_tokens(m1_texts)
                )
                removed = {
                    int(index)
                    for index in (by_code[code][claim_id].get("manipulation_meta") or {}).get(
                        "removed_idx", []
                    )
                }
                row["dangling_anaphora"] = sorted(
                    int(sentence["idx"])
                    for sentence in evidence
                    if int(sentence["idx"]) - 1 in removed
                    and ANAPHOR.match(str(sentence["text"]).strip())
                )
                row["n_remaining"] = len(evidence)
                row["overlap_flag"] = bool(screen[claim_id]["flagged"])
            else:  # pragma: no cover - every code is handled above
                raise DataError(f"unhandled condition {code!r}")
            rows.append(row)

    rows.sort(key=lambda row: (row["id"], CODES.index(row["condition"])))
    target = Path(out_path) if out_path is not None else conditions_dir / "flags.jsonl"
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows
    )
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(payload + "\n")

    counts: dict[str, int] = {code: 0 for code in CODES}
    for row in rows:
        counts[row["condition"]] += 1
    return {
        "rows": len(rows),
        "per_condition": counts,
        "claims": len(all_ids),
        "output": str(target),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=_ROOT)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    try:
        summary = build(args.root, out_path=args.out)
    except DataError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
