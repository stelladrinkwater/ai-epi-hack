"""Build the M1 / C1 / C2 manual audit sheets.

No model is called. The sheets shown to the coder contain only the claim and
the evidence being judged; ids, labels, flags, removed gold text and model
output live only in audit/key.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
for extra in (_ROOT, _ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from common import load_config, read_jsonl  # noqa: E402

DEFAULT_OUT = Path("audit")
M1_EXTRA = 12
FILLER_PER_CONDITION = 12

CODING_RULE = """strong = the remaining text alone would make a careful reader say SUPPORTED or REFUTED;
weak = there is a hint but not enough to decide; none = nothing relevant.
topically_related = yes if the inserted sentence concerns the same disease, gene, intervention or
outcome as the claim. Code independently, then compare.
"""


def write_text_lf(path: Path, text: str) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def write_csv(path: Path, header: list[str], rows: list[list[Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def numbered(sentences: list[dict[str, Any]]) -> str:
    return "\n".join(f"{index}. {sentence['text']}" for index, sentence in enumerate(sentences, 1))


def build(config_path: Path, *, out_dir: Path | None = None) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    root = config_path.parent
    config = load_config(config_path)
    seed = int(config["audit"]["seed"])
    output = Path(out_dir) if out_dir is not None else root / DEFAULT_OUT

    conditions = root / "data" / "conditions"
    m1_rows = {row["id"]: row for row in read_jsonl(conditions / "M1.jsonl")}
    c1_rows = {row["id"]: row for row in read_jsonl(conditions / "C1.jsonl")}
    c2_rows = {row["id"]: row for row in read_jsonl(conditions / "C2.jsonl")}
    screen = read_jsonl(conditions / "m1_overlap_screen.jsonl")
    flagged = {row["id"] for row in screen if row["flagged"]}
    if not flagged <= set(m1_rows):
        raise ValueError("overlap screen names claims that are not in M1.jsonl")

    others = sorted(set(m1_rows) - flagged)
    extra = random.Random(f"{seed}|M1|extra").sample(others, M1_EXTRA)
    selected = sorted(flagged) + sorted(extra)
    random.Random(f"{seed}|M1|order").shuffle(selected)

    m1_lines = [
        "# M1 leftover-evidence audit",
        "",
        "Code `leftover_support` as none / weak / strong (see audit/README.md).",
        "Do not open audit/key.csv until this sheet is coded.",
        "",
    ]
    m1_csv: list[list[Any]] = []
    key_rows: list[list[Any]] = []
    for number, cid in enumerate(selected, 1):
        row = m1_rows[cid]
        evidence = numbered(row["evidence_sentences"])
        m1_csv.append([number, row["claim"], evidence, "", ""])
        m1_lines += [
            f"## Row {number}",
            "",
            f"**Claim:** {row['claim']}",
            "",
            "**Remaining evidence:**",
            "",
            evidence if evidence else "(none)",
            "",
            "**leftover_support:** ",
            "",
            "**notes:** ",
            "",
        ]
        key_rows.append(["m1_audit", number, cid, "M1", cid in flagged])

    c1_ids = sorted(c1_rows)
    c2_ids = sorted(c2_rows)
    c1_selected = random.Random(f"{seed}|C1").sample(c1_ids, FILLER_PER_CONDITION)
    c2_selected = random.Random(f"{seed}|C2").sample(c2_ids, FILLER_PER_CONDITION)

    c_lines = [
        "# C1 / C2 inserted-sentence audit",
        "",
        "Code `topically_related` as yes / no / unsure (see audit/README.md).",
        "Only the inserted sentence(s) are shown, not the original evidence.",
        "Do not open audit/key.csv until this sheet is coded.",
        "",
    ]
    c_csv: list[list[Any]] = []
    for condition, chosen in (("C1", c1_selected), ("C2", c2_selected)):
        for number, cid in enumerate(chosen, 1):
            row = (c1_rows if condition == "C1" else c2_rows)[cid]
            inserted = [s for s in row["evidence_sentences"] if s.get("idx") is None]
            sentences = numbered(inserted)
            c_csv.append([condition, number, condition, row["claim"], sentences, ""])
            c_lines += [
                f"## {condition} Row {number}",
                "",
                f"**Claim:** {row['claim']}",
                "",
                "**Inserted sentence(s):**",
                "",
                sentences if sentences else "(none)",
                "",
                "**topically_related:** ",
                "",
            ]
            key_rows.append(["c_filler_audit", number, cid, condition, "na"])

    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "m1_audit.csv", ["row", "claim", "evidence", "leftover_support", "notes"], m1_csv)
    write_text_lf(output / "m1_audit.md", "\n".join(m1_lines) + "\n")
    write_csv(
        output / "c_filler_audit.csv",
        ["sheet", "row", "condition", "claim", "inserted_sentences", "topically_related"],
        c_csv,
    )
    write_text_lf(output / "c_filler_audit.md", "\n".join(c_lines) + "\n")
    write_csv(output / "key.csv", ["sheet", "row", "id", "condition", "flagged"], key_rows)
    write_text_lf(output / "README.md", "# Audit coding rule\n\n" + CODING_RULE)

    return {
        "m1_rows": len(selected),
        "m1_flagged": len(flagged),
        "m1_extra": len(extra),
        "c1_rows": len(c1_selected),
        "c2_rows": len(c2_selected),
        "key_rows": len(key_rows),
        "outputs": sorted(str(path) for path in output.iterdir() if path.is_file()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).resolve().parents[1] / "config.yaml"
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    summary = build(args.config, out_dir=args.out)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
