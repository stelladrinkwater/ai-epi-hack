"""Write the one markdown analysis summary plus all CSVs (LF)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from analysis import contrasts as contrasts_mod
from analysis import strata as strata_mod


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_summary(
    contrast_rows: list[dict[str, Any]],
    strata_data: dict[str, Any],
    out_dir: Path,
    *,
    seed: int | None = None,
    n_resamples: int | None = None,
    commit: str | None = None,
    logs: dict[str, str] | None = None,
) -> Path:
    out = Path(out_dir)
    lines = ["# Analysis summary", ""]
    if seed is not None or n_resamples is not None or commit or logs:
        lines += ["## Provenance", ""]
        if commit:
            lines.append(f"- commit: {commit}")
        if seed is not None:
            lines.append(f"- bootstrap seed: {seed}")
        if n_resamples is not None:
            lines.append(f"- resamples: {n_resamples:,}")
        for tier, path in (logs or {}).items():
            lines.append(f"- {tier} log: {path}")
        lines.append("")
    lines += ["## Contrasts", ""]
    lines += ["| contrast | class | estimate | 95% CI | p | p_holm | n_claims | method |", "|---|---|---|---|---|---|---|---|"]
    for row in contrast_rows:
        ci = (
            f"[{_fmt(row['ci_low'])}, {_fmt(row['ci_high'])}]"
            if row.get("ci_low") is not None
            else "-"
        )
        lines.append(
            f"| {row['contrast']} | {row['class']} | {_fmt(row['estimate'])} | {ci} | "
            f"{_fmt(row['p_value'])} | {_fmt(row['p_holm'])} | {row['n_claims']} | {row['method']} |"
        )
    lines += ["", "## M1 strata", ""]
    lines += ["| descriptor | value | claims | samples | not_as_expected | rate |", "|---|---|---|---|---|---|"]
    for row in strata_data.get("m1", []):
        lines.append(
            f"| {row['descriptor']} | {row['value']} | {row['claims']} | {row['samples']} | "
            f"{row['not_as_expected']} | {_fmt(row['rate'])} |"
        )
    if strata_data.get("audit_available") is False:
        lines += ["", "_M1 audit stratum omitted: audit/coded.csv is missing._"]
    lines += ["", "## M2 strata", ""]
    lines += ["| descriptor | value | claims | samples | not_as_expected | rate |", "|---|---|---|---|---|---|"]
    for row in strata_data.get("m2", []):
        lines.append(
            f"| {row['descriptor']} | {row['value']} | {row['claims']} | {row['samples']} | "
            f"{row['not_as_expected']} | {_fmt(row['rate'])} |"
        )
    lines += ["", "## Followed-injection confidence", ""]
    for row in strata_data.get("injection", []):
        lines.append(
            f"- {row['group']}: {row['followed']} followed, "
            f"{row['matched_stated_confidence']} matched ({_fmt(row['share'])})"
        )
    lines += ["", "## Stability vs mean confidence (Spearman, exploratory)", ""]
    for row in strata_data.get("spearman", []):
        lines.append(
            f"- {row['manipulation']}: rho={_fmt(row['spearman'])}, "
            f"stability==1.0 share={_fmt(row['degenerate_stability_share'])}, "
            f"degenerate={row['degenerate']}"
        )
    lines.append("")
    path = out / "summary.md"
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines))
    return path


def write_all(
    samples: list[dict[str, Any]],
    seed: int,
    out_dir: Path,
    *,
    audit_dir: Path | None = None,
    n_resamples: int = contrasts_mod.N_RESAMPLES,
) -> dict[str, Any]:
    contrast_rows = contrasts_mod.run_contrasts(samples, seed, n_resamples=n_resamples)
    contrasts_mod.write_contrasts(contrast_rows, out_dir)
    strata_data = strata_mod.write_strata(samples, out_dir, audit_dir=audit_dir)
    write_summary(contrast_rows, strata_data, out_dir)
    return {"contrasts": contrast_rows, "strata": strata_data}
