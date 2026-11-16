"""Experiment runner for the DeepSeek matrix (SPEC §3, §5, §6, §7).

Dry-run by default. A real run needs --confirm and every frozen-state guard.
The runner never prints prompts, keys or ids, never deletes a log, and writes
only to results/raw/ and the configured cache directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable

_ROOT = Path(__file__).resolve().parents[1]
for extra in (_ROOT / "scripts", _ROOT / "src"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from backend import DeepSeekBackend  # noqa: E402
from common import DataError, read_jsonl  # noqa: E402
from escaping import escape_tags  # noqa: E402
from prompts import PromptBundle, PromptError, render_evidence, render_user  # noqa: E402
from verifier_client import (  # noqa: E402
    ConfigError,
    RunAborted,
    StateError,
    VerifierClient,
    api_key_from_env,
    load_cache_dir,
    load_model_config,
    load_prompts,
)

NAIVE_SYSTEM = "verifier_system_naive.txt"
LOG_NAMES = {
    "core": "core_warned.jsonl",
    "extended": "extended_warned.jsonl",
    "naive": "naive.jsonl",
}
PROGRESS_EVERY = 50
WINDOW = 100
MAX_API_ERRORS_IN_WINDOW = 20


class GuardError(RuntimeError):
    """A frozen-state guard failed; a real run must not start."""


# --------------------------------------------------------------------------- #
# Config and units
# --------------------------------------------------------------------------- #
def run_config(config: dict[str, Any]) -> dict[str, Any]:
    section = config.get("run")
    if not isinstance(section, dict):
        raise ConfigError("config key 'run' is missing")
    for key in ("conditions_core", "conditions_extended", "k", "order_seed", "tag_required", "shards"):
        if key not in section:
            raise ConfigError(f"config key 'run.{key}' is missing")
    if not isinstance(section["k"], int) or section["k"] < 1:
        raise ConfigError("config key 'run.k' must be a positive integer")
    if not isinstance(section["shards"], int) or section["shards"] < 1:
        raise ConfigError("config key 'run.shards' must be a positive integer")
    for key in ("conditions_core", "conditions_extended"):
        if not isinstance(section[key], list) or not all(isinstance(c, str) for c in section[key]):
            raise ConfigError(f"config key 'run.{key}' must be a list of condition codes")
    return section


def resolve_conditions(config: dict[str, Any], tier: str, override: list[str] | None = None) -> list[str]:
    section = run_config(config)
    if tier == "core":
        conditions = list(section["conditions_core"])
    elif tier == "extended":
        conditions = list(section["conditions_extended"])
    elif tier == "naive":
        conditions = ["C0", "M2"]
    else:
        raise ConfigError(f"unknown tier {tier!r}")
    if override is not None:
        if tier == "naive" and set(override) != {"C0", "M2"}:
            raise ConfigError("naive tier runs only C0 and M2")
        conditions = list(override)
    return conditions


def condition_path(root: Path, code: str) -> Path:
    base = root / "data" / "conditions"
    return base / ("C0.jsonl" if code == "C0p" else f"{code}.jsonl")


def build_units(root: Path, config: dict[str, Any], tier: str) -> list[dict[str, Any]]:
    k = int(run_config(config)["k"])
    units: list[dict[str, Any]] = []
    for code in resolve_conditions(config, tier):
        rows = read_jsonl(condition_path(root, code))
        for row in rows:
            evidence = row["evidence_sentences"]
            for sample_idx in range(k):
                units.append(
                    {
                        "claim_id": row["id"],
                        "claim": row["claim"],
                        "evidence_sentences": evidence,
                        "sample_idx": sample_idx,
                        "condition": code if code != "C0p" else "C0p",
                    }
                )
    return units


def ordered_units(units: list[dict[str, Any]], order_seed: int) -> list[dict[str, Any]]:
    if isinstance(order_seed, bool) or not isinstance(order_seed, int):
        raise ConfigError("run.order_seed must be an integer before ordering units")
    ordered = [dict(unit) for unit in units]
    random.Random(order_seed).shuffle(ordered)
    return ordered


def shard_slice(units: list[dict[str, Any]], index: int, count: int) -> list[dict[str, Any]]:
    if count < 1 or not 1 <= index <= count:
        raise ConfigError(f"invalid shard {index}/{count}")
    start = (index - 1) * len(units) // count
    end = index * len(units) // count
    return units[start:end]


def parse_shard(text: str | None, default_count: int) -> tuple[int, int]:
    if text is None:
        return 1, default_count
    match = re.fullmatch(r"(\d+)/(\d+)", text.strip())
    if not match:
        raise ConfigError(f"invalid --shard {text!r}; expected i/n")
    return int(match.group(1)), int(match.group(2))


# --------------------------------------------------------------------------- #
# Prompts and planning
# --------------------------------------------------------------------------- #
def prompt_bundle(root: Path, tier: str) -> tuple[PromptBundle, bool]:
    prompts_dir = root / "prompts"
    if tier != "naive":
        return load_prompts(prompts_dir), False
    system_path = prompts_dir / NAIVE_SYSTEM
    user_path = prompts_dir / "verifier_user.txt"
    if not user_path.is_file():
        raise PromptError(f"missing prompt file: {user_path}")
    user_bytes = user_path.read_bytes()
    user_template = user_bytes.decode("utf-8")
    if not system_path.is_file():
        return (
            PromptBundle(
                system="",
                user_template=user_template,
                system_sha256="",
                user_sha256=hashlib.sha256(user_bytes).hexdigest(),
            ),
            True,
        )
    system_bytes = system_path.read_bytes()
    return (
        PromptBundle(
            system=system_bytes.decode("utf-8"),
            user_template=user_template,
            system_sha256=hashlib.sha256(system_bytes).hexdigest(),
            user_sha256=hashlib.sha256(user_bytes).hexdigest(),
        ),
        False,
    )


def results_dir_for(root: Path) -> Path:
    """``run.results_dir`` resolved against the config file (SPEC v1.5 §7)."""
    config_path = Path(root) / "config.yaml"
    fallback = Path(root) / "results"
    if not config_path.is_file():
        return fallback
    config = load_config_for(config_path)
    value = (config.get("run") or {}).get("results_dir")
    if not isinstance(value, str) or not value:
        return fallback
    path = Path(value)
    return path if path.is_absolute() else config_path.resolve().parent / path


def log_path_for(root: Path, tier: str) -> Path:
    """One log per tier and prompt variant, under ``run.results_dir``."""
    if tier not in LOG_NAMES:
        raise ConfigError(f"unknown tier {tier!r}")
    return results_dir_for(root) / "raw" / LOG_NAMES[tier]


def cache_dir_for(root: Path, config_path: Path, tier: str) -> Path:
    return load_cache_dir(config_path) / tier


def prompt_chars(units: Iterable[dict[str, Any]], bundle: PromptBundle) -> int:
    total = 0
    for unit in units:
        claim_text, _ = escape_tags(unit["claim"])
        evidence_text, _ = escape_tags(render_evidence(unit["evidence_sentences"]))
        prompt = render_user(bundle.user_template, claim_text, evidence_text)
        total += len(bundle.system) + len(prompt)
    return total


def done_keys(log_path: Path) -> set[tuple[str, str, int]]:
    done: set[tuple[str, str, int]] = set()
    if not log_path.is_file():
        return done
    with log_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if row.get("is_result"):
                done.add((str(row["claim_id"]), str(row["condition"]), int(row["sample_idx"])))
    return done


def dry_run(root: Path, config_path: Path, tier: str, shard: str | None = None) -> dict[str, Any]:
    config = load_config_for(config_path)
    section = run_config(config)
    units = build_units(root, config, tier)
    order_seed = section["order_seed"]
    index, count = parse_shard(shard, section["shards"])
    selected = units
    if order_seed is not None:
        selected = shard_slice(ordered_units(units, order_seed), index, count)
    bundle, missing_naive = prompt_bundle(root, tier)
    log_path = log_path_for(root, tier)
    per_condition: dict[str, int] = {}
    for unit in units:
        per_condition[unit["condition"]] = per_condition.get(unit["condition"], 0) + 1
    done = done_keys(log_path)
    already = sum(
        1 for unit in selected if (unit["claim_id"], unit["condition"], unit["sample_idx"]) in done
    )
    return {
        "tier": tier,
        "shard": f"{index}/{count}",
        "units_per_condition": per_condition,
        "total_calls": len(units),
        "shard_calls": len(selected),
        "already_done": already,
        "prompt_chars": None if missing_naive else prompt_chars(units, bundle),
        "missing_naive_prompt": missing_naive,
        "log": str(log_path),
        "cache_dir": str(cache_dir_for(root, config_path, tier)),
    }


def load_config_for(config_path: Path) -> dict[str, Any]:
    from common import load_config

    return load_config(config_path)


# --------------------------------------------------------------------------- #
# Frozen-state guards
# --------------------------------------------------------------------------- #
def git_tree_clean(root: Path) -> bool:
    result = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain"],
        capture_output=True, text=True,
    )
    return result.returncode == 0 and not result.stdout.strip()


def tag_is_ancestor(root: Path, tag: str) -> bool:
    exists = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "--verify", tag],
        capture_output=True, text=True,
    )
    if exists.returncode != 0:
        return False
    ancestor = subprocess.run(
        ["git", "-C", str(root), "merge-base", "--is-ancestor", tag, "HEAD"],
        capture_output=True, text=True,
    )
    return ancestor.returncode == 0


SPEC_HASH_RE = re.compile(r"SPEC SHA-256: `([0-9a-f]{64})`")


def latest_spec_hash(changelog_text: str) -> str | None:
    """The SPEC hash of the newest dated entry, wherever it sits.

    Changelogs may be newest-first or newest-last; the date on the ``##``
    heading decides. On a tie (more than one entry on the newest date) the
    first entry in file order wins, which is the top of the file in both
    conventions.
    """
    newest_date: str | None = None
    newest_hash: str | None = None
    for section in re.split(r"(?m)^## ", changelog_text)[1:]:
        heading, _, body = section.partition("\n")
        match = re.match(r"(\d{4}-\d{2}-\d{2})", heading.strip())
        if not match:
            continue
        hashes = SPEC_HASH_RE.findall(body)
        if not hashes:
            continue
        if newest_date is None or match.group(1) > newest_date:
            newest_date = match.group(1)
            newest_hash = hashes[-1]
    return newest_hash


def spec_hash_matches(root: Path) -> bool:
    changelog = root / "CHANGELOG.md"
    if not changelog.is_file():
        return False
    expected = latest_spec_hash(changelog.read_text(encoding="utf-8"))
    actual = hashlib.sha256((root / "SPEC.md").read_bytes()).hexdigest()
    return expected == actual


def read_hash_file(path: Path) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.replace("\r", "").strip()
        if not line:
            continue
        digest, relative = line.split(None, 1)
        rows.append((digest.strip(), relative.strip()))
    return rows


def hash_file_matches(root: Path, path: Path) -> bool:
    """Every listed file must match, and the list must cover its directory.

    The manifest covers every file beside it except itself, so deleting a row
    (or adding a file the manifest does not name) fails the guard.
    """
    if not path.is_file():
        return False
    rows = read_hash_file(path)
    if not rows:
        return False
    covered: set[str] = set()
    for digest, relative in rows:
        target = root / relative
        if not target.is_file():
            return False
        if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
            return False
        covered.add(Path(relative).name)
    siblings = {
        child.name
        for child in path.parent.iterdir()
        if child.is_file() and child.name != path.name
    }
    return covered == siblings


def hash_files_match(root: Path) -> bool:
    return hash_file_matches(root, root / "data" / "conditions" / "HASHES.txt") and hash_file_matches(
        root, root / "data" / "HASHES.txt"
    )


ANALYSIS_HASH = "ANALYSIS.HASH"
ANALYSIS_DIRS = ("analysis", "src", "prompts")
ANALYSIS_FILES = (
    "config.yaml",
    "scripts/run_experiment.py",
    "scripts/run_analysis.py",
    "scripts/common.py",
    "scripts/make_flags.py",
)


def analysis_hash_targets(root: Path) -> list[str]:
    """The frozen analysis surface that exists (SPEC v1.5 §7).

    Fixed paths count only when present so a minimal frozen fixture still
    passes; a listed file that disappears afterwards still fails
    ``analysis_hash_matches`` through its ANALYSIS.HASH row.
    """
    targets = [relative for relative in ANALYSIS_FILES if (root / relative).is_file()]
    for directory in ANALYSIS_DIRS:
        for path in sorted((root / directory).rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                targets.append(path.relative_to(root).as_posix())
    return targets


def analysis_hash_matches(root: Path) -> bool:
    """ANALYSIS.HASH covers the frozen surface and every digest still holds."""
    manifest = root / ANALYSIS_HASH
    if not manifest.is_file():
        return False
    listed = {relative: digest for digest, relative in read_hash_file(manifest)}
    if not listed:
        return False
    for relative, digest in listed.items():
        target = root / relative
        if not target.is_file():
            return False
        if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
            return False
    return all(target in listed for target in analysis_hash_targets(root))


def prompt_hashes_match(root: Path) -> bool:
    changelog = (root / "CHANGELOG.md").read_text(encoding="utf-8")
    pairs = re.findall(r"`prompts/([^`]+)`\s*\|\s*`([0-9a-f]{64})`", changelog)
    if not pairs:
        return False
    for name, digest in pairs:
        path = root / "prompts" / name
        if not path.is_file():
            return False
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            return False
    return True


def guard_report(root: Path, config: dict[str, Any]) -> list[tuple[str, bool, str]]:
    """One (name, passed, detail) row per frozen-state guard."""
    tag = run_config(config)["tag_required"]
    report: list[tuple[str, bool, str]] = []

    clean = git_tree_clean(root)
    report.append(
        ("git tree is not clean", clean,
         "clean" if clean else "uncommitted changes, or git is unavailable")
    )
    ancestor = tag_is_ancestor(root, tag)
    report.append(
        (f"tag {tag!r} is missing or not an ancestor of HEAD", ancestor,
         "tag reaches HEAD" if ancestor else f"tag {tag!r} does not reach HEAD")
    )
    spec_ok = spec_hash_matches(root)
    report.append(
        ("SPEC.md hash does not match the latest CHANGELOG value", spec_ok,
         "SPEC.md matches the newest CHANGELOG entry" if spec_ok
         else "SPEC.md differs from the newest CHANGELOG SPEC SHA-256")
    )
    data_ok = hash_files_match(root)
    report.append(
        ("HASHES.txt files do not match their files", data_ok,
         "both data manifests cover their directories" if data_ok
         else "a HASHES.txt row is stale, missing or does not cover its directory")
    )
    analysis_ok = analysis_hash_matches(root)
    report.append(
        ("ANALYSIS.HASH does not match the frozen analysis surface", analysis_ok,
         "analysis surface matches ANALYSIS.HASH" if analysis_ok
         else "ANALYSIS.HASH is missing, stale or incomplete")
    )
    prompts_ok = prompt_hashes_match(root)
    report.append(
        ("prompt hashes do not match CHANGELOG", prompts_ok,
         "prompt hashes match" if prompts_ok
         else "a prompts/ digest differs from CHANGELOG")
    )
    return report


def assert_guards(root: Path, config: dict[str, Any]) -> None:
    failures = [
        f"{name}: {detail}" for name, ok, detail in guard_report(root, config) if not ok
    ]
    if failures:
        raise GuardError("; ".join(failures))


# --------------------------------------------------------------------------- #
# Execution
# --------------------------------------------------------------------------- #
def api_error_window_abort(result_rows: list[dict[str, Any]]) -> bool:
    outcomes = [row.get("outcome") for row in result_rows]
    for start in range(0, max(0, len(outcomes) - WINDOW + 1)):
        window = outcomes[start : start + WINDOW]
        if sum(1 for outcome in window if outcome == "api_error") > MAX_API_ERRORS_IN_WINDOW:
            return True
    return False


def execute(
    client: VerifierClient,
    units: list[dict[str, Any]],
    *,
    order_seed: int,
    resume_passes: int = 2,
    progress_every: int = PROGRESS_EVERY,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    started = time.monotonic()
    completed_count = 0

    def report_unit_complete(row: dict[str, Any]) -> None:
        nonlocal completed_count
        completed_count += 1
        elapsed = time.monotonic() - started
        fraction = completed_count / len(units) if units else 1.0
        estimated_total = elapsed / fraction if fraction else 0.0
        duration_seconds = (
            0.0 if row.get("cached") else
            row.get("_unit_elapsed_seconds")
        )
        print(
            json.dumps({
                "event": "unit_complete",
                "finished": completed_count,
                "total": len(units),
                "percent_done": round(100.0 * fraction, 1),
                "call_seconds": duration_seconds,
                "elapsed_seconds": round(elapsed, 1),
                "estimated_total_seconds": round(estimated_total, 1),
                "estimated_remaining_seconds": round(max(0.0, estimated_total - elapsed), 1),
                "outcome": row.get("outcome"),
                "cached": bool(row.get("cached")),
            }),
            file=sys.stderr,
            flush=True,
        )

    for offset in range(0, len(units), progress_every):
        chunk = units[offset : offset + progress_every]
        batch = client.run_batch(
            chunk,
            order_seed=order_seed,
            resume_passes=resume_passes,
            on_unit_complete=report_unit_complete,
        )
        rows.extend(batch)
        results.extend(row for row in batch if row.get("is_result"))
        if api_error_window_abort(results):
            raise RunAborted(0, "more than 20 api_error rows in 100 consecutive results")
        counts: dict[str, int] = {}
        finishes: dict[str, int] = {}
        for row in batch:
            counts[row["outcome"]] = counts.get(row["outcome"], 0) + 1
            finishes[str(row.get("finish_reason"))] = finishes.get(str(row.get("finish_reason")), 0) + 1
        print(
            json.dumps(
                {"processed": min(offset + len(chunk), len(units)), "outcomes": counts, "finish": finishes}
            ),
            file=sys.stderr,
        )
    return rows


def check_guards(root: Path, config: dict[str, Any]) -> int:
    """Print each guard PASS/FAIL with its reason; build no API client."""
    failures = 0
    for name, ok, detail in guard_report(root, config):
        print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}")
        failures += 0 if ok else 1
    print("all guards passed" if failures == 0 else f"{failures} guard(s) failed")
    return 0 if failures == 0 else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", choices=("core", "extended", "naive"), default="core")
    parser.add_argument("--shard", default=None, help="i/n; default is shard 1/n")
    parser.add_argument("--confirm", action="store_true", help="execute a real run")
    parser.add_argument(
        "--check-guards", action="store_true",
        help="print each frozen-state guard PASS/FAIL with its reason and exit",
    )
    parser.add_argument("--config", type=Path, default=_ROOT / "config.yaml")
    args = parser.parse_args(argv)

    config_path = args.config.resolve()
    root = config_path.parent
    try:
        config = load_config_for(config_path)
        section = run_config(config)
        if args.check_guards:
            return check_guards(root, config)
        index, count = parse_shard(args.shard, section["shards"])
        if not args.confirm:
            print(json.dumps(dry_run(root, config_path, args.tier, args.shard), indent=2))
            return 0
        if section["order_seed"] is None:
            raise ConfigError("config key 'run.order_seed' is null; set it before a real run")
        assert_guards(root, config)
        units = ordered_units(build_units(root, config, args.tier), int(section["order_seed"]))
        shard = shard_slice(units, index, count)
        bundle, missing_naive = prompt_bundle(root, args.tier)
        if missing_naive:
            raise ConfigError(f"missing prompt file: {root / 'prompts' / NAIVE_SYSTEM}")
        model = load_model_config(config_path)
        backend = DeepSeekBackend(
            base_url=model.base_url,
            api_key=api_key_from_env(model),
            model=model.model,
        )
        client = VerifierClient(
            model,
            backend,
            bundle,
            log_path_for(root, args.tier),
            cache_dir=cache_dir_for(root, config_path, args.tier),
        )
        rows = execute(client, shard, order_seed=int(section["order_seed"]))
        print(json.dumps({"tier": args.tier, "shard": f"{index}/{count}", "rows": len(rows)}))
        if count > 1:
            print("health shard complete; later shards need --shard", file=sys.stderr)
        return 0
    except (ConfigError, DataError, GuardError, StateError, RunAborted, PromptError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
