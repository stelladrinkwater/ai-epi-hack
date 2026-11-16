"""Review-1 adversarial findings.

One contract test per finding id (``test_r<N>_...``), written against the
contract the reviewer and the human agreed on, not against the predicate the
current code happens to use. Each is expected to be RED at tag ``review-1``.
Beside each one is a guard test (``test_r<N>_guard_...``) that is GREEN today
and pins the neighbouring behaviour the fix must not break.

Everything is written inside ``tmp_path``. The only touches of the real
repository are ``git check-ignore``, reading ``data/gold.jsonl`` and hashing
tracked files -- none of which modify it.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_experiment as rx  # noqa: E402
import verifier_client as vc  # noqa: E402
from analysis import contrasts, loaders, tables  # noqa: E402
from backend import BackendError, BackendResult, RateLimitError  # noqa: E402
from prompts import PromptBundle  # noqa: E402

VALID = '{"verdict": "SUPPORTED", "confidence": 80, "rationale": "r", "evidence_concerns": []}'
GOLD_VERDICT = {"SUPPORT": "SUPPORTED", "REFUTE": "REFUTED", "NEI": "NOT_ENOUGH_INFO"}
PRIMARY_CONTRASTS = {"M1_vs_C0p", "M2_vs_C2"}
CLASS_NAMES = {"SUPPORT", "REFUTE", "NEI"}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def git(root: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)


def git_commit_all(root: Path, message: str) -> None:
    assert git(root, "add", "-A").returncode == 0
    done = git(root, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", message)
    assert done.returncode == 0, done.stderr


def git_init(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    assert subprocess.run(
        ["git", "init", "-q", str(root)], capture_output=True, text=True
    ).returncode == 0


def sample(
    claim_id: str,
    manipulation: str,
    sample_idx: int,
    verdict: str | None,
    expected: str,
    *,
    confidence: int | None = 90,
    bin_name: str | None = "high",
    class_name: str = "SUPPORT",
    outcome: str = "ok",
) -> dict:
    as_expected = None if verdict is None else verdict == expected
    if verdict is None:
        outcome_column = "no_verdict"
    elif as_expected:
        outcome_column = "as_expected"
    else:
        outcome_column = "not_as_expected"
    return {
        "manipulation": manipulation,
        "claim_id": claim_id,
        "class": class_name,
        "sample_idx": sample_idx,
        "model_verdict": verdict,
        "expected_verdict": expected,
        "as_expected": as_expected,
        "confidence": confidence,
        "bin": bin_name,
        "outcome_column": outcome_column,
        "outcome": outcome,
        "finish_reason": "stop",
        "raw_response": None,
        "manipulation_meta": {},
        "c0_majority": None,
        "flags": {},
    }


class MockBackend:
    """``responses`` items: a str, an Exception, or a (text, finish_reason) tuple."""

    timeout = 120.0

    def __init__(self, responses: list, *, cycle: object = None):
        self.responses = list(responses)
        self.cycle = cycle
        self.calls = 0

    def complete(self, system, user, **kwargs):
        self.calls += 1
        item = self.cycle if not self.responses else self.responses.pop(0)
        if item is None:
            raise AssertionError("mock backend ran out of responses")
        if isinstance(item, Exception):
            raise item
        text, finish_reason = item if isinstance(item, tuple) else (item, "stop")
        return BackendResult(
            text=text, input_tokens=10, output_tokens=5, latency_ms=1.0,
            finish_reason=finish_reason, http_status=200,
        )


def model_config(**overrides) -> vc.ModelConfig:
    section = {
        "base_url": "https://api.example.test",
        "model": "deepseek-test",
        "api_key_env": "DEEPSEEK_API_KEY",
        "temperature": 0.7,
        "top_p": None,
        "max_tokens": 64,
        "thinking": {"enabled": False, "budget_tokens": None},
        "concurrency": 1,
        "max_retries": 2,
        "retry_on": ["api_error"],
        "blocked_body_markers": ["content exists risk"],
        "price_table": None,
    }
    section.update(overrides)
    return vc.ModelConfig.from_dict(section)


def bundle() -> PromptBundle:
    return PromptBundle(
        system="SYS", user_template="claim: {claim}\nevidence: {evidence}",
        system_sha256=sha256_text("SYS"), user_sha256=sha256_text("tmpl"),
    )


def make_client(tmp_path: Path, responses: list, *, cycle=None, sleep=None, **overrides):
    sleeps: list[float] = []
    backend = MockBackend(responses, cycle=cycle)
    client = vc.VerifierClient(
        model_config(**overrides), backend, bundle(),
        tmp_path / "log.jsonl", cache_dir=tmp_path / "cache",
        sleep=sleep or sleeps.append,
    )
    return client, backend, sleeps


def unit(claim_id: str, condition: str = "C0", sample_idx: int = 0) -> dict:
    return {
        "claim_id": claim_id,
        "claim": f"claim {claim_id}",
        "evidence_sentences": [{"idx": 0, "text": f"evidence for {claim_id}"}],
        "sample_idx": sample_idx,
        "condition": condition,
    }


def log_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def real_claims(per_class: int = 12) -> dict[str, list[str]]:
    """A few real claim ids per gold class, read from the frozen gold file."""
    picked: dict[str, list[str]] = {"SUPPORT": [], "REFUTE": [], "NEI": []}
    for line in (ROOT / "data" / "gold.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if len(picked[row["label"]]) < per_class:
            picked[row["label"]].append(row["id"])
    return picked


def synthetic_log(path: Path, *, short_cell: tuple[str, str] | None = None) -> Path:
    """A complete core-tier log over real claim ids, 5 result rows per cell."""
    claims = real_claims()
    rows: list[dict] = []
    for label, ids in claims.items():
        for position, claim_id in enumerate(ids):
            conditions = ["C0", "C0p", "C2", "M2"] + ([] if label == "NEI" else ["M1"])
            for condition in conditions:
                expected = "NOT_ENOUGH_INFO" if condition == "M1" else GOLD_VERDICT[label]
                k = 5
                if short_cell == (claim_id, condition):
                    k = 4
                for sample_idx in range(k):
                    # mostly expected, so the C0 filter keeps the claim
                    wrong = sample_idx == 4 and position % 3 == 0
                    verdict = GOLD_VERDICT[label] if condition != "M1" else (
                        expected if not wrong else GOLD_VERDICT[label]
                    )
                    if condition != "M1" and wrong:
                        verdict = "SUPPORTED" if verdict != "SUPPORTED" else "REFUTED"
                    rows.append({
                        "claim_id": claim_id,
                        "condition": condition,
                        "sample_idx": sample_idx,
                        "is_result": True,
                        "outcome": "ok",
                        "finish_reason": "stop",
                        "verdict": verdict,
                        "confidence": 60 + (position * 7 + sample_idx * 11) % 41,
                        "raw_response": VALID,
                        "cache_key": sha256_text(f"{claim_id}|{condition}|{sample_idx}"),
                    })
    write_jsonl(path, rows)
    return path


@pytest.fixture
def frozen_repo(tmp_path: Path) -> dict:
    """A minimal repository in which all five current freeze guards pass."""
    root = tmp_path / "frozen"
    (root / "prompts").mkdir(parents=True)
    (root / "analysis").mkdir()
    (root / "src").mkdir()
    (root / "scripts").mkdir()
    (root / "data" / "conditions").mkdir(parents=True)

    spec = "# SPEC stub\n\nsection 5, 6 and 7 live here.\n"
    (root / "SPEC.md").write_text(spec, encoding="utf-8", newline="\n")
    system_prompt = "you are a verifier\n"
    (root / "prompts" / "verifier_system.txt").write_text(system_prompt, encoding="utf-8", newline="\n")
    (root / "config.yaml").write_text("run:\n  k: 5\n", encoding="utf-8", newline="\n")
    (root / "analysis" / "contrasts.py").write_text("N_RESAMPLES = 10000\n", encoding="utf-8", newline="\n")
    (root / "src" / "verifier_client.py").write_text("CLIENT = 1\n", encoding="utf-8", newline="\n")
    (root / "scripts" / "run_experiment.py").write_text("RUNNER = 1\n", encoding="utf-8", newline="\n")

    gold = '{"id":"c_1","label":"SUPPORT"}\n'
    (root / "data" / "gold.jsonl").write_text(gold, encoding="utf-8", newline="\n")
    c0 = '{"id":"c_1","claim":"x","evidence_sentences":[]}\n'
    (root / "data" / "conditions" / "C0.jsonl").write_text(c0, encoding="utf-8", newline="\n")

    (root / "data" / "HASHES.txt").write_text(
        f"{sha256_text(gold)}  data/gold.jsonl\n", encoding="utf-8", newline="\n"
    )
    (root / "data" / "conditions" / "HASHES.txt").write_text(
        f"{sha256_text(c0)}  data/conditions/C0.jsonl\n", encoding="utf-8", newline="\n"
    )
    (root / "CHANGELOG.md").write_text(
        "# CHANGELOG\n\n"
        "## 2026-11-15 - newest\n\n"
        f"SPEC SHA-256: `{sha256_text(spec)}`\n\n"
        "| file | SHA-256 |\n|---|---|\n"
        f"| `prompts/verifier_system.txt` | `{sha256_text(system_prompt)}` |\n",
        encoding="utf-8", newline="\n",
    )

    # the hash manifest the fix is required to add and verify
    listed = [
        "config.yaml", "analysis/contrasts.py", "src/verifier_client.py",
        "prompts/verifier_system.txt", "scripts/run_experiment.py",
    ]
    (root / "ANALYSIS.HASH").write_text(
        "".join(
            f"{sha256_text((root / rel).read_text(encoding='utf-8'))}  {rel}\n" for rel in listed
        ),
        encoding="utf-8", newline="\n",
    )

    git_init(root)
    git_commit_all(root, "frozen state")
    assert git(root, "tag", "spec-frozen").returncode == 0
    config = {
        "run": {
            "conditions_core": ["C0"], "conditions_extended": [], "k": 5,
            "order_seed": 1, "tag_required": "spec-frozen", "shards": 10,
        }
    }
    rx.assert_guards(root, config)  # sanity: the stub really does pass today
    return {"root": root, "config": config, "listed": listed}


# --------------------------------------------------------------------------- #
# R1 - run outputs must be gitignored so a finished shard leaves the tree clean
# --------------------------------------------------------------------------- #
R1_OUTPUTS = (
    "results/raw/deepseek_warned.jsonl",
    "results/raw/deepseek_naive.jsonl",
    "analysis/out/contrasts.csv",
    "analysis/cutpoints.json",
    "analysis/CUTPOINTS.HASH",
)


def test_r1_run_outputs_are_ignored_and_a_finished_shard_leaves_the_tree_clean(tmp_path):
    unignored = [rel for rel in R1_OUTPUTS if git(ROOT, "check-ignore", "-q", rel).returncode != 0]

    repo = tmp_path / "repo"
    git_init(repo)
    (repo / ".gitignore").write_bytes((ROOT / ".gitignore").read_bytes())
    (repo / "tracked.txt").write_text("tracked\n", encoding="utf-8", newline="\n")
    git_commit_all(repo, "initial")
    for rel in R1_OUTPUTS:
        target = repo / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("output\n", encoding="utf-8", newline="\n")

    assert not unignored, f"git does not ignore the run/analysis outputs: {unignored}"
    assert rx.git_tree_clean(repo) is True, (
        "after a shard wrote its outputs the tree is dirty, so every later shard "
        f"fails the freeze guard; git reports: {git(repo, 'status', '--porcelain').stdout!r}"
    )


def test_r1_guard_a_modified_tracked_file_still_makes_the_tree_dirty(tmp_path):
    repo = tmp_path / "repo"
    git_init(repo)
    (repo / "config.yaml").write_text("k: 5\n", encoding="utf-8", newline="\n")
    git_commit_all(repo, "initial")
    assert rx.git_tree_clean(repo) is True
    (repo / "config.yaml").write_text("k: 9\n", encoding="utf-8", newline="\n")
    assert rx.git_tree_clean(repo) is False, "editing a tracked file must still fail the guard"


# --------------------------------------------------------------------------- #
# R3 - scripts/run_analysis.py is the single analysis entry point
# --------------------------------------------------------------------------- #
RUN_ANALYSIS = ROOT / "scripts" / "run_analysis.py"
R3_ARTIFACTS = {
    "long table": ("long",),
    "cut points": ("cutpoint",),
    "C0 accuracy": ("accuracy",),
    "included / excluded counts": ("included",),
    "key-cell table": ("key_cell",),
    "contrasts": ("contrast",),
    "M1 strata": ("strata_m1",),
    "M2 strata": ("strata_m2",),
    "summary": ("summary.md",),
}


def run_driver(log: Path, out: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(RUN_ANALYSIS), "--log", str(log), "--out", str(out)],
        capture_output=True, text=True, cwd=str(ROOT),
    )


def test_r3_run_analysis_is_one_entry_point_that_writes_every_required_artifact(tmp_path):
    assert RUN_ANALYSIS.is_file(), (
        "no analysis entry point exists: nothing outside tests/ loads a log, computes "
        "cut points, assigns bins and writes the SPEC section 6 tables, so the order of "
        "operations is unenforced"
    )
    out = tmp_path / "out"
    done = run_driver(synthetic_log(tmp_path / "warned.jsonl"), out)
    assert done.returncode == 0, f"driver failed on a complete log: {done.stderr[-2000:]}"
    names = [path.name.lower() for path in out.rglob("*") if path.is_file()]
    missing = [
        label for label, needles in R3_ARTIFACTS.items()
        if not any(any(needle in name for needle in needles) for name in names)
    ]
    assert not missing, f"the driver did not write: {missing}; it wrote {sorted(names)}"

    short = synthetic_log(tmp_path / "short.jsonl", short_cell=(real_claims()["SUPPORT"][0], "C0"))
    refused = run_driver(short, tmp_path / "out_short")
    assert refused.returncode != 0, (
        "the driver analysed a log in which one (claim, condition) cell has only 4 result "
        "rows; an incomplete cell must stop the analysis, not silently rescale the majority"
    )


def test_r3_guard_report_write_all_keeps_writing_contrasts_and_summary_with_lf(tmp_path):
    from analysis import report

    rows = []
    for index in range(12):
        claim = f"c{index}"
        for sample_idx in range(5):
            rows.append(sample(claim, "C0", sample_idx, "SUPPORTED", "SUPPORTED"))
            rows.append(sample(claim, "C0p", sample_idx, "SUPPORTED", "SUPPORTED"))
            rows.append(sample(claim, "M1", sample_idx, "SUPPORTED", "NOT_ENOUGH_INFO"))
    out = tmp_path / "out"
    report.write_all(rows, seed=5, out_dir=out, n_resamples=200)
    contrast_bytes = (out / "contrasts.csv").read_bytes()
    assert b"\r\n" not in contrast_bytes, "contrasts.csv must stay LF-only"
    assert b"\r\n" not in (out / "summary.md").read_bytes(), "summary.md must stay LF-only"
    header = contrast_bytes.decode("utf-8").splitlines()[0].split(",")
    for column in ("contrast", "class", "estimate", "method", "ci_low", "ci_high", "p_value"):
        assert column in header, f"contrasts.csv lost the {column!r} column"


# --------------------------------------------------------------------------- #
# R5 - a second process on the same log must exit before any API call
# --------------------------------------------------------------------------- #
CHILD = '''
import json, sys
from pathlib import Path
REPO, LOG, CACHE, SENTINEL = sys.argv[1:5]
sys.path[:0] = [REPO, REPO + "/scripts", REPO + "/src"]
import verifier_client as vc
from backend import BackendResult
from prompts import PromptBundle


class Loud:
    timeout = 120.0

    def complete(self, system, user, **kwargs):
        Path(SENTINEL).write_text("api call made", encoding="utf-8")
        return BackendResult(text='{}', input_tokens=1, output_tokens=1, latency_ms=1.0,
                             finish_reason="stop", http_status=200)


client = vc.VerifierClient(
    vc.ModelConfig.from_dict(json.loads(sys.argv[5])), Loud(),
    PromptBundle(system="SYS", user_template="{claim} {evidence}",
                 system_sha256="a" * 64, user_sha256="b" * 64),
    Path(LOG), cache_dir=Path(CACHE),
)
client.run_batch([{ "claim_id": "c_x", "claim": "x",
                    "evidence_sentences": [{"idx": 0, "text": "y"}],
                    "sample_idx": 0, "condition": "C0" }], order_seed=1, resume_passes=0)
print("second writer ran to completion")
'''


def test_r5_a_second_process_on_the_same_log_exits_before_calling_the_api(tmp_path):
    client, _, _ = make_client(tmp_path, [VALID])
    client.verify(**unit("c_a"))
    assert client.log_path.is_file()

    script = tmp_path / "child.py"
    script.write_text(textwrap.dedent(CHILD), encoding="utf-8")
    sentinel = tmp_path / "api_called"
    section = {
        "base_url": "https://api.example.test", "model": "deepseek-test",
        "api_key_env": "DEEPSEEK_API_KEY", "temperature": 0.7, "top_p": None,
        "max_tokens": 64, "thinking": {"enabled": False, "budget_tokens": None},
        "concurrency": 1, "max_retries": 2, "retry_on": ["api_error"],
        "blocked_body_markers": [], "price_table": None,
    }
    child = subprocess.run(
        [sys.executable, str(script), str(ROOT), str(client.log_path),
         str(tmp_path / "cache"), str(sentinel), json.dumps(section)],
        capture_output=True, text=True,
    )
    assert child.returncode != 0, (
        "a second writer opened the log held by the first one; concurrent appends tear "
        f"JSONL lines and duplicate paid calls. child stdout: {child.stdout.strip()!r}"
    )
    assert not sentinel.exists(), "the second writer reached the API before being refused"


def test_r5_guard_one_client_writes_every_row_as_a_parseable_line(tmp_path):
    client, _, _ = make_client(tmp_path, [VALID] * 6)
    for index in range(6):
        client.verify(**unit(f"c_{index}"))
    text = client.log_path.read_text(encoding="utf-8")
    assert b"\r\n" not in client.log_path.read_bytes(), "the log must stay LF-only"
    lines = [line for line in text.splitlines() if line.strip()]
    assert len(lines) == 6
    for line in lines:
        json.loads(line)


# --------------------------------------------------------------------------- #
# R6 - an empty arm must never produce a number
# --------------------------------------------------------------------------- #
def nei_dataset_without_m1() -> list[dict]:
    rows: list[dict] = []
    for index in range(12):
        claim = f"n{index}"
        for sample_idx in range(5):
            for condition in ("C0", "C0p", "C2", "M2"):
                expected = "NOT_ENOUGH_INFO"
                if condition == "C0":
                    verdict = expected                      # the filter keeps every claim
                elif condition == "C0p":
                    # a real noise floor: one in five C0p samples is not as expected
                    verdict = expected if sample_idx < 4 else "SUPPORTED"
                else:
                    verdict = expected if sample_idx < 3 else "SUPPORTED"
                rows.append(sample(claim, condition, sample_idx, verdict, expected,
                                   class_name="NEI"))
    return rows


def test_r6_a_contrast_with_an_empty_arm_reports_no_estimate(tmp_path):
    rows = contrasts.run_contrasts(nei_dataset_without_m1(), seed=5, n_resamples=200)
    offenders = [
        row for row in rows
        if (row["n_manipulation"] == 0 or row["n_control"] == 0) and row["estimate"] is not None
    ]
    assert not offenders, (
        "a contrast with zero contributing samples in one arm published a number: "
        + "; ".join(
            f"{row['contrast']}/{row['class']} estimate={row['estimate']} "
            f"n_manipulation={row['n_manipulation']} n_control={row['n_control']}"
            for row in offenders
        )
    )


def test_r6_guard_a_populated_contrast_still_reports_the_exact_rate_difference(tmp_path):
    rows: list[dict] = []
    for index in range(12):
        claim = f"c{index}"
        for sample_idx in range(5):
            rows.append(sample(claim, "C0", sample_idx, "SUPPORTED", "SUPPORTED"))
            rows.append(sample(claim, "C0p", sample_idx, "SUPPORTED", "SUPPORTED"))
            # M1 expects NOT_ENOUGH_INFO and never gives it: every M1 sample is an error
            rows.append(sample(claim, "M1", sample_idx, "SUPPORTED", "NOT_ENOUGH_INFO"))
    result = contrasts.run_contrasts(rows, seed=5, n_resamples=200)
    m1 = [row for row in result if row["contrast"] == "M1_vs_C0p" and row["n_manipulation"]]
    assert m1, "the populated M1 contrast disappeared"
    row = m1[0]
    expected = row["not_as_expected_manipulation"] / row["n_manipulation"] - (
        row["not_as_expected_control"] / row["n_control"]
    )
    assert row["estimate"] == pytest.approx(expected), "the point estimate is no longer the rate difference"
    assert row["estimate"] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# R7 - the freeze must cover config.yaml, analysis/, src/ and the runner
# --------------------------------------------------------------------------- #
def analysis_hash_targets() -> list[str]:
    targets = ["config.yaml", "scripts/run_experiment.py"]
    for directory in ("analysis", "src", "prompts"):
        for path in sorted((ROOT / directory).rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                targets.append(path.relative_to(ROOT).as_posix())
    return targets


def test_r7_analysis_hash_pins_config_analysis_src_and_the_runner(frozen_repo):
    manifest = ROOT / "ANALYSIS.HASH"
    assert manifest.is_file(), (
        "ANALYSIS.HASH does not exist, so config.yaml and every analysis script can be "
        "edited after the spec-frozen tag with no guard noticing"
    )
    listed = {
        relative.strip(): digest.strip()
        for digest, relative in (
            line.split(None, 1)
            for line in (part.strip() for part in manifest.read_text(encoding="utf-8").splitlines())
            if line
        )
    }
    missing = [target for target in analysis_hash_targets() if target not in listed]
    assert not missing, f"ANALYSIS.HASH does not cover: {missing}"
    stale = [
        relative for relative, digest in listed.items()
        if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != digest
    ]
    assert not stale, f"ANALYSIS.HASH digests are stale for: {stale}"

    root, config = frozen_repo["root"], frozen_repo["config"]
    (root / "analysis" / "contrasts.py").write_text(
        "N_RESAMPLES = 50\n", encoding="utf-8", newline="\n"
    )
    git_commit_all(root, "post-freeze edit to an analysis script")
    with pytest.raises(rx.GuardError):
        rx.assert_guards(root, config)


def test_r7_guard_a_changed_prompt_file_still_fails_the_guard(frozen_repo):
    root, config = frozen_repo["root"], frozen_repo["config"]
    rx.assert_guards(root, config)
    (root / "prompts" / "verifier_system.txt").write_text(
        "you are a different verifier\n", encoding="utf-8", newline="\n"
    )
    git_commit_all(root, "post-freeze edit to a prompt")
    with pytest.raises(rx.GuardError):
        rx.assert_guards(root, config)


# --------------------------------------------------------------------------- #
# R8 - a hash manifest that lists nothing must not pass
# --------------------------------------------------------------------------- #
def test_r8_an_empty_or_incomplete_hashes_file_fails_the_guard(tmp_path):
    covered = tmp_path / "data"
    covered.mkdir()
    for name in ("a.jsonl", "b.jsonl", "c.jsonl"):
        (covered / name).write_text(f"{name}\n", encoding="utf-8", newline="\n")
    manifest = covered / "HASHES.txt"

    manifest.write_bytes(b"")
    empty_passes = rx.hash_file_matches(tmp_path, manifest)
    manifest.write_bytes(b"\r\n   \n\n")
    blank_passes = rx.hash_file_matches(tmp_path, manifest)
    manifest.write_text(
        "".join(
            f"{sha256_text((covered / name).read_text(encoding='utf-8'))}  data/{name}\n"
            for name in ("a.jsonl", "b.jsonl")
        ),
        encoding="utf-8", newline="\n",
    )
    incomplete_passes = rx.hash_file_matches(tmp_path, manifest)

    assert empty_passes is False, "an empty HASHES.txt passed the data-freeze guard vacuously"
    assert blank_passes is False, "a whitespace-only HASHES.txt passed the data-freeze guard"
    assert incomplete_passes is False, (
        "a HASHES.txt covering 2 of the 3 files in its directory passed; the row count must "
        "equal the number of files it covers"
    )


def test_r8_guard_a_complete_manifest_with_crlf_rows_still_validates(tmp_path):
    covered = tmp_path / "data"
    covered.mkdir()
    (covered / "a.jsonl").write_text("a.jsonl\n", encoding="utf-8", newline="\n")
    manifest = covered / "HASHES.txt"
    digest = sha256_text("a.jsonl\n")
    manifest.write_bytes(f"{digest}  data/a.jsonl\r\n".encode("utf-8"))
    assert rx.hash_file_matches(tmp_path, manifest) is True, (
        "CR tolerance in the manifest reader must survive the fix"
    )
    (covered / "a.jsonl").write_text("tampered\n", encoding="utf-8", newline="\n")
    assert rx.hash_file_matches(tmp_path, manifest) is False, (
        "a changed data file must still fail its digest"
    )


# --------------------------------------------------------------------------- #
# R9 - one log per tier and prompt variant; no globbing across files
# --------------------------------------------------------------------------- #
def test_r9_first_result_rows_reads_one_named_log_and_tiers_do_not_share_one(tmp_path):
    raw = tmp_path / "raw"
    warned = raw / "deepseek_warned.jsonl"
    naive = raw / "deepseek_naive.jsonl"
    key = {"claim_id": "c_1", "condition": "M2", "sample_idx": 0}
    write_jsonl(warned, [{**key, "is_result": True, "outcome": "ok", "verdict": "REFUTED",
                          "confidence": 90, "rationale": "WARNED", "cache_key": "k1"}])
    write_jsonl(naive, [{**key, "is_result": True, "outcome": "ok", "verdict": "NOT_ENOUGH_INFO",
                         "confidence": 1, "rationale": "NAIVE", "cache_key": "k1"}])

    rows = loaders.first_result_rows(warned)
    rationales = [row.get("rationale") for row in rows]
    assert rationales == ["WARNED"], (
        "first_result_rows must read the log it is given; it globs a directory instead, so "
        "deepseek_naive.jsonl sorts first and silently replaces the warned C0/M2 rows "
        f"(got {rationales!r})"
    )
    assert rx.log_path_for(ROOT, "core") != rx.log_path_for(ROOT, "extended"), (
        "the core and extended tiers write to the same log, so 'the core tier is complete "
        "and logged' cannot be read off the log"
    )


def test_r9_guard_within_one_log_the_first_result_row_per_key_still_wins(tmp_path):
    log = tmp_path / "raw" / "deepseek_warned.jsonl"
    key = {"claim_id": "c_1", "condition": "C0", "sample_idx": 0}
    write_jsonl(log, [
        {**key, "is_result": False, "outcome": "api_error", "verdict": None, "cache_key": "k"},
        {**key, "is_result": True, "outcome": "ok", "verdict": "SUPPORTED", "rationale": "FIRST",
         "cache_key": "k"},
        {**key, "is_result": True, "outcome": "ok", "verdict": "REFUTED", "rationale": "LATER",
         "cache_key": "k"},
    ])
    rows = loaders.first_result_rows(log.parent)
    assert [row["rationale"] for row in rows] == ["FIRST"], (
        "the first result row per key must win and is_result=false rows must be dropped"
    )


# --------------------------------------------------------------------------- #
# R10 - a persistent unmarked 4xx must end as exactly one api_error result row
# --------------------------------------------------------------------------- #
def unmarked_4xx() -> BackendError:
    return BackendError("request rejected 400", 400, '{"error":"invalid request"}')


def test_r10_a_persistent_unmarked_4xx_ends_as_one_api_error_result_row(tmp_path):
    client, backend, _ = make_client(tmp_path, [], cycle=unmarked_4xx())
    client.run_batch([unit("c_a", "C2", 0)], order_seed=1, resume_passes=2)
    results = [row for row in log_rows(client.log_path) if row.get("is_result")]
    assert len(results) == 1, (
        "a unit that always gets an unmarked non-auth 4xx produced "
        f"{len(results)} result rows; the sample vanishes from the data with no error, so "
        "its (claim, condition) cell silently has fewer than 5 samples"
    )
    assert results[0]["outcome"] == "api_error"
    assert results[0]["verdict"] is None


def test_r10_guard_three_consecutive_unmarked_4xx_in_one_pass_abort_the_run(tmp_path):
    client, _, _ = make_client(tmp_path, [], cycle=unmarked_4xx())
    units = [unit(f"c_{index}", "C0", 0) for index in range(5)]
    with pytest.raises(vc.RunAborted):
        client.run_batch(units, order_seed=1, resume_passes=0)

    other, _, _ = make_client(
        tmp_path / "second",
        [unmarked_4xx(), unmarked_4xx(), VALID, unmarked_4xx(), unmarked_4xx()],
    )
    other.run_batch(
        [unit(f"d_{index}", "C0", 0) for index in range(5)], order_seed=1, resume_passes=0
    )
    outcomes = [row["outcome"] for row in log_rows(other.log_path) if row.get("cache_key")]
    assert "ok" in outcomes, "a success between 4xx responses must reset the abort counter"


# --------------------------------------------------------------------------- #
# R11 - majority and stability need exactly five rows
# --------------------------------------------------------------------------- #
def test_r11_majority_and_stability_require_exactly_five_rows(tmp_path):
    four = ["SUPPORTED", "SUPPORTED", "SUPPORTED", "REFUTED"]
    rows = [sample("c_1", "C0", index, verdict, "SUPPORTED", confidence=70 + index)
            for index, verdict in enumerate(four)]
    stability_rows = tables.stability_table(rows)
    assert len(stability_rows) == 1
    cell = stability_rows[0]

    assert tables.majority(four) is None, (
        "majority resolved on a 4-sample group; SPEC section 6 defines it as at least 3 of "
        "k=5, so a group that is not 5 rows has no majority"
    )
    assert loaders._majority(four) is None, "the loader's copy of majority must agree"
    assert cell["stability"] is None, (
        f"stability was reported as {cell['stability']} for a 4-row group; the share must be "
        "out of 5 or undefined"
    )
    assert cell["confidence_sd"] is None, (
        f"confidence_sd was reported as {cell['confidence_sd']} over fewer than 5 confidences"
    )


def test_r11_guard_a_five_row_group_with_three_agreeing_still_resolves(tmp_path):
    five = ["SUPPORTED", "SUPPORTED", "SUPPORTED", "REFUTED", None]
    rows = [sample("c_1", "C0", index, verdict, "SUPPORTED", confidence=None if verdict is None else 70 + index)
            for index, verdict in enumerate(five)]
    assert tables.majority(five) == "SUPPORTED"
    cell = tables.stability_table(rows)[0]
    assert cell["majority"] == "SUPPORTED"
    assert cell["stability"] == pytest.approx(0.6), "3 of 5 samples equal to the majority"
    assert cell["confidence_sd"] is not None


# --------------------------------------------------------------------------- #
# R12 - the filter accounting must add up
# --------------------------------------------------------------------------- #
def filter_dataset() -> list[dict]:
    rows: list[dict] = []
    # 3 claims the filter keeps
    for index in range(3):
        for sample_idx in range(5):
            rows.append(sample(f"keep{index}", "C0", sample_idx, "SUPPORTED", "SUPPORTED"))
    # 1 claim with no C0 majority (2-2-1 across verdicts and a no-verdict sample)
    for sample_idx, verdict in enumerate(["SUPPORTED", "SUPPORTED", "REFUTED", "REFUTED", None]):
        rows.append(sample("nomaj", "C0", sample_idx, verdict, "SUPPORTED"))
    # 1 claim whose C0 majority is confidently wrong
    for sample_idx in range(5):
        rows.append(sample("wrong", "C0", sample_idx, "REFUTED", "SUPPORTED"))
    return rows


def test_r12_included_and_both_exclusion_counts_add_up_to_every_claim(tmp_path):
    counts = tables.included_claims(filter_dataset())
    assert "excluded_wrong_majority" in counts, (
        "included_claims reports only the included and no-majority counts, so claims whose "
        "C0 majority exists but differs from gold are in neither number and the accounting "
        f"does not add up to the subset (got keys {sorted(counts)})"
    )
    included = counts["included"]
    no_majority = counts["excluded_no_majority"]
    wrong = counts["excluded_wrong_majority"]
    assert isinstance(no_majority, dict) and isinstance(wrong, dict), (
        "both exclusion counts must be reported per class, like the included counts"
    )
    for label in ("SUPPORT", "REFUTE", "NEI"):
        expected = {"SUPPORT": 5, "REFUTE": 0, "NEI": 0}[label]
        total = included.get(label, 0) + no_majority.get(label, 0) + wrong.get(label, 0)
        assert total == expected, f"{label}: {total} accounted for, {expected} claims exist"


def test_r12_guard_the_included_per_class_counts_stay_correct(tmp_path):
    counts = tables.included_claims(filter_dataset())
    assert counts["included"]["SUPPORT"] == 3, (
        "only the claims whose C0 majority equals gold are included"
    )
    assert counts["included"]["REFUTE"] == 0 and counts["included"]["NEI"] == 0


# --------------------------------------------------------------------------- #
# R13 - both sensitivity tables come out of the same driver call
# --------------------------------------------------------------------------- #
def test_r13_the_driver_emits_both_sensitivity_tables_with_the_primary_one(tmp_path):
    assert RUN_ANALYSIS.is_file(), (
        "no driver exists, so neither SPEC section 6 sensitivity analysis is produced at "
        "all: no_verdict samples counted as not_as_expected, and the fixed high >= 90 "
        "threshold (cutpoints.SENSITIVITY_HIGH is written to disk and read by nothing)"
    )
    out = tmp_path / "out"
    done = run_driver(synthetic_log(tmp_path / "warned.jsonl"), out)
    assert done.returncode == 0, f"driver failed: {done.stderr[-2000:]}"
    names = [path.name.lower() for path in out.rglob("*") if path.is_file()]
    sensitivity = [name for name in names if "sensitivity" in name]
    assert any("no_verdict" in name for name in sensitivity), (
        f"no no_verdict sensitivity table among {sorted(names)}"
    )
    assert any("90" in name for name in sensitivity), (
        f"no fixed-threshold (high >= 90) sensitivity table among {sorted(names)}"
    )


def test_r13_guard_the_fixed_sensitivity_threshold_is_still_declared_as_ninety(tmp_path):
    from analysis import cutpoints as cp

    assert cp.SENSITIVITY_HIGH == 90
    points = cp.write_cutpoints(
        [sample("c_1", "C0", index, "SUPPORTED", "SUPPORTED", confidence=50 + index * 10)
         for index in range(5)],
        tmp_path / "analysis",
    )
    assert points["sensitivity_high"] == 90, "cutpoints.json must keep recording the threshold"
    assert (tmp_path / "analysis" / "CUTPOINTS.HASH").is_file()


# --------------------------------------------------------------------------- #
# R14 - the bootstrap resamples the filtered claims only
# --------------------------------------------------------------------------- #
def paired_dataset(extra_excluded: int = 0) -> list[dict]:
    """20 claims the filter keeps, with per-claim error rates spread 0..1 so that
    resampling claims actually moves the statistic."""
    rows: list[dict] = []
    for index in range(20):
        claim = f"c{index}"
        m1_correct = index % 6          # 0..5 of the 5 M1 samples answer NOT_ENOUGH_INFO
        c0p_correct = 3 + index % 3     # 3..5 of the 5 C0p samples answer gold
        for sample_idx in range(5):
            rows.append(sample(claim, "C0", sample_idx, "SUPPORTED", "SUPPORTED"))
            rows.append(sample(claim, "C0p", sample_idx,
                               "SUPPORTED" if sample_idx < c0p_correct else "REFUTED",
                               "SUPPORTED"))
            rows.append(sample(claim, "M1", sample_idx,
                               "NOT_ENOUGH_INFO" if sample_idx < m1_correct else "SUPPORTED",
                               "NOT_ENOUGH_INFO"))
    for index in range(extra_excluded):
        claim = f"x{index}"
        for sample_idx in range(5):
            # C0 majority is REFUTED while gold is SUPPORT: the filter drops this claim
            rows.append(sample(claim, "C0", sample_idx, "REFUTED", "SUPPORTED"))
            rows.append(sample(claim, "C0p", sample_idx, "SUPPORTED", "SUPPORTED"))
            rows.append(sample(claim, "M1", sample_idx, "SUPPORTED", "NOT_ENOUGH_INFO"))
    return rows


def interval(rows: list[dict]) -> tuple:
    for row in contrasts.run_contrasts(rows, seed=11, n_resamples=400):
        if row["contrast"] == "M1_vs_C0p" and row["n_manipulation"]:
            return row["ci_low"], row["ci_high"]
    raise AssertionError("no populated M1_vs_C0p row")


def test_r14_filter_excluded_claims_never_enter_a_resample(tmp_path):
    without = interval(paired_dataset())
    with_excluded = interval(paired_dataset(extra_excluded=20))
    assert without == pytest.approx(with_excluded), (
        "adding 20 claims that the C0 filter excludes moved the interval from "
        f"{without} to {with_excluded}; excluded claims are being drawn into every resample "
        "as zero-weight units, which changes the effective resample size"
    )


def test_r14_guard_the_bootstrap_leaves_the_confidence_bins_untouched(tmp_path):
    rows = paired_dataset()
    before = [row["bin"] for row in rows]
    contrasts.run_contrasts(rows, seed=11, n_resamples=200)
    assert [row["bin"] for row in rows] == before, (
        "cut points and bins must stay fixed across the bootstrap"
    )


# --------------------------------------------------------------------------- #
# R15 - the two primary contrasts are pooled across classes
# --------------------------------------------------------------------------- #
def three_class_dataset() -> list[dict]:
    rows: list[dict] = []
    for label in ("SUPPORT", "REFUTE", "NEI"):
        gold = GOLD_VERDICT[label]
        for index in range(12):
            claim = f"{label}{index}"
            for sample_idx in range(5):
                rows.append(sample(claim, "C0", sample_idx, gold, gold, class_name=label))
                rows.append(sample(claim, "C0p", sample_idx, gold, gold, class_name=label))
                rows.append(sample(claim, "C2", sample_idx, gold, gold, class_name=label))
                rows.append(sample(claim, "M2", sample_idx,
                                   gold if sample_idx < 2 else "SUPPORTED", gold,
                                   class_name=label))
                if label != "NEI":
                    rows.append(sample(claim, "M1", sample_idx,
                                       "NOT_ENOUGH_INFO" if sample_idx < 2 else gold,
                                       "NOT_ENOUGH_INFO", class_name=label))
    return rows


def test_r15_holm_covers_exactly_two_pooled_primary_p_values(tmp_path):
    rows = contrasts.run_contrasts(three_class_dataset(), seed=7, n_resamples=300)
    with_p = [row for row in rows if row.get("p_value") is not None]
    assert len(with_p) == 2, (
        "the primary estimand must be pooled across classes, giving exactly two p-values for "
        f"Holm; {len(with_p)} rows carry a p-value "
        + str([(row["contrast"], row["class"]) for row in with_p])
    )
    assert {row["contrast"] for row in with_p} == PRIMARY_CONTRASTS
    assert not [row for row in with_p if row["class"] in CLASS_NAMES], (
        "the two pooled rows must not be labelled with a single class"
    )
    per_class = [row for row in rows if row["class"] in CLASS_NAMES and row.get("p_value") is not None]
    assert not per_class, f"per-class rows must carry no p-value: {[r['contrast'] for r in per_class]}"


def test_r15_guard_the_secondary_contrast_never_receives_a_holm_adjustment(tmp_path):
    rows = contrasts.run_contrasts(three_class_dataset(), seed=7, n_resamples=300)
    secondary = [row for row in rows if row["contrast"] == "M2_vs_C0p"]
    assert secondary, "the secondary M2 vs C0p contrast disappeared"
    assert all(row["p_holm"] is None for row in secondary), (
        "Holm must stay restricted to the primary contrasts"
    )


# --------------------------------------------------------------------------- #
# R18 - the log lives outside the repo and an append is retried
# --------------------------------------------------------------------------- #
def test_r18_logs_live_outside_the_repo_and_a_locked_append_is_retried(tmp_path, monkeypatch):
    import yaml

    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    results_dir = (config.get("run") or {}).get("results_dir")
    assert results_dir, (
        "config.yaml has no run.results_dir, so the runner writes its log inside the "
        "repository under OneDrive; the cache was moved out of OneDrive but the log was not"
    )
    resolved = Path(results_dir)
    assert resolved.is_absolute(), "run.results_dir must be an absolute path outside the repo"
    assert ROOT not in resolved.parents and resolved != ROOT, f"{resolved} is inside the repo"
    assert not any("onedrive" in part.lower() for part in resolved.parts), (
        f"{resolved} is on a OneDrive-synced path"
    )

    client, _, sleeps = make_client(tmp_path, [VALID])
    real_open = Path.open
    failures = {"left": 2}

    def flaky_open(self, *args, **kwargs):
        mode = args[0] if args else kwargs.get("mode", "r")
        if self == client.log_path and "a" in mode and failures["left"]:
            failures["left"] -= 1
            raise PermissionError(13, "The process cannot access the file")
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", flaky_open)
    client.verify(**unit("c_a"))
    monkeypatch.undo()

    rows = log_rows(client.log_path)
    assert len(rows) == 1, (
        "a PermissionError on appending to the log lost the row; a sync lock on the log must "
        "be retried, not propagated out of _log"
    )
    assert sleeps, "the retries must use the injected sleep, not wall-clock time"


def test_r18_guard_logging_is_append_only_and_keeps_every_earlier_row(tmp_path):
    client, _, _ = make_client(tmp_path, [VALID, VALID])
    client.verify(**unit("c_a"))
    first = client.log_path.read_text(encoding="utf-8")
    client.verify(**unit("c_b"))
    second = client.log_path.read_text(encoding="utf-8")
    assert second.startswith(first), "logging must append, never rewrite"
    assert len(log_rows(client.log_path)) == 2


# --------------------------------------------------------------------------- #
# R19 - cumulative sleep per run is bounded
# --------------------------------------------------------------------------- #
def test_r19_cumulative_retry_sleep_is_capped_at_1800_seconds(tmp_path):
    sleeps: list[float] = []
    client, _, _ = make_client(
        tmp_path, [], cycle=RateLimitError("rate limited", retry_after=900.0),
        sleep=sleeps.append, max_retries=6,
    )
    aborted = False
    try:
        client.verify(**unit("c_a"))
    except vc.RunAborted:
        aborted = True
    assert sum(sleeps) <= 1800.0 and aborted, (
        f"a provider answering every attempt with Retry-After: 900 made one call sleep "
        f"{sum(sleeps)}s across {len(sleeps)} waits ({sleeps}) and "
        f"{'did not abort' if not aborted else 'aborted'}; cumulative sleep per run must be "
        "capped at 1800s and exceeding it must raise RunAborted"
    )


def test_r19_guard_a_single_retry_after_is_still_capped_at_900_seconds(tmp_path):
    sleeps: list[float] = []
    client, _, _ = make_client(
        tmp_path, [RateLimitError("rate limited", retry_after=5000.0), VALID],
        sleep=sleeps.append,
    )
    row = client.verify(**unit("c_a"))
    assert row["outcome"] == "ok"
    assert sleeps == [900.0], f"the per-wait Retry-After cap must stay 900s, slept {sleeps}"


# --------------------------------------------------------------------------- #
# R20 - the dry run reports the tier total whatever the config says
# --------------------------------------------------------------------------- #
def tiny_run_root(tmp_path: Path, *, order_seed, naive: bool) -> tuple[Path, Path]:
    root = tmp_path / "run"
    (root / "data" / "conditions").mkdir(parents=True)
    (root / "prompts").mkdir()
    rows = [{"id": f"c_{index}", "claim": f"claim {index}",
             "evidence_sentences": [{"idx": 0, "text": f"sentence {index}"}]}
            for index in range(10)]
    for code in ("C0", "C2", "M1", "M2", "C1", "C_closed"):
        write_jsonl(root / "data" / "conditions" / f"{code}.jsonl", rows)
    (root / "prompts" / "verifier_system.txt").write_text("sys\n", encoding="utf-8", newline="\n")
    (root / "prompts" / "verifier_user.txt").write_text(
        "claim: {claim}\nevidence: {evidence}\n", encoding="utf-8", newline="\n"
    )
    if naive:
        (root / "prompts" / "verifier_system_naive.txt").write_text(
            "naive sys\n", encoding="utf-8", newline="\n"
        )
    config = {
        "seed": 1,
        "data": {
            "name": "scifact", "source_url": "https://example.invalid/d.tar.gz",
            "source_sha256": "0" * 64, "raw_dir": "data/raw/scifact",
            "splits": ["train", "dev"],
            "files": {"corpus": "corpus.jsonl", "claims": "claims_{split}.jsonl"},
            "output": {"verifier_input": "data/verifier_input.jsonl", "gold": "data/gold.jsonl",
                       "key": "data/key.jsonl", "hashes": "data/HASHES.txt"},
        },
        "subset": {"per_label": {"SUPPORT": 1, "REFUTE": 1, "NEI": 1},
                   "id_prefix": "c_", "id_hex_len": 10},
        "pilot": {"seed": 1, "per_class": 1},
        "cache": {"dir": str(tmp_path / "cache")},
        "run": {"conditions_core": ["C0", "C0p", "C2", "M1", "M2"],
                "conditions_extended": ["C1", "C_closed"], "k": 5,
                "order_seed": order_seed, "tag_required": "spec-frozen", "shards": 10},
    }
    import yaml

    path = root / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8", newline="\n")
    return root, path


def test_r20_dry_run_counts_do_not_depend_on_order_seed_or_the_naive_prompt(tmp_path):
    unset_root, unset_config = tiny_run_root(tmp_path / "a", order_seed=None, naive=False)
    set_root, set_config = tiny_run_root(tmp_path / "b", order_seed=20261118, naive=True)

    unset = rx.dry_run(unset_root, unset_config, "core")
    with_seed = rx.dry_run(set_root, set_config, "core")
    assert with_seed["total_calls"] == unset["total_calls"] == 10 * 5 * 5, (
        "total_calls means the size of the tier, but setting run.order_seed silently turns it "
        f"into the size of shard 1 ({with_seed['total_calls']} vs {unset['total_calls']}); "
        "the one change a real run needs therefore turns the suite red"
    )
    naive = rx.dry_run(set_root, set_config, "naive")
    assert naive["total_calls"] == 10 * 5 * 2
    assert naive["missing_naive_prompt"] is False
    assert naive["prompt_chars"] is not None


def test_r20_guard_the_shard_slice_still_partitions_the_tier_exactly_once(tmp_path):
    root, config_path = tiny_run_root(tmp_path, order_seed=20261118, naive=False)
    config = rx.load_config_for(config_path)
    units = rx.ordered_units(rx.build_units(root, config, "core"), 20261118)
    seen: list[tuple] = []
    for index in range(1, 11):
        for item in rx.shard_slice(units, index, 10):
            seen.append((item["claim_id"], item["condition"], item["sample_idx"]))
    assert len(seen) == len(units) == 250
    assert len(set(seen)) == 250, "the shards must partition the tier with no overlap or gap"


# --------------------------------------------------------------------------- #
# R30 - the SPEC-hash guard reads the newest CHANGELOG entry
# --------------------------------------------------------------------------- #
def changelog(entries: list[tuple[str, str]]) -> str:
    parts = ["# CHANGELOG", ""]
    for date, digest in entries:
        parts += [f"## {date} - entry", "", f"SPEC SHA-256: `{digest}`", ""]
    return "\n".join(parts) + "\n"


def test_r30_the_spec_hash_guard_uses_the_newest_changelog_entry_by_date(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    spec = "# SPEC v2\n\ncurrent text\n"
    (root / "SPEC.md").write_text(spec, encoding="utf-8", newline="\n")
    stale = sha256_text("# SPEC v1\n\nold text\n")
    (root / "CHANGELOG.md").write_text(
        changelog([("2026-11-01", stale), ("2026-11-15", sha256_text(spec))]),
        encoding="utf-8", newline="\n",
    )
    assert rx.spec_hash_matches(root) is True, (
        "the newest entry sits at the bottom of the file, which is the ordinary changelog "
        "convention; the guard takes the first regex match and so compares SPEC.md against "
        "a stale hash"
    )


def test_r30_guard_a_newest_first_changelog_still_validates_and_a_tampered_spec_does_not(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    spec = "# SPEC v2\n\ncurrent text\n"
    (root / "SPEC.md").write_text(spec, encoding="utf-8", newline="\n")
    (root / "CHANGELOG.md").write_text(
        changelog([("2026-11-15", sha256_text(spec)), ("2026-11-01", sha256_text("old\n"))]),
        encoding="utf-8", newline="\n",
    )
    assert rx.spec_hash_matches(root) is True, "newest-first must keep working"
    (root / "SPEC.md").write_text(spec + "a quiet edit\n", encoding="utf-8", newline="\n")
    assert rx.spec_hash_matches(root) is False, "an edited SPEC.md must still fail the guard"
