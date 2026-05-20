#!/usr/bin/env python3
"""End-to-end smoke test for the OpenCode skills-intervention pipeline.

Runs in the PR gate (via ``tests/unit/tools/test-skills-value-smoke.sh``).
Doesn't require pytest — uses ``assert`` so failures show up cleanly in
both the GitHub Actions log and a local run.

Covers:

1. ``_classify_validity`` on the four signature shapes we care about
   (ok / infra_fail / incomplete).
2. ``_classify_failure_mode`` on every F-code branch (F0/F7/F8/F9/F10/
   F12/F13) plus the pass-fast-path (None).
3. ``_attempts_to_pass`` on first-attempt-pass, third-attempt-pass,
   never-pass, missing-attempts, and a malformed attempts list.
4. ``_wilson_ci`` bounds: 11/12 ~= [0.65, 0.99], 0/8 ~= [0.0, 0.32],
   0/0 is None.
5. ``compare_skills_value.py`` on a "no clean off-baseline" scenario
   (``pass_rate_off_clean`` must be None, no headline delta in the
   markdown).
6. ``compare_skills_value.py`` on a "clean baseline + unlock"
   scenario (``pass_rate_off_clean`` is a number, the markdown emits
   the "clean baseline available" table, freshness aggregates are
   populated, ``failure_mode_counts_*`` and ``attempts_to_pass_*``
   land on the per-profile aggregate, Wilson CIs are present).
7. ``compare_skills_value.py`` with ``--partial-run`` and
   ``--legs-status-file`` — the produced summary embeds both.
8. ``tests/tools/merge_evidence_artifacts.sh`` — the synthetic
   artifact dirs (fresh per-leg files + stale cross-leg copies of
   each filename) merge such that the freshest per-leg files survive
   regardless of artifact iteration order.
9. ``tests/tools/detect_partial_run.py`` — synthetic GH-API ``jobs``
   payloads produce the expected ``partial_run`` boolean and write to
   ``$GITHUB_OUTPUT``.
10. ``tests/tools/generate_dashboard.py`` — the produced HTML carries
    the inlined summary keys for both verdict branches, the new
    failure-mode chip helpers, and the CI hint helper.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parent.parent
TOOLS = REPO / "tests" / "tools"
sys.path.insert(0, str(TOOLS))
import compare_skills_value as csv_mod  # noqa: E402


# ---------------------------------------------------------------------------
# Synthetic evidence builders.
# ---------------------------------------------------------------------------


def _ev_template(*, mode, op, dtype, model, tokens_total, artifacts,
                 kernel_path, verification_status, score_accepted,
                 semantic_passed, static_verify,
                 verification_mode="simulator",
                 attempts=None,
                 date="2026-05-19T00:00:00Z",
                 profile="cloud-default") -> dict:
    """Build a schema-v3 evidence-shaped dict for testing.

    All fields documented in collect_generative_evidence.py's output
    schema are present; tests can override individual fields. The
    default attempts list mirrors what the collector emits (a single
    record per attempt with ``outcome``/``kernel_found`` keys); callers
    can pass ``attempts=[...]`` to exercise the attempts-to-pass logic.
    """
    if attempts is None:
        if tokens_total:
            attempts = [{
                "n": 1, "outcome": "pass" if score_accepted else "fail",
                "kernel_found": bool(kernel_path),
                "runtime_status": verification_status,
                "prompt_label": "primary",
                "elapsed_s": 60.0, "exit": 0, "model": model,
            }]
        else:
            attempts = []
    return {
        "schema_version": "3",
        "kind": "generative",
        "operation": op,
        "dtype": dtype,
        "prompt": "test prompt",
        "kernel_path": kernel_path,
        "date": date,
        "agent": {
            "platform": "opencode",
            "timeout_s": 420,
            "completed": True,
            "artifacts_found": artifacts,
        },
        "skills_mode": mode,
        "model_profile": profile,
        "model": model,
        "tokens": {
            "input": tokens_total // 2 if tokens_total else 0,
            "output": tokens_total // 2 if tokens_total else 0,
            "cache_read": 0,
            "total": tokens_total,
        },
        "elapsed_total_s": 120.0 if tokens_total else 30.0,
        "attempts": attempts,
        "verification": {
            "mode": verification_mode,
            "status": verification_status,
        },
        "semantic_check": {"passed": semantic_passed},
        "score": {
            "value": 16.0 if score_accepted else 0.0,
            "threshold": 12,
            "accepted": score_accepted,
        },
        "static_verify": static_verify,
        "notes": "smoke",
        "ci_run_url": "",
        "history": [],
    }


def _write_ev(d: Path, name: str, ev: dict) -> None:
    (d / name).write_text(json.dumps(ev, indent=2))


def _run_aggregator(ev_dir: Path, *extra_args: str) -> tuple[dict, str]:
    out_json = ev_dir / "skills-value-summary.json"
    cmd = [
        sys.executable, str(TOOLS / "compare_skills_value.py"),
        "--evidence-dir", str(ev_dir),
        "--output", str(out_json),
        *extra_args,
    ]
    proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return json.loads(out_json.read_text()), proc.stdout


# ---------------------------------------------------------------------------
# Test cases.
# ---------------------------------------------------------------------------


def smoke_classifier_validity() -> None:
    print("[1/10] _classify_validity contract...")
    cases = [
        ("infra_fail (no signals)", "infra_fail",
         dict(model=None, tokens_total=0, artifacts=[], kernel_path="")),
        ("clean ok", "ok",
         dict(model="dashscope/glm-5", tokens_total=140311,
              artifacts=["kernel.py"], kernel_path="teams/abs/kernel.py")),
        ("incomplete: model only", "incomplete",
         dict(model="glm-5", tokens_total=0, artifacts=[], kernel_path="")),
        ("incomplete: tokens but no model", "incomplete",
         dict(model=None, tokens_total=5000, artifacts=[], kernel_path="")),
        ("incomplete: artifact only", "incomplete",
         dict(model=None, tokens_total=0, artifacts=["kernel.py"],
              kernel_path="")),
        ("incomplete: kernel path only", "incomplete",
         dict(model="", tokens_total=0, artifacts=[], kernel_path="kernel.py")),
    ]
    for label, want, overrides in cases:
        ev = _ev_template(
            mode="off", op="abs", dtype="float16",
            verification_status="fail", score_accepted=False,
            semantic_passed=False, static_verify="fail", **overrides,
        )
        got = csv_mod._classify_validity(ev)
        assert got == want, f"{label}: want={want} got={got}"
        print(f"  [OK] {label}: {got}")


def smoke_classifier_failure_mode() -> None:
    print("[2/10] _classify_failure_mode covers every F-code...")
    # F13: infra_fail — no model, no tokens, no artifacts, no kernel.
    ev_f13 = _ev_template(
        mode="off", op="x", dtype="float16",
        model=None, tokens_total=0, artifacts=[], kernel_path="",
        verification_status="fail", score_accepted=False,
        semantic_passed=False, static_verify="fail",
    )
    assert csv_mod._classify_failure_mode(ev_f13) == "F13_infra_fail"
    # F12: incomplete — only model, no tokens, no artifacts.
    ev_f12 = _ev_template(
        mode="off", op="x", dtype="float16",
        model="some-model", tokens_total=0, artifacts=[], kernel_path="",
        verification_status="fail", score_accepted=False,
        semantic_passed=False, static_verify="fail",
    )
    assert csv_mod._classify_failure_mode(ev_f12) == "F12_incomplete"
    # F10: harness executed (model + tokens) but no kernel artifact.
    ev_f10 = _ev_template(
        mode="off", op="x", dtype="float16",
        model="m", tokens_total=10000, artifacts=["design.md"],
        kernel_path="",
        verification_status="fail", score_accepted=False,
        semantic_passed=False, static_verify="fail",
    )
    assert csv_mod._classify_failure_mode(ev_f10) == "F10_no_artifact"
    # F7: kernel exists, static_verify fails.
    ev_f7 = _ev_template(
        mode="off", op="x", dtype="float16",
        model="m", tokens_total=10000, artifacts=["kernel.py"],
        kernel_path="t/x/kernel.py",
        verification_status="fail", score_accepted=False,
        semantic_passed=False, static_verify="fail",
    )
    assert csv_mod._classify_failure_mode(ev_f7) == "F7_static"
    # F9: static passes, semantic_check.passed is False.
    ev_f9 = _ev_template(
        mode="off", op="x", dtype="float16",
        model="m", tokens_total=10000, artifacts=["kernel.py"],
        kernel_path="t/x/kernel.py",
        verification_status="fail", score_accepted=False,
        semantic_passed=False, static_verify="pass",
    )
    assert csv_mod._classify_failure_mode(ev_f9) == "F9_semantic"
    # F8: static + semantic pass, simulator reports fail.
    ev_f8 = _ev_template(
        mode="off", op="x", dtype="float16",
        model="m", tokens_total=10000, artifacts=["kernel.py"],
        kernel_path="t/x/kernel.py",
        verification_status="fail", score_accepted=False,
        semantic_passed=True, static_verify="pass",
        verification_mode="simulator",
    )
    assert csv_mod._classify_failure_mode(ev_f8) == "F8_correctness"
    # F0: didn't pass, but doesn't fit any rule — e.g. score not
    # accepted, no failing signal in any of the explicit fields.
    ev_f0 = _ev_template(
        mode="off", op="x", dtype="float16",
        model="m", tokens_total=10000, artifacts=["kernel.py"],
        kernel_path="t/x/kernel.py",
        verification_status="pass", score_accepted=False,
        semantic_passed=True, static_verify="pass",
        verification_mode="static_only",
    )
    assert csv_mod._classify_failure_mode(ev_f0) == "F0_unknown"
    # None: cell actually passed.
    ev_pass = _ev_template(
        mode="on", op="x", dtype="float16",
        model="m", tokens_total=10000, artifacts=["kernel.py"],
        kernel_path="t/x/kernel.py",
        verification_status="pass", score_accepted=True,
        semantic_passed=True, static_verify="pass",
    )
    assert csv_mod._classify_failure_mode(ev_pass) is None
    print("  [OK] all F-codes + pass fast-path")


def smoke_attempts_to_pass() -> None:
    print("[3/10] _attempts_to_pass...")
    base = dict(
        mode="on", op="x", dtype="float16", model="m", tokens_total=10000,
        artifacts=["kernel.py"], kernel_path="t/x/kernel.py",
        verification_status="pass", score_accepted=True,
        semantic_passed=True, static_verify="pass",
    )
    # Attempt 1 already passes.
    ev = _ev_template(attempts=[
        {"outcome": "pass", "kernel_found": True},
    ], **base)
    assert csv_mod._attempts_to_pass(ev) == 1
    # First fails, second is incomplete (no kernel), third passes.
    ev = _ev_template(attempts=[
        {"outcome": "fail", "kernel_found": False},
        {"outcome": "fail", "kernel_found": False},
        {"outcome": "pass", "kernel_found": True},
    ], **base)
    assert csv_mod._attempts_to_pass(ev) == 3
    # Pass after a fail-with-kernel (the failed attempt's
    # kernel_found shouldn't count as a usable kernel; only an
    # attempt that's both non-fail AND has kernel_found counts).
    ev = _ev_template(attempts=[
        {"outcome": "fail", "kernel_found": True},
        {"outcome": "pass", "kernel_found": True},
    ], **base)
    assert csv_mod._attempts_to_pass(ev) == 2
    # No attempts at all.
    ev = _ev_template(attempts=[], **base)
    assert csv_mod._attempts_to_pass(ev) is None
    # All attempts fail.
    ev = _ev_template(attempts=[
        {"outcome": "fail", "kernel_found": False},
        {"outcome": "fail", "kernel_found": False},
    ], **base)
    assert csv_mod._attempts_to_pass(ev) is None
    # Malformed (non-dict) entries are skipped.
    ev = _ev_template(attempts=[
        "not-a-dict", None,
        {"outcome": "pass", "kernel_found": True},
    ], **base)
    assert csv_mod._attempts_to_pass(ev) == 3
    print("  [OK] attempts_to_pass all branches")


def smoke_wilson_ci() -> None:
    print("[4/10] _wilson_ci bounds...")
    # 0/0 → None
    assert csv_mod._wilson_ci(0, 0) is None
    # 11/12: classic small-sample case; the bare 92% reads precise
    # but the CI ~[65%, 99%] makes the uncertainty visible.
    lo, hi = csv_mod._wilson_ci(11, 12)
    assert 0.60 <= lo <= 0.70, f"11/12 low: {lo}"
    assert 0.95 <= hi <= 1.00, f"11/12 high: {hi}"
    # 0/8: floor is 0 but the upper bound rules out everything
    # above ~32%.
    lo, hi = csv_mod._wilson_ci(0, 8)
    assert lo == 0.0
    assert 0.30 <= hi <= 0.35, f"0/8 high: {hi}"
    # 8/8: ceil is 1 but the lower bound is around ~63%.
    lo, hi = csv_mod._wilson_ci(8, 8)
    assert 0.60 <= lo <= 0.70, f"8/8 low: {lo}"
    assert hi == 1.0
    # 1/1: degenerate — Wilson still works.
    lo, hi = csv_mod._wilson_ci(1, 1)
    assert hi == 1.0 and lo > 0.0
    print("  [OK] Wilson CI on 0/0, 0/8, 8/8, 11/12, 1/1")


def smoke_no_clean_baseline() -> None:
    print("[5/10] aggregator on 'no clean off-baseline'...")
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        on = _ev_template(
            mode="on", op="abs", dtype="float16",
            model="dashscope/glm-5", tokens_total=140311,
            artifacts=["kernel.py"], kernel_path="teams/abs/kernel.py",
            verification_status="pass", score_accepted=True,
            semantic_passed=True, static_verify="pass",
        )
        _write_ev(d, "abs-f16-generative.json", on)
        off = _ev_template(
            mode="off", op="abs", dtype="float16",
            model=None, tokens_total=0, artifacts=[], kernel_path="",
            verification_status="fail", score_accepted=False,
            semantic_passed=False, static_verify="fail",
        )
        _write_ev(d, "abs-f16-generative-cloud-default-off.json", off)
        summary, _ = _run_aggregator(d)
        p = summary["by_profile"]["cloud-default"]
        assert summary["schema_version"] == "2"
        assert summary["partial_run"] is False
        assert summary["legs_status"] is None
        assert p["pass_rate_off_clean"] is None
        assert p["pass_rate_off_clean_ci"] is None, (
            "Wilson CI must be None when denominator is 0"
        )
        # On CI is computed (1/1 -> Wilson [low, 1.0]).
        assert isinstance(p["pass_rate_on_ci"], list)
        assert p["pass_rate_on_ci"][1] == 1.0
        assert p["cells_off_infra_fail"] == 1
        assert p["viability_unlocked_clean"] == 0
        assert p["unresolved_due_to_off_infra"] == 1
        # Failure-mode counts: off has F13 from infra-fail BUT off's
        # validity is "infra_fail" so it should NOT land in
        # failure_mode_counts_off; verify the off dict is empty.
        assert p["failure_mode_counts_off"] == {}
        assert p["failure_mode_counts_on"] == {}, (
            "on passed; no failure-mode chips expected"
        )
        # Attempts-to-pass: skills-on passed; skills-off never
        # produced a clean baseline so off mean is None.
        assert p["attempts_to_pass_on_n"] == 1
        assert p["attempts_to_pass_off_clean_n"] == 0
        assert p["attempts_to_pass_off_clean_mean"] is None
        # Freshness still works.
        assert p["on_last_run_at"] is not None
        assert p["off_max_staleness_days"] is None
        print("  no-clean-baseline: PASSED")


def smoke_clean_baseline_unlock() -> None:
    print("[6/10] aggregator on 'clean off-baseline + unlock'...")
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        # Cell 1: on passes (attempts_to_pass=2), off has clean
        # baseline that fails with F8.
        on = _ev_template(
            mode="on", op="abs", dtype="float16",
            model="dashscope/glm-5", tokens_total=140311,
            artifacts=["kernel.py"], kernel_path="teams/abs/kernel.py",
            verification_status="pass", score_accepted=True,
            semantic_passed=True, static_verify="pass",
            attempts=[
                {"outcome": "fail", "kernel_found": False},
                {"outcome": "pass", "kernel_found": True},
            ],
            date="2026-05-19T12:00:00Z",
        )
        _write_ev(d, "abs-f16-generative.json", on)
        off = _ev_template(
            mode="off", op="abs", dtype="float16",
            model="dashscope/glm-5", tokens_total=42000,
            artifacts=["kernel.py"], kernel_path="teams/abs/kernel.py",
            verification_status="fail", score_accepted=False,
            semantic_passed=True, static_verify="pass",
            verification_mode="simulator",
            attempts=[{"outcome": "fail", "kernel_found": True}],
            date="2026-05-19T13:00:00Z",
        )
        _write_ev(d, "abs-f16-generative-cloud-default-off.json", off)
        # Cell 2: both legs pass (off is a clean baseline that
        # passed on its 1st attempt).
        on2 = _ev_template(
            mode="on", op="add", dtype="float16",
            model="dashscope/glm-5", tokens_total=80000,
            artifacts=["kernel.py"], kernel_path="teams/add/kernel.py",
            verification_status="pass", score_accepted=True,
            semantic_passed=True, static_verify="pass",
            attempts=[{"outcome": "pass", "kernel_found": True}],
            date="2026-05-19T12:00:00Z",
        )
        _write_ev(d, "add-f16-generative.json", on2)
        off2 = _ev_template(
            mode="off", op="add", dtype="float16",
            model="dashscope/glm-5", tokens_total=33000,
            artifacts=["kernel.py"], kernel_path="teams/add/kernel.py",
            verification_status="pass", score_accepted=True,
            semantic_passed=True, static_verify="pass",
            attempts=[{"outcome": "pass", "kernel_found": True}],
            date="2026-05-19T13:00:00Z",
        )
        _write_ev(d, "add-f16-generative-cloud-default-off.json", off2)
        summary, _ = _run_aggregator(d)
        p = summary["by_profile"]["cloud-default"]
        assert p["cells_compared"] == 2
        assert p["cells_off_ok"] == 2
        assert p["pass_off_clean"] == 1
        assert p["pass_rate_off_clean"] == 0.5
        assert p["viability_unlocked_clean"] == 1
        # New: failure-mode counts. abs cell off-leg failed with F8
        # (simulator status=fail, static+semantic ok).
        assert p["failure_mode_counts_off"] == {"F8_correctness": 1}
        assert p["failure_mode_counts_on"] == {}
        # New: attempts-to-pass. on legs passed cells 1 and 2 with
        # attempts 2 and 1 respectively; off-clean passed only cell 2
        # with attempt 1.
        assert p["attempts_to_pass_on_n"] == 2
        assert p["attempts_to_pass_on_mean"] == 1.5
        assert p["attempts_to_pass_on_median"] in (1, 2)
        assert p["attempts_to_pass_off_clean_n"] == 1
        assert p["attempts_to_pass_off_clean_mean"] == 1.0
        # New: Wilson CIs are populated and bracket the point estimate.
        ci_on = p["pass_rate_on_ci"]
        assert isinstance(ci_on, list) and len(ci_on) == 2
        assert ci_on[0] <= 1.0 <= ci_on[1]
        ci_off = p["pass_rate_off_clean_ci"]
        assert isinstance(ci_off, list) and len(ci_off) == 2
        assert ci_off[0] <= 0.5 <= ci_off[1]
        # Existing freshness checks still work.
        assert p["on_last_run_at"].startswith("2026-05-19T12:00:00")
        assert p["off_last_run_at"].startswith("2026-05-19T13:00:00")
        assert isinstance(p["off_max_staleness_days"], int)
        # Per-cell deltas record the new failure_off field.
        c_abs = next(r for r in summary["cells"] if r["op"] == "abs")
        assert c_abs["on_date"] == "2026-05-19T12:00:00Z"
        assert c_abs["off_date"] == "2026-05-19T13:00:00Z"
        assert c_abs["failure_on"] is None  # on passed
        assert c_abs["failure_off"] == "F8_correctness"
        print("  clean-baseline-unlock: PASSED")


def smoke_partial_run_flag() -> None:
    print("[7/10] aggregator with --partial-run + --legs-status-file...")
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        on = _ev_template(
            mode="on", op="abs", dtype="float16",
            model="dashscope/glm-5", tokens_total=140311,
            artifacts=["kernel.py"], kernel_path="teams/abs/kernel.py",
            verification_status="pass", score_accepted=True,
            semantic_passed=True, static_verify="pass",
        )
        _write_ev(d, "abs-f16-generative.json", on)
        off = _ev_template(
            mode="off", op="abs", dtype="float16",
            model="dashscope/glm-5", tokens_total=42000,
            artifacts=["partial.py"], kernel_path="teams/abs/kernel.py",
            verification_status="fail", score_accepted=False,
            semantic_passed=False, static_verify="fail",
        )
        _write_ev(d, "abs-f16-generative-cloud-default-off.json", off)
        legs_path = d / "legs.json"
        legs_path.write_text(json.dumps({
            "needs": {"nightly-gate": "cancelled",
                      "local-stability-gate": "success"},
            "partial_run": True,
            "legs": [
                {"name": "nightly-gate (on)", "conclusion": "cancelled"},
                {"name": "local-stability-gate (local-qwen-coder-7b, on)",
                 "conclusion": "success"},
            ],
        }, indent=2))
        summary, _ = _run_aggregator(
            d, "--partial-run", "--legs-status-file", str(legs_path),
        )
        assert summary["partial_run"] is True
        legs_status = summary["legs_status"]
        assert isinstance(legs_status, dict)
        assert legs_status["partial_run"] is True
        names = [l["name"] for l in legs_status["legs"]]
        assert "nightly-gate (on)" in names
        print("  partial-run flag: PASSED")


def smoke_merge_helper() -> None:
    print("[8/10] merge_evidence_artifacts.sh preserves fresh files...")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        art = root / "artifacts"
        ev_out = root / "evidence"
        art.mkdir(); ev_out.mkdir()
        legs = {
            "evidence-cloud-off": {
                "abs-f16-generative-cloud-default-off.json": {
                    "kind": "generative", "date": "2026-05-19T00:00:00Z",
                    "model": "dashscope/glm-5", "tokens": {"total": 100},
                    "tag": "FRESH-cloud-off",
                },
            },
            "evidence-cloud-on": {
                "abs-f16-generative-cloud-default-off.json": {
                    "kind": "generative", "date": "2026-05-12T00:00:00Z",
                    "model": None, "tokens": {"total": 0},
                    "tag": "STALE-cloud-off-from-cloud-on-checkout",
                },
                "abs-f16-generative.json": {
                    "kind": "generative", "date": "2026-05-19T00:00:00Z",
                    "model": "dashscope/glm-5", "tokens": {"total": 140000},
                    "tag": "FRESH-cloud-on",
                },
            },
            "evidence-local-qwen-coder-7b-on": {
                "abs-f16-generative-cloud-default-off.json": {
                    "kind": "generative", "date": "2026-05-12T00:00:00Z",
                    "model": None, "tokens": {"total": 0},
                    "tag": "STALE-cloud-off-from-local-on-checkout",
                },
                "abs-f16-generative-local-qwen-coder-7b-on.json": {
                    "kind": "generative", "date": "2026-05-19T00:00:00Z",
                    "model": "ollama/qwen2.5-coder:7b",
                    "tokens": {"total": 12000},
                    "tag": "FRESH-local-qwen-on",
                },
            },
        }
        for leg, files in legs.items():
            d = art / leg
            d.mkdir()
            for name, payload in files.items():
                (d / name).write_text(json.dumps(payload))
        subprocess.run(
            ["bash", str(TOOLS / "merge_evidence_artifacts.sh"),
             str(art), str(ev_out)],
            check=True, capture_output=True, text=True,
        )
        merged_off = json.loads(
            (ev_out / "abs-f16-generative-cloud-default-off.json").read_text()
        )
        assert merged_off["tag"] == "FRESH-cloud-off"
        merged_on = json.loads(
            (ev_out / "abs-f16-generative.json").read_text()
        )
        assert merged_on["tag"] == "FRESH-cloud-on"
        merged_local = json.loads(
            (ev_out / "abs-f16-generative-local-qwen-coder-7b-on.json").read_text()
        )
        assert merged_local["tag"] == "FRESH-local-qwen-on"
        print("  merge helper: PASSED (no stale-overwrite race)")


def smoke_detect_partial_run() -> None:
    print("[9/10] detect_partial_run.py classifies CI jobs correctly...")
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        jobs_ok = {"jobs": [
            {"name": "pr-gate", "conclusion": "success"},
            {"name": "merge-gate", "conclusion": "success"},
            {"name": "nightly-gate (on)", "conclusion": "success"},
            {"name": "nightly-gate (off)", "conclusion": "success"},
            {"name": "local-stability-gate (local-qwen-coder-7b, on)",
             "conclusion": "success"},
            {"name": "skills-value-report", "conclusion": "in_progress"},
        ]}
        (d / "jobs_ok.json").write_text(json.dumps(jobs_ok))
        out_ok = d / "legs_ok.json"
        gh_out = d / "github_output_ok"
        gh_out.write_text("")
        env = os.environ.copy()
        env["NIGHTLY_RESULT"] = "success"
        env["LOCAL_RESULT"] = "success"
        subprocess.run([
            sys.executable, str(TOOLS / "detect_partial_run.py"),
            "--jobs-file", str(d / "jobs_ok.json"),
            "--output", str(out_ok),
            "--github-output", str(gh_out),
        ], env=env, check=True, capture_output=True, text=True)
        rec_ok = json.loads(out_ok.read_text())
        assert rec_ok["partial_run"] is False, rec_ok
        assert "partial_run=false" in gh_out.read_text()
        assert len(rec_ok["legs"]) == 3, rec_ok["legs"]
        jobs_cancelled = {"jobs": [
            {"name": "nightly-gate (on)", "conclusion": "cancelled"},
            {"name": "nightly-gate (off)", "conclusion": "success"},
            {"name": "local-stability-gate (local-llama-3.1-8b, off)",
             "conclusion": "cancelled"},
        ]}
        (d / "jobs_cancel.json").write_text(json.dumps(jobs_cancelled))
        out_c = d / "legs_cancel.json"
        gh_out_c = d / "github_output_cancel"
        gh_out_c.write_text("")
        env["NIGHTLY_RESULT"] = "cancelled"
        env["LOCAL_RESULT"] = "cancelled"
        subprocess.run([
            sys.executable, str(TOOLS / "detect_partial_run.py"),
            "--jobs-file", str(d / "jobs_cancel.json"),
            "--output", str(out_c),
            "--github-output", str(gh_out_c),
        ], env=env, check=True, capture_output=True, text=True)
        rec_c = json.loads(out_c.read_text())
        assert rec_c["partial_run"] is True
        assert "partial_run=true" in gh_out_c.read_text()
        assert rec_c["needs"]["nightly-gate"] == "cancelled"
        print("  detect_partial_run: PASSED (both branches)")


def smoke_dashboard_payload() -> None:
    print("[10/10] dashboard inlines the new explainability/efficiency keys...")
    original = (REPO / "evidence" / "skills-value-summary.json").read_text()
    try:
        synth = {
            "schema_version": "2",
            "generated_at": "2026-05-19T00:00:00Z",
            "partial_run": True,
            "legs_status": {
                "partial_run": True,
                "needs": {"nightly-gate": "cancelled",
                          "local-stability-gate": "success"},
                "legs": [
                    {"name": "nightly-gate (on)", "conclusion": "cancelled"},
                ],
            },
            "cells": [],
            "by_profile": {
                "cloud-default": {
                    "cells_total": 12, "cells_compared": 12,
                    "pass_on": 11, "pass_off": 0,
                    "pass_off_clean": 0,
                    "cells_off_ok": 12,
                    "cells_off_infra_fail": 0,
                    "cells_off_incomplete": 0,
                    "tokens_on_sum": 1_400_000, "tokens_off_sum": 400_000,
                    "cost_on_sum": 0, "cost_off_sum": 0,
                    "elapsed_on_sum": 2400.0, "elapsed_off_sum": 600.0,
                    "viability_unlocked_count": 11,
                    "viability_unlocked_clean": 11,
                    "unresolved_due_to_off_infra": 0,
                    "model": "dashscope/glm-5",
                    "pass_rate_on": 0.917, "pass_rate_off": 0.0,
                    "pass_rate_off_clean": 0.0,
                    "tokens_delta_avg": 83333,
                    "cost_delta_avg_usd": 0,
                    "elapsed_delta_avg_s": 150.0,
                    "on_last_run_at": "2026-05-19T12:00:00Z",
                    "off_last_run_at": "2026-05-19T13:00:00Z",
                    "off_max_staleness_days": 0,
                    "failure_mode_counts_on": {"F8_correctness": 1},
                    "failure_mode_counts_off": {
                        "F10_no_artifact": 8, "F7_static": 4,
                    },
                    "attempts_to_pass_on_mean": 1.18,
                    "attempts_to_pass_on_median": 1,
                    "attempts_to_pass_on_n": 11,
                    "attempts_to_pass_off_clean_mean": None,
                    "attempts_to_pass_off_clean_median": None,
                    "attempts_to_pass_off_clean_n": 0,
                    "pass_rate_on_ci": [0.646, 0.985],
                    "pass_rate_off_clean_ci": [0.0, 0.243],
                },
            },
        }
        (REPO / "evidence" / "skills-value-summary.json").write_text(
            json.dumps(synth, indent=2)
        )
        out = Path(tempfile.mkdtemp())
        try:
            subprocess.run([
                sys.executable, str(TOOLS / "generate_dashboard.py"),
                "--output-dir", str(out),
            ], check=True, capture_output=True, text=True)
            h = (out / "index.html").read_text()
        finally:
            shutil.rmtree(out, ignore_errors=True)
    finally:
        (REPO / "evidence" / "skills-value-summary.json").write_text(original)
    # Inlined summary payload contains the new fields.
    expects_inlined = [
        '"partial_run": true',
        '"on_last_run_at": "2026-05-19T12:00:00Z"',
        '"off_max_staleness_days": 0',
        '"failure_mode_counts_on"',
        '"F8_correctness": 1',
        '"F10_no_artifact": 8',
        '"attempts_to_pass_on_mean": 1.18',
        '"pass_rate_on_ci"',
    ]
    for needle in expects_inlined:
        assert needle in h, f"missing inlined payload key: {needle!r}"
        print(f"  [OK] HTML contains {needle!r}")
    # Renderer helpers / chip CSS / tip entries are present.
    expects_static = [
        "svb-card-freshness",
        "fmtAgo",
        "Partial nightly",
        "svb-fchips",
        "SVB_FCODE_LABELS",
        "fmtCiPct",
        "Attempts to pass",
        "failureModes",
        "attemptsToPass",
        "passRateCi",
    ]
    for needle in expects_static:
        assert needle in h, f"missing template symbol: {needle!r}"
        print(f"  [OK] template carries {needle!r}")


def main() -> int:
    smoke_classifier_validity(); print()
    smoke_classifier_failure_mode(); print()
    smoke_attempts_to_pass(); print()
    smoke_wilson_ci(); print()
    smoke_no_clean_baseline(); print()
    smoke_clean_baseline_unlock(); print()
    smoke_partial_run_flag(); print()
    smoke_merge_helper(); print()
    smoke_detect_partial_run(); print()
    smoke_dashboard_payload(); print()
    print("ALL SMOKE TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
