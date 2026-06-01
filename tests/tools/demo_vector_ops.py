#!/usr/bin/env python3
"""Phase 11 — perf-vs-AscendC demo orchestrator.

For each cell, measures the canonical ops-math AscendC reference and the
skill-stack-generated pyasc kernel on the same camodel (``Ascend950PR_9599``)
at the same shape, computes ``ratio = ref_ticks / gen_ticks``, and prints the
demo table::

    | cell             | ref_ticks | gen_ticks | ratio | gate |
    |------------------|-----------|-----------|-------|------|
    | abs/float16      |      4345 |      4686 |  0.93 | PASS |

Gate: ``ratio >= 0.70`` (generated kernel within ~30% of hand-written AscendC).

Both ticks are camodel ``Total tick`` for a single launch (symmetric across the
two implementations). The reference is the *canonical* compiled ops-math
operator (no hand-rolled fallback); see ascendc_ref_runner.py. The generated
side is the cached oracle_guided kernel by default; ``--regen`` re-runs the
opencode agent first (live reproduction).

Usage::

    python demo_vector_ops.py --cell abs/float16
    python demo_vector_ops.py --all
    python demo_vector_ops.py --all --runs 1           # fast demo
    python demo_vector_ops.py --cell abs/float16 --regen
"""

from __future__ import annotations

import argparse
import datetime as _dt
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PERF_DIR = Path(__file__).resolve().parent / "perf"
EVIDENCE = REPO_ROOT / "evidence" / "perf-vs-ascendc"
GATE = 0.70


def _load(mod_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod

_ref = _load("ascendc_ref_runner", PERF_DIR / "ascendc_ref_runner.py")
_gen = _load("pyasc_gen_runner", PERF_DIR / "pyasc_gen_runner.py")

KERNELS_ROOT = REPO_ROOT / "teams" / "pyasc-kernel-dev-team" / "kernels"

# cell -> {ref op/dtype, gen kernel path + dtype, comparability shape}
CELLS = {
    "abs/float16": {
        "ref_op": "abs", "ref_dtype": "f16",
        "kernel": KERNELS_ROOT / "abs_f16" / "kernel.py", "gen_dtype": "float16",
        "shape": [32, 4096],
    },
    "add/float16": {
        "ref_op": "add", "ref_dtype": "f16",
        "kernel": KERNELS_ROOT / "add_f16" / "kernel.py", "gen_dtype": "float16",
        "shape": [32, 4096],
    },
    "reduce_sum/float32": {
        "ref_op": "reduce_sum", "ref_dtype": "f32",
        "kernel": KERNELS_ROOT / "reduce_sum_f32" / "kernel.py", "gen_dtype": "float32",
        "shape": [32, 4096],
    },
}


def _regen(cell: str, dest_kernel: Path, *, verbose: bool) -> dict:
    """Live reproduction: re-run the opencode agent for this cell with the
    ``oracle_guided`` prompt + skills on, then land the freshly generated
    kernel at the cached location ``dest_kernel`` so the gen runner measures
    the agent's output (not a stale checked-in file).

    The collect harness deletes its temp project after each attempt, so we
    archive the winning kernel via ``--archive-dir`` and copy it into place.
    Best-effort and honest: raises on failure, never fabricates a kernel.
    """
    op, dtype = cell.split("/")
    script = REPO_ROOT / "tests" / "tools" / "collect_generative_evidence.py"
    archive_dir = REPO_ROOT / "evidence" / "perf-vs-ascendc" / "regen-archive" / cell.replace("/", "-")
    if archive_dir.exists():
        shutil.rmtree(archive_dir, ignore_errors=True)
    archive_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(script),
        "--op", op, "--dtype", dtype,
        "--prompt-variant", "oracle_guided",
        "--model-profile", "cloud-default",
        "--skills-mode", "on",
        "--max-attempts", "3",
        "--timeout", "420",
        "--archive-dir", str(archive_dir),
    ]
    if verbose:
        print(f"[regen] {cell}: {' '.join(cmd)}")
    r = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=3600)
    if r.returncode != 0:
        tail = "\n".join((r.stdout + r.stderr).splitlines()[-25:])
        raise RuntimeError(f"opencode regen for {cell} failed (exit {r.returncode}):\n{tail}")

    # The harness archives the winning kernel's directory under archive_dir
    # only on an overall pass; find the freshly generated kernel.py and land
    # it at the cached path the gen runner reads.
    archived = sorted(archive_dir.glob("**/kernel.py"))
    if not archived:
        tail = "\n".join((r.stdout + r.stderr).splitlines()[-25:])
        raise RuntimeError(
            f"opencode regen for {cell} reported success but no kernel.py was "
            f"archived under {archive_dir} (nothing to measure):\n{tail}"
        )
    src = archived[0]
    dest_kernel.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest_kernel)
    if verbose:
        print(f"[regen] {cell}: landed fresh kernel {src} -> {dest_kernel}")
    return {"archived_kernel": str(src), "landed_kernel": str(dest_kernel),
            "archive_dir": str(archive_dir)}


def run_cell(cell: str, *, runs: int, regen: bool, verbose: bool) -> dict:
    if cell not in CELLS:
        raise KeyError(f"unknown cell '{cell}' (known: {sorted(CELLS)})")
    spec = CELLS[cell]
    shape = spec["shape"]

    regen_info = None
    if regen:
        regen_info = _regen(cell, spec["kernel"], verbose=verbose)

    # Canonical reference is measured first and independently: it is the
    # ground truth and must never be skipped or faked (plan: canonical_only).
    ref = _ref.measure(
        spec["ref_op"], spec["ref_dtype"], shape,
        ops_math=_ref.DEFAULT_OPS_MATH, ascend=_ref.DEFAULT_ASCEND,
        runs=runs, verbose=verbose,
    )
    ref_ticks = ref["ref_ticks"]

    # The generated (pyasc) side can be blocked by environment-level codegen
    # faults (see evidence/perf-vs-ascendc/BLOCKER-gen-side-multiinput-reduction.md).
    # We surface that as a recorded ref-only row rather than fabricating ticks.
    try:
        gen = _gen.measure(
            spec["kernel"], shape, spec["gen_dtype"],
            eval_root=_gen.DEFAULT_EVAL_ROOT, ascend=_gen.DEFAULT_ASCEND,
            runs=runs, python="python3.11", verbose=verbose,
        )
    except Exception as exc:  # noqa: BLE001 - record honest blocker, keep ref
        return {
            "cell": cell,
            "shape": shape,
            "arch": _ref.ARCH_PIN,
            "ref_ticks": ref_ticks,
            "gen_ticks": None,
            "ratio": None,
            "gate": GATE,
            "passed": False,
            "status": "gen_blocked",
            "gen_blocker": str(exc),
            "kernel_mode": "regen" if regen else "cached",
            "regen": regen_info,
            "ref_detail": ref,
            "gen_detail": None,
        }

    gen_ticks = gen["gen_ticks"]
    ratio = ref_ticks / gen_ticks if gen_ticks else 0.0
    return {
        "cell": cell,
        "shape": shape,
        "arch": _ref.ARCH_PIN,
        "ref_ticks": ref_ticks,
        "gen_ticks": gen_ticks,
        "ratio": round(ratio, 4),
        "gate": GATE,
        "passed": ratio >= GATE,
        "status": "ok",
        "kernel_mode": "regen" if regen else "cached",
        "regen": regen_info,
        "ref_detail": ref,
        "gen_detail": gen,
    }


def _print_table(results: list[dict]) -> None:
    print()
    print(f"| {'cell':<20} | {'ref_ticks':>9} | {'gen_ticks':>9} | {'ratio':>5} | gate    |")
    print(f"|{'-' * 22}|{'-' * 11}|{'-' * 11}|{'-' * 7}|---------|")
    for r in results:
        if r.get("status") == "gen_blocked":
            gen_s, ratio_s, gate = "BLOCKED", "  -  ", "GEN-BLK"
        else:
            gen_s = str(r["gen_ticks"])
            ratio_s = f"{r['ratio']:>5.2f}"
            gate = "PASS" if r["passed"] else "FAIL"
        print(f"| {r['cell']:<20} | {r['ref_ticks']:>9} | {gen_s:>9} | "
              f"{ratio_s:>5} | {gate:<7} |")
    print()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Phase 11 perf-vs-AscendC demo")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--cell", help="e.g. abs/float16")
    g.add_argument("--all", action="store_true", help="run all demo cells")
    km = ap.add_mutually_exclusive_group()
    km.add_argument("--use-cached-kernel", action="store_true", default=True,
                    help="use the cached generated kernel (default)")
    km.add_argument("--regen", action="store_true",
                    help="re-run the opencode agent to regenerate the kernel first")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("-q", "--quiet", action="store_true")
    args = ap.parse_args(argv)

    cells = sorted(CELLS) if args.all else [args.cell]
    verbose = not args.quiet
    results: list[dict] = []
    failures: list[str] = []
    for cell in cells:
        try:
            res = run_cell(cell, runs=args.runs, regen=args.regen, verbose=verbose)
        except Exception as exc:  # noqa: BLE001 - surface per-cell blockers, keep going
            print(f"[BLOCKED] {cell}: {exc}", file=sys.stderr)
            failures.append(cell)
            continue
        results.append(res)

    if results:
        _print_table(results)
        EVIDENCE.mkdir(parents=True, exist_ok=True)
        ts = _dt.datetime.now().strftime("%Y%m%dT%H%M%S")
        for r in results:
            tag = r["cell"].replace("/", "-")
            out = EVIDENCE / f"{tag}-{ts}.json"
            out.write_text(json.dumps(r, indent=2) + "\n")
            print(f"[evidence] {r['cell']} -> {out}")

    if failures:
        print(f"\n[WARN] {len(failures)} cell(s) blocked: {', '.join(failures)}", file=sys.stderr)
    if any(not r["passed"] for r in results) or failures:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
