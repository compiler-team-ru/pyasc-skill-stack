#!/usr/bin/env python3
"""Compiler SIMD vector-fusion (--cce-simd-vf-fusion) A/B demo.

pyasc2 lowers to the AscendC *high-level* API and deliberately uses no
micro-api: the official positioning is that SIMD vector fusion is delegated to
the bisheng compiler flag ``--cce-simd-vf-fusion=true``. pyasc's JIT does not
pass that flag today (CANN's ascend950 default is ``--cce-simd-vf-fusion=false``)
while the hand-written ops-math/ops-nn references compile it *on* via their
``ascendc_config.json``.

This harness verifies the positioning with a clean single-variable A/B on the
*same* generated kernel: compile + measure camodel ticks with fusion **off**
(the current default) vs **on**, on the pinned ``Ascend950PR_9599`` camodel. The
flag is the only thing that changes between the two runs (see
``pyasc_gen_runner.measure(vf_fusion=...)``). The canonical AscendC reference
(already fusion-on) is measured too, for ratio-to-reference context: does turning
the flag on close the gap to the hand-written operator?

Per cell we record:
  - ``ticks_off`` / ``ticks_on``  -> camodel Total tick, median of N runs
  - ``fusion_speedup = ticks_off / ticks_on`` (>1 => fusion helped)
  - ``ratio_off`` / ``ratio_on`` = ``ref_ticks / ticks_{off,on}``

Honest by construction: if the generated side cannot be measured (codegen or
environment fault) the cell is recorded ``status: gen_blocked`` with the ref
captured, never fabricated.

Usage::

    python demo_vf_fusion.py --cell tanh/float16
    python demo_vf_fusion.py --all
    python demo_vf_fusion.py --all --runs 1        # fast
"""

from __future__ import annotations

import argparse
import datetime as _dt
import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = Path(__file__).resolve().parent
EVIDENCE = REPO_ROOT / "evidence" / "vf-fusion"

# Reuse the perf orchestrator's cell map + runner handles (it loads
# ascendc_ref_runner / pyasc_gen_runner and defines the canonical CELLS dict).
_spec = importlib.util.spec_from_file_location(
    "demo_vector_ops", TOOLS_DIR / "demo_vector_ops.py")
_dvo = importlib.util.module_from_spec(_spec)
sys.modules["demo_vector_ops"] = _dvo
_spec.loader.exec_module(_dvo)

CELLS = _dvo.CELLS
_ref = _dvo._ref
_gen = _dvo._gen
GATE = _dvo.GATE

# fusion_speedup classification band (camodel tick noise is well under 1%, so a
# +-2% deadband keeps single-op kernels that genuinely don't fuse out of the
# "improved"/"regressed" buckets).
IMPROVED_AT = 1.02
REGRESSED_AT = 0.98


def _classify(speedup: float) -> str:
    if speedup >= IMPROVED_AT:
        return "improved"
    if speedup <= REGRESSED_AT:
        return "regressed"
    return "neutral"


def run_cell(cell: str, *, runs: int, verbose: bool) -> dict:
    if cell not in CELLS:
        raise KeyError(f"unknown cell '{cell}' (known: {sorted(CELLS)})")
    spec = CELLS[cell]
    shape = spec["shape"]
    op_name = cell.split("/")[0]

    # Canonical reference (always fusion-on) — best-effort context. A ref build
    # fault must not erase a valid off/on A/B, so it is recorded as null rather
    # than aborting the cell.
    ref_ticks = None
    ref_detail = None
    try:
        ref_detail = _ref.measure(
            spec["ref_op"], spec["ref_dtype"], shape,
            ops_math=_ref.DEFAULT_OPS_MATH, ascend=_ref.DEFAULT_ASCEND,
            runs=runs, verbose=verbose,
        )
        ref_ticks = ref_detail["ref_ticks"]
    except Exception as exc:  # noqa: BLE001 - ref is context only
        if verbose:
            print(f"[vf-fusion] {cell}: reference unavailable ({exc}); "
                  f"recording A/B without ratio-to-ref")

    arg_specs = _gen.arg_specs_for(cell, shape, spec["gen_dtype"])

    def _measure(vf: str) -> dict:
        return _gen.measure(
            spec["kernel"], shape, spec["gen_dtype"],
            eval_root=_gen.DEFAULT_EVAL_ROOT, ascend=_gen.DEFAULT_ASCEND,
            runs=runs, python="python3.11", verbose=verbose,
            arg_specs=arg_specs, op_name=op_name, vf_fusion=vf,
        )

    try:
        gen_off = _measure("false")
        gen_on = _measure("true")
    except Exception as exc:  # noqa: BLE001 - honest blocker, keep ref
        return {
            "cell": cell,
            "op": op_name,
            "dtype": spec["gen_dtype"],
            "shape": shape,
            "arch": _ref.ARCH_PIN,
            "ref_ticks": ref_ticks,
            "ticks_off": None,
            "ticks_on": None,
            "fusion_speedup": None,
            "ratio_off": None,
            "ratio_on": None,
            "gate": GATE,
            "status": "gen_blocked",
            "gen_blocker": str(exc),
            "flag": "--cce-simd-vf-fusion",
            "ref_detail": ref_detail,
            "gen_detail_off": None,
            "gen_detail_on": None,
        }

    ticks_off = gen_off["gen_ticks"]
    ticks_on = gen_on["gen_ticks"]
    speedup = round(ticks_off / ticks_on, 4) if ticks_on else None
    ratio_off = round(ref_ticks / ticks_off, 4) if (ref_ticks and ticks_off) else None
    ratio_on = round(ref_ticks / ticks_on, 4) if (ref_ticks and ticks_on) else None
    return {
        "cell": cell,
        "op": op_name,
        "dtype": spec["gen_dtype"],
        "shape": shape,
        "arch": _ref.ARCH_PIN,
        "ref_ticks": ref_ticks,
        "ticks_off": ticks_off,
        "ticks_on": ticks_on,
        "fusion_speedup": speedup,
        "ratio_off": ratio_off,
        "ratio_on": ratio_on,
        "gate": GATE,
        "status": _classify(speedup) if speedup is not None else "gen_blocked",
        "flag": "--cce-simd-vf-fusion",
        "ref_detail": ref_detail,
        "gen_detail_off": gen_off,
        "gen_detail_on": gen_on,
    }


def _print_table(results: list[dict]) -> None:
    print()
    print(f"| {'cell':<24} | {'off':>8} | {'on':>8} | {'speedup':>7} | "
          f"{'r->ref off':>10} | {'r->ref on':>9} | {'status':<10} |")
    print(f"|{'-' * 26}|{'-' * 10}|{'-' * 10}|{'-' * 9}|"
          f"{'-' * 12}|{'-' * 11}|{'-' * 12}|")
    for r in results:
        if r.get("status") == "gen_blocked":
            off_s, on_s, sp_s, ro_s, rn_s = "BLOCKED", "  -  ", "  -  ", "  -  ", "  -  "
        else:
            off_s = str(r["ticks_off"])
            on_s = str(r["ticks_on"])
            sp_s = f"{r['fusion_speedup']:.3f}" if r["fusion_speedup"] is not None else "  -  "
            ro_s = f"{r['ratio_off']:.2f}" if r["ratio_off"] is not None else "  -  "
            rn_s = f"{r['ratio_on']:.2f}" if r["ratio_on"] is not None else "  -  "
        print(f"| {r['cell']:<24} | {off_s:>8} | {on_s:>8} | {sp_s:>7} | "
              f"{ro_s:>10} | {rn_s:>9} | {r['status']:<10} |")
    print()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="--cce-simd-vf-fusion off-vs-on A/B for pyasc-lowered kernels")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--cell", help="e.g. tanh/float16")
    g.add_argument("--all", action="store_true", help="run all demo cells")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("-q", "--quiet", action="store_true")
    args = ap.parse_args(argv)

    verbose = not args.quiet
    cells = sorted(CELLS) if args.all else [args.cell]

    results: list[dict] = []
    for cell in cells:
        if verbose:
            print(f"\n=== vf-fusion A/B: {cell} ===")
        try:
            results.append(run_cell(cell, runs=args.runs, verbose=verbose))
        except KeyError as exc:
            print(exc, file=sys.stderr)
            return 2

    if results:
        _print_table(results)
        EVIDENCE.mkdir(parents=True, exist_ok=True)
        ts = _dt.datetime.now().strftime("%Y%m%dT%H%M%S")
        for r in results:
            tag = r["cell"].replace("/", "-")
            out = EVIDENCE / f"{tag}-{ts}.json"
            out.write_text(json.dumps(r, indent=2) + "\n")
            print(f"[evidence] {r['cell']} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
