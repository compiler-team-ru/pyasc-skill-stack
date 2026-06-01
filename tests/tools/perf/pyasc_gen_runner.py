#!/usr/bin/env python3
"""Phase 11 — pyasc generated-side runner.

Measures the *per-launch* camodel tick count (``gen_ticks``) of a generated
pyasc kernel at a pinned shape, using the warm-up + ``current_tick()`` delta
pattern from ``docs/perf-methodology/ticks-calculation.md`` §6.3. This avoids
the cumulative ``Total tick:`` multi-launch trap: a single process does one
warm-up launch (populates the JIT/codegen cache, advances the counter) and
then N measured launches, reading ``asc.lib.runtime.current_tick()`` before and
after each.

Host runtime: python3.11 + the ``pyasc-v2-eval`` checkout (the built
``asc._C.libpyasc.cpython-311`` extension), CANN simulator libs for
``Ascend950PR_9599`` on ``LD_LIBRARY_PATH``. Both this and the AscendC
reference load a byte-identical ``libmodel_top.so`` camodel, so the ticks are
directly comparable.

Usage::

    python pyasc_gen_runner.py --cell abs/float16 --shape 32x4096
    python pyasc_gen_runner.py --kernel teams/.../abs_f16/kernel.py --shape 32x4096 --json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_EVAL_ROOT = Path(
    os.environ.get("PYASC_EVAL_ROOT", "/home/aloschilov/workspace/pyasc-v2-eval")
)
DEFAULT_ASCEND = Path(
    os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/cann-9.0.0")
)
GEN_EVIDENCE = REPO_ROOT / "evidence" / "perf" / "pyasc-gen"
BUILD_CACHE = REPO_ROOT / "evidence" / "perf" / "_build_cache"
ARCH_PIN = "Ascend950PR_9599"

# cell -> (cached kernel.py relative path, numpy dtype)
KERNELS_ROOT = REPO_ROOT / "teams" / "pyasc-kernel-dev-team" / "kernels"
CELL_TO_KERNEL = {
    "abs/float16": (KERNELS_ROOT / "abs_f16" / "kernel.py", "float16"),
    "add/float16": (KERNELS_ROOT / "add_f16" / "kernel.py", "float16"),
    "reduce_sum/float32": (KERNELS_ROOT / "reduce_sum_f32" / "kernel.py", "float32"),
}


class GenError(RuntimeError):
    pass


# The probe runs inside python3.11 with asc importable. It imports the kernel
# module from its file, finds the ``*_launch`` wrapper, builds inputs at the
# target shape, and does exactly ONE kernel launch + synchronize, then exits so
# the camodel prints ``Total tick:`` for that single launch. This is symmetric
# with the AscendC reference (one aclnn launch -> Total tick) and avoids the
# multi-launch-in-one-process instability with ``always_compile=True`` kernels.
# Determinism comes from running this probe in N separate processes (see
# ticks-calculation.md §3: for single-launch processes Total tick == per-launch
# ticks; §2: JIT compile does not advance the tick counter).
PROBE_TEMPLATE = r'''
import importlib.util, inspect, sys
import numpy as np
import asc.runtime.config as config
from asc.lib.runtime import synchronize

KERNEL_PATH = {kernel_path!r}
SHAPE = {shape!r}
DTYPE = "{dtype}"
BACKEND = "Model"
PLATFORM = "{platform}"

spec = importlib.util.spec_from_file_location("gen_kernel_mod", KERNEL_PATH)
mod = importlib.util.module_from_spec(spec)
sys.modules["gen_kernel_mod"] = mod
spec.loader.exec_module(mod)

config.set_platform(config.Backend(BACKEND), config.Platform(PLATFORM))

launch = None
for name in dir(mod):
    obj = getattr(mod, name)
    if callable(obj) and name.endswith("_launch"):
        launch = obj
        break
if launch is None:
    print("PROBE_ERROR no *_launch wrapper found in kernel module")
    sys.exit(3)

# Only the REQUIRED positional parameters are input tensors; parameters
# with defaults (e.g. a generated ``out_pad=OUT_PAD`` scalar) must keep
# their default, otherwise we'd pass an ndarray where a scalar is expected
# and the kernel silently runs a no-op (see Phase 11b reduce_sum probe).
_sig = inspect.signature(launch)
_input_params = [
    p for p in _sig.parameters.values()
    if p.default is inspect.Parameter.empty
    and p.kind in (inspect.Parameter.POSITIONAL_ONLY,
                   inspect.Parameter.POSITIONAL_OR_KEYWORD)
]
nparams = len(_input_params)
rng = np.random.default_rng(seed=2026)
npdt = np.float16 if DTYPE == "float16" else np.float32
def make_input():
    return (rng.random(SHAPE, dtype=np.float32) * 10 - 5).astype(npdt)
args = [make_input() for _ in range(nparams)]

launch(*args)
synchronize()
print(f"PROBE_DONE launch={{launch.__name__}} nparams={{nparams}}")
'''


def _probe_source(kernel_path: Path, shape: list[int], dtype: str, platform: str) -> str:
    return PROBE_TEMPLATE.format(
        kernel_path=str(kernel_path),
        shape=tuple(shape),
        dtype=dtype,
        platform=platform,
    )


_TICK_RE = re.compile(r"Total tick:\s*(\d+)")
_WALL_RE = re.compile(r"Model RUN TIME:\s*([\d.]+)")


def measure(kernel_path: Path, shape: list[int], dtype: str, *, eval_root: Path,
            ascend: Path, runs: int, python: str, verbose: bool) -> dict:
    if not kernel_path.exists():
        raise GenError(f"cached kernel not found: {kernel_path}")
    sim_lib = ascend / "tools" / "simulator" / ARCH_PIN / "lib"
    if not sim_lib.is_dir():
        raise GenError(f"simulator lib dir missing: {sim_lib}")
    eval_python = eval_root / "python"
    if not (eval_python / "asc" / "__init__.py").exists():
        raise GenError(f"pyasc-v2-eval asc package not found under {eval_python}")

    probe = BUILD_CACHE / "drivers" / f"probe_{kernel_path.parent.name}_{'x'.join(map(str, shape))}.py"
    probe.parent.mkdir(parents=True, exist_ok=True)
    probe.write_text(_probe_source(kernel_path, shape, dtype, ARCH_PIN))

    setenv = ascend / "set_env.sh"
    env_prefix = (
        f"source {setenv}; "
        f"export PYTHONPATH={eval_python}:$PYTHONPATH; "
        f"export LD_LIBRARY_PATH={sim_lib}:$LD_LIBRARY_PATH; "
        f"export PYASC_DUMP_PATH=/tmp/pyasc-eval-dump; "
    )
    if verbose:
        print(f"[gen] {kernel_path.parent.name} @ {'x'.join(map(str, shape))}: {runs} single-launch processes")

    ticks: list[int] = []
    wall = 0.0
    last_log = BUILD_CACHE / "logs" / f"{probe.stem}.log"
    last_log.parent.mkdir(parents=True, exist_ok=True)
    for i in range(runs):
        log = last_log.with_name(f"{probe.stem}.run{i + 1}.log")
        cmd = f"{env_prefix} {python} {probe} > {log} 2>&1; echo EXIT=$?"
        subprocess.run(["bash", "-c", cmd], cwd=str(REPO_ROOT),
                       capture_output=True, text=True, timeout=1200)
        text = log.read_text(errors="replace") if log.exists() else ""
        m = _TICK_RE.search(text)
        if "PROBE_DONE" not in text or m is None:
            tail = "\n".join(text.splitlines()[-25:])
            raise GenError(f"pyasc probe run {i + 1} did not complete cleanly:\n{tail}")
        ticks.append(int(m.group(1)))
        w = _WALL_RE.search(text)
        if w:
            wall = float(w.group(1))
        last_log = log
        if verbose:
            print(f"[run {i + 1}/{runs}] Total tick = {ticks[-1]}")

    n = 1
    for d in shape:
        n *= d
    median = int(statistics.median(ticks))
    spread = (max(ticks) - min(ticks)) / median if median else 0.0
    return {
        "cell": f"{kernel_path.parent.name}",
        "kernel": str(kernel_path),
        "dtype": dtype,
        "shape": shape,
        "n_elements": n,
        "arch": ARCH_PIN,
        "metric": "camodel_total_tick",
        "method": "single-launch process, Total tick, median of processes (ticks-calculation.md 3)",
        "gen_ticks": median,
        "gen_ticks_runs": ticks,
        "gen_ticks_method": f"median_of_{runs}",
        "gen_ticks_spread": round(spread, 4),
        "wall_ms_last": wall,
        "camodel_log": str(last_log),
    }


def _parse_shape(s: str) -> list[int]:
    return [int(x) for x in re.split(r"[x,]", s) if x]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Phase 11 pyasc generated-side runner")
    ap.add_argument("--cell", help="e.g. abs/float16 (resolves the cached kernel)")
    ap.add_argument("--kernel", help="path to a kernel.py (overrides --cell)")
    ap.add_argument("--dtype", choices=["float16", "float32"], help="required with --kernel")
    ap.add_argument("--shape", default="32x4096")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--eval-root", default=str(DEFAULT_EVAL_ROOT))
    ap.add_argument("--ascend-home", default=str(DEFAULT_ASCEND))
    ap.add_argument("--python", default="python3.11")
    ap.add_argument("--out", help="evidence JSON path")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("-q", "--quiet", action="store_true")
    args = ap.parse_args(argv)

    if args.kernel:
        kernel_path = Path(args.kernel)
        if not args.dtype:
            raise SystemExit("--dtype required when using --kernel")
        dtype = args.dtype
        cell_tag = kernel_path.parent.name
    elif args.cell:
        if args.cell not in CELL_TO_KERNEL:
            print(f"unknown cell '{args.cell}'", file=sys.stderr)
            return 2
        kernel_path, dtype = CELL_TO_KERNEL[args.cell]
        cell_tag = args.cell.replace("/", "-")
    else:
        print("provide --cell OR (--kernel and --dtype)", file=sys.stderr)
        return 2

    shape = _parse_shape(args.shape)
    verbose = not args.quiet
    try:
        result = measure(kernel_path, shape, dtype, eval_root=Path(args.eval_root),
                         ascend=Path(args.ascend_home), runs=args.runs,
                         python=args.python, verbose=verbose)
    except GenError as exc:
        print(f"[BLOCKED] {exc}", file=sys.stderr)
        return 1

    shape_s = "x".join(map(str, shape))
    out = Path(args.out) if args.out else (GEN_EVIDENCE / f"{cell_tag}-ascend950-{shape_s}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    if verbose or args.json:
        print(json.dumps(result, indent=2))
    print(f"[gen] {result['cell']} @ {shape_s}: gen_ticks={result['gen_ticks']} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
