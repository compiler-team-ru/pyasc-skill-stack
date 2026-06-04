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
    "tanh/float16": (KERNELS_ROOT / "tanh_f16" / "kernel.py", "float16"),
    "rms_norm/float16": (KERNELS_ROOT / "rms_norm_f16" / "kernel.py", "float16"),
    "rms_norm/float32": (KERNELS_ROOT / "rms_norm_f32" / "kernel.py", "float32"),
    "drop_out_do_mask/float16": (KERNELS_ROOT / "drop_out_do_mask_f16" / "kernel.py", "float16"),
    "batch_norm_v3/float32": (KERNELS_ROOT / "batch_norm_v3_f32" / "kernel.py", "float32"),
    "apply_adam/float32": (KERNELS_ROOT / "apply_adam_f32" / "kernel.py", "float32"),
}


# Per-cell input-arg builders for the gen probe. None (or absent) -> auto
# elementwise (numpy inputs at SHAPE for every required positional param).
# Multi-shape / multi-framework ops describe their launch args explicitly so
# the *_launch wrapper receives correctly-shaped, correctly-framed inputs.
def _rms_norm_inputs(shape: list[int], dtype: str) -> list:
    rows, cols = shape
    return [
        {"kind": "tensor", "shape": [rows, cols], "dtype": dtype, "fw": "torch"},
        {"kind": "tensor", "shape": [cols], "dtype": dtype, "fw": "torch"},
    ]


def arg_specs_for(cell: str, shape: list[int], dtype: str) -> list | None:
    op = cell.split("/")[0]
    builder = _CELL_INPUT_BUILDERS.get(op)
    return builder(shape, dtype) if builder else None


_CELL_INPUT_BUILDERS = {
    "rms_norm": _rms_norm_inputs,
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
import importlib.util, inspect, json, sys
import numpy as np
import asc.runtime.config as config
from asc.lib.runtime import synchronize

# Optional A/B injection of the bisheng SIMD vector-fusion flag. Empty string
# leaves pyasc's default compile flags untouched (CANN ascend950 default is
# --cce-simd-vf-fusion=false); "true"/"false" force --cce-simd-vf-fusion=<value>
# onto every JIT compile command. The only built-in Python hook (CompileOptions
# .bisheng_options) is per-kernel, so we patch the compiler instead and the
# checked-in kernels need no edits. This flag is the single A/B variable; if the
# compiler entry point is missing we fail loudly rather than silently measuring
# an un-flagged build (which would corrupt the comparison).
VF_FUSION = "{vf_fusion}"
if VF_FUSION:
    import asc.runtime.compiler as _cc
    if not hasattr(_cc.Compiler, "_get_compiler_cmd"):
        print("PROBE_ERROR cannot inject --cce-simd-vf-fusion: "
              "asc.runtime.compiler.Compiler._get_compiler_cmd missing")
        sys.exit(6)
    _vf_orig_cmd = _cc.Compiler._get_compiler_cmd
    def _vf_patched_cmd(self, *a, **k):
        cmd = [c for c in _vf_orig_cmd(self, *a, **k)
               if not str(c).startswith("--cce-simd-vf-fusion")]
        return cmd + ["--cce-simd-vf-fusion=%s" % VF_FUSION]
    _cc.Compiler._get_compiler_cmd = _vf_patched_cmd
    print("VF_FUSION_PATCH --cce-simd-vf-fusion=%s" % VF_FUSION)

KERNEL_PATH = {kernel_path!r}
SHAPE = {shape!r}
DTYPE = "{dtype}"
BACKEND = "Model"
PLATFORM = "{platform}"
OP_NAME = "{op_name}"
# Optional per-op argument specs (JSON). None -> auto elementwise (one numpy
# input per REQUIRED positional param, all at SHAPE). Multi-shape / multi-dtype
# ops (rms_norm gamma is 1-D, dropout mask is uint8, etc.) provide explicit
# specs so the launch gets correctly-shaped, correctly-framed inputs.
ARG_SPECS = json.loads({arg_specs_json!r})

try:
    import torch
except Exception:
    torch = None

spec = importlib.util.spec_from_file_location("gen_kernel_mod", KERNEL_PATH)
mod = importlib.util.module_from_spec(spec)
sys.modules["gen_kernel_mod"] = mod
spec.loader.exec_module(mod)

config.set_platform(config.Backend(BACKEND), config.Platform(PLATFORM))

# Pick the public host dispatcher. A kernel module may expose several
# ``*_launch`` callables (e.g. rms_norm has private ``_full_row_launch`` /
# ``_split_d_launch`` helpers plus the public ``rms_norm_launch`` dispatcher);
# ``dir()`` sorts the underscore-prefixed helpers first, so prefer the public,
# op-named wrapper instead of blindly taking the first match.
_cands = [n for n in dir(mod) if callable(getattr(mod, n)) and n.endswith("_launch")]
_public = [n for n in _cands if not n.startswith("_")]
_pool = _public or _cands
_exact = [n for n in _pool if n == OP_NAME + "_launch"]
_chosen = (_exact or _pool)[0] if _pool else None
if _chosen is None:
    print("PROBE_ERROR no *_launch wrapper found in kernel module")
    sys.exit(3)
launch = getattr(mod, _chosen)

rng = np.random.default_rng(seed=2026)

def _npdt(name):
    return {{"float16": np.float16, "float32": np.float32,
            "uint8": np.uint8, "int32": np.int32}}[name]

def _make_tensor(shape, dtype, fw):
    if dtype == "uint8":
        arr = rng.integers(0, 256, size=shape, dtype=np.uint8)
    else:
        arr = (rng.random(shape, dtype=np.float32) * 10 - 5).astype(_npdt(dtype))
    if fw == "torch":
        if torch is None:
            print("PROBE_ERROR torch required but unavailable")
            sys.exit(4)
        return torch.from_numpy(np.ascontiguousarray(arr))
    return arr

if ARG_SPECS is None:
    # Auto: only REQUIRED positional params get a numpy input at SHAPE; params
    # with defaults (e.g. a generated ``out_pad=OUT_PAD`` scalar) keep their
    # default, else we'd pass an ndarray where a scalar is expected and the
    # kernel silently runs a no-op (see Phase 11b reduce_sum probe).
    _sig = inspect.signature(launch)
    _input_params = [
        p for p in _sig.parameters.values()
        if p.default is inspect.Parameter.empty
        and p.kind in (inspect.Parameter.POSITIONAL_ONLY,
                       inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    nparams = len(_input_params)
    args = [_make_tensor(SHAPE, DTYPE, "numpy") for _ in range(nparams)]
else:
    args = []
    for s in ARG_SPECS:
        if s["kind"] == "tensor":
            args.append(_make_tensor(tuple(s["shape"]), s["dtype"], s.get("fw", "numpy")))
        elif s["kind"] == "scalar":
            args.append(s["value"])
        else:
            print(f"PROBE_ERROR unknown arg spec kind {{s.get('kind')}}")
            sys.exit(5)
    nparams = len(args)

launch(*args)
synchronize()
print(f"PROBE_DONE launch={{launch.__name__}} nparams={{nparams}}")
'''


def _probe_source(kernel_path: Path, shape: list[int], dtype: str, platform: str,
                  arg_specs: list | None, op_name: str, vf_fusion: str = "") -> str:
    return PROBE_TEMPLATE.format(
        kernel_path=str(kernel_path),
        shape=tuple(shape),
        dtype=dtype,
        platform=platform,
        arg_specs_json=json.dumps(arg_specs),
        op_name=op_name,
        vf_fusion=vf_fusion,
    )


_TICK_RE = re.compile(r"Total tick:\s*(\d+)")
_WALL_RE = re.compile(r"Model RUN TIME:\s*([\d.]+)")


def measure(kernel_path: Path, shape: list[int], dtype: str, *, eval_root: Path,
            ascend: Path, runs: int, python: str, verbose: bool,
            arg_specs: list | None = None, op_name: str = "",
            vf_fusion: str | None = None) -> dict:
    if not kernel_path.exists():
        raise GenError(f"cached kernel not found: {kernel_path}")
    if vf_fusion not in (None, "", "true", "false"):
        raise GenError(f"vf_fusion must be None/''/'true'/'false', got {vf_fusion!r}")
    vf = vf_fusion or ""
    sim_lib = ascend / "tools" / "simulator" / ARCH_PIN / "lib"
    if not sim_lib.is_dir():
        raise GenError(f"simulator lib dir missing: {sim_lib}")
    eval_python = eval_root / "python"
    if not (eval_python / "asc" / "__init__.py").exists():
        raise GenError(f"pyasc-v2-eval asc package not found under {eval_python}")

    # Per-variant tag so the off/on runs use isolated probe files, logs, JIT
    # caches and dump dirs. The vf flag is injected inside _get_compiler_cmd
    # (below the CompileOptions hash), so without a separate PYASC_CACHE_DIR the
    # "on" run could read the "off" run's cached binary and silently tie.
    tag = f"vf-{vf}" if vf else "default"
    probe = (BUILD_CACHE / "drivers" /
             f"probe_{kernel_path.parent.name}_{'x'.join(map(str, shape))}_{tag}.py")
    probe.parent.mkdir(parents=True, exist_ok=True)
    probe.write_text(_probe_source(kernel_path, shape, dtype, ARCH_PIN, arg_specs, op_name, vf))

    setenv = ascend / "set_env.sh"
    env_prefix = (
        f"source {setenv}; "
        f"export PYTHONPATH={eval_python}:$PYTHONPATH; "
        f"export LD_LIBRARY_PATH={sim_lib}:$LD_LIBRARY_PATH; "
        f"export PYASC_DUMP_PATH=/tmp/pyasc-eval-dump/{tag}; "
        f"export PYASC_CACHE_DIR=/tmp/pyasc-eval-cache/{tag}; "
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
        if vf and "VF_FUSION_PATCH" not in text:
            tail = "\n".join(text.splitlines()[-25:])
            raise GenError(
                f"vf_fusion={vf} requested but the compiler patch was not applied "
                f"(no VF_FUSION_PATCH marker in run {i + 1}); refusing to report an "
                f"un-flagged measurement:\n{tail}")
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
        "vf_fusion": vf or None,
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
    ap.add_argument("--vf-fusion", choices=["true", "false"],
                    help="force --cce-simd-vf-fusion=<value> onto the JIT compile")
    ap.add_argument("--out", help="evidence JSON path")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("-q", "--quiet", action="store_true")
    args = ap.parse_args(argv)

    arg_specs = None
    op_name = ""
    if args.kernel:
        kernel_path = Path(args.kernel)
        if not args.dtype:
            raise SystemExit("--dtype required when using --kernel")
        dtype = args.dtype
        cell_tag = kernel_path.parent.name
        if args.cell:
            op_name = args.cell.split("/")[0]
            arg_specs = arg_specs_for(args.cell, _parse_shape(args.shape), dtype)
    elif args.cell:
        if args.cell not in CELL_TO_KERNEL:
            print(f"unknown cell '{args.cell}'", file=sys.stderr)
            return 2
        kernel_path, dtype = CELL_TO_KERNEL[args.cell]
        cell_tag = args.cell.replace("/", "-")
        op_name = args.cell.split("/")[0]
        arg_specs = arg_specs_for(args.cell, _parse_shape(args.shape), dtype)
    else:
        print("provide --cell OR (--kernel and --dtype)", file=sys.stderr)
        return 2

    shape = _parse_shape(args.shape)
    verbose = not args.quiet
    try:
        result = measure(kernel_path, shape, dtype, eval_root=Path(args.eval_root),
                         ascend=Path(args.ascend_home), runs=args.runs,
                         python=args.python, verbose=verbose, arg_specs=arg_specs,
                         op_name=op_name, vf_fusion=args.vf_fusion)
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
