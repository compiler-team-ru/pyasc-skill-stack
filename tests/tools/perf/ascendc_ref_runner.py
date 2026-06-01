#!/usr/bin/env python3
"""Phase 11 — AscendC reference runner.

Builds the *canonical* ops-math operator (no hand-rolled fallback), runs its
aclnn example on the ascend950 camodel, and reports the camodel ``Total tick:``
as ``ref_ticks`` for a cell (op + dtype + shape).

Pipeline (all cached under ``evidence/perf/_build_cache/``; the cache is
gitignored, the parsed tick JSON under ``evidence/perf/ascendc-ref/`` is not):

1. ``build.sh --pkg --soc=ascend950 --ops=<op>`` in the ops-math tree, producing
   a self-contained ``cann-ops-math-custom_*.run`` package (kernel bin + opapi).
2. Install the ``.run`` to a writable custom-opp path (no root needed).
3. Apply three env fix-ups required to run a custom op on the camodel without
   root (none touch the operator/kernel source):
     * a writable sim shadow dir providing ``libruntime.so`` /
       ``libascend_hal.so`` -> the camodel libs (CANN sim dir is root-owned);
     * ``vendors/config.ini`` (``load_priority=custom_math``) + an ``aclnnop``
       include shim;
     * copy ``<op>_apt.json`` -> ``<op>.json`` in the kernel config dir (the
       runtime looks up the per-op dynamic config by op-type snake name).
4. Generate + compile a perf driver (the canonical example with shape/dtype
   pinned to the comparability contract), run it 3x on the camodel, take the
   median ``Total tick:``.

Both the reference (``dav_3510``) and the pyasc side (``Ascend950PR_9599``) load
a byte-identical ``libmodel_top.so`` camodel, so the ticks are directly
comparable.

Usage::

    python ascendc_ref_runner.py --op abs --dtype f16 --shape 32x4096
    python ascendc_ref_runner.py --cell abs/float16 --json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OPS_MATH = Path(
    os.environ.get("OPS_MATH_HOME", "/home/aloschilov/workspace/ops-math")
)
DEFAULT_ASCEND = Path(
    os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/cann-9.0.0")
)
BUILD_CACHE = REPO_ROOT / "evidence" / "perf" / "_build_cache"
REF_EVIDENCE = REPO_ROOT / "evidence" / "perf" / "ascendc-ref"

# camodel chip dir for soc=ascend950 (see ops-math build.sh get_simulator_chip_version)
REF_SIM_CHIP = "dav_3510"
ARCH_PIN = "Ascend950PR_9599"

# dtype -> (aclDataType, host C type, byte-pattern initialiser for "abs(x)=known")
DTYPES = {
    "f16": {"acl": "ACL_FLOAT16", "ctype": "uint16_t", "init": "0xBC00", "name": "float16"},
    "f32": {"acl": "ACL_FLOAT", "ctype": "float", "init": "-1.0f", "name": "float32"},
}

CELL_TO_OP_DTYPE = {
    "abs/float16": ("abs", "f16"),
    "add/float16": ("add", "f16"),
    "reduce_sum/float32": ("reduce_sum", "f32"),
}


class RefError(RuntimeError):
    pass


# --------------------------------------------------------------------------- #
# Per-op driver registry.  Each entry knows the aclnn header and how to build
# the tensors + invoke the op.  Keeps the canonical kernel untouched; only the
# host driver shape/dtype change (the comparability contract, see plan R2).
# --------------------------------------------------------------------------- #
def _shape_literal(shape: list[int]) -> str:
    return ", ".join(str(d) for d in shape)


def _abs_body(dt: dict, shape: list[int]) -> str:
    n = 1
    for d in shape:
        n *= d
    return f"""
  std::vector<int64_t> shape = {{ {_shape_literal(shape)} }};
  int64_t n = {n};
  void *xAddr=nullptr,*yAddr=nullptr; aclTensor *x=nullptr,*y=nullptr;
  std::vector<{dt['ctype']}> xh(n, {dt['init']}), yh(n, 0);
  if (CreateAclTensor(xh, shape, &xAddr, aclDataType::{dt['acl']}, &x)) return 1;
  if (CreateAclTensor(yh, shape, &yAddr, aclDataType::{dt['acl']}, &y)) return 1;
  uint64_t ws=0; aclOpExecutor* exe;
  ACL_CALL(aclnnAbsGetWorkspaceSize(x, y, &ws, &exe));
  void* wsAddr=nullptr; if (ws>0) ACL_CALL(aclrtMalloc(&wsAddr, ws, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CALL(aclnnAbs(wsAddr, ws, exe, stream));
  ACL_CALL(aclrtSynchronizeStream(stream));
  if (ws>0) aclrtFree(wsAddr); aclrtFree(xAddr); aclrtFree(yAddr);
  aclDestroyTensor(x); aclDestroyTensor(y);
"""


def _add_body(dt: dict, shape: list[int]) -> str:
    n = 1
    for d in shape:
        n *= d
    # aclnnAdd(self, other, alpha, out); alpha=1.0 -> out = self + other.
    return f"""
  std::vector<int64_t> shape = {{ {_shape_literal(shape)} }};
  int64_t n = {n};
  void *xAddr=nullptr,*yAddr=nullptr,*oAddr=nullptr;
  aclTensor *x=nullptr,*y=nullptr,*o=nullptr;
  std::vector<{dt['ctype']}> xh(n, {dt['init']}), yh(n, {dt['init']}), oh(n, 0);
  if (CreateAclTensor(xh, shape, &xAddr, aclDataType::{dt['acl']}, &x)) return 1;
  if (CreateAclTensor(yh, shape, &yAddr, aclDataType::{dt['acl']}, &y)) return 1;
  if (CreateAclTensor(oh, shape, &oAddr, aclDataType::{dt['acl']}, &o)) return 1;
  float alphaVal = 1.0f;
  aclScalar* alpha = aclCreateScalar(&alphaVal, aclDataType::ACL_FLOAT);
  uint64_t ws=0; aclOpExecutor* exe;
  ACL_CALL(aclnnAddGetWorkspaceSize(x, y, alpha, o, &ws, &exe));
  void* wsAddr=nullptr; if (ws>0) ACL_CALL(aclrtMalloc(&wsAddr, ws, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CALL(aclnnAdd(wsAddr, ws, exe, stream));
  ACL_CALL(aclrtSynchronizeStream(stream));
  if (ws>0) aclrtFree(wsAddr); aclrtFree(xAddr); aclrtFree(yAddr); aclrtFree(oAddr);
  aclDestroyScalar(alpha); aclDestroyTensor(x); aclDestroyTensor(y); aclDestroyTensor(o);
"""


def _reduce_sum_body(dt: dict, shape: list[int]) -> str:
    # Reduce the last axis: [num_rows, num_cols] -> [num_rows], matching the
    # pyasc reduce_sum kernel (reduce_axis=-1, keepdims=false). dtype f32.
    if len(shape) != 2:
        raise RefError("reduce_sum driver expects a 2D shape [num_rows, num_cols]")
    n = shape[0] * shape[1]
    return f"""
  std::vector<int64_t> shape = {{ {_shape_literal(shape)} }};
  std::vector<int64_t> outShape = {{ {shape[0]} }};
  int64_t n = {n};
  void *xAddr=nullptr,*oAddr=nullptr; aclTensor *x=nullptr,*o=nullptr;
  std::vector<{dt['ctype']}> xh(n, {dt['init']}), oh({shape[0]}, 0);
  if (CreateAclTensor(xh, shape, &xAddr, aclDataType::{dt['acl']}, &x)) return 1;
  if (CreateAclTensor(oh, outShape, &oAddr, aclDataType::{dt['acl']}, &o)) return 1;
  std::vector<int64_t> dimsData = {{ 1 }};
  aclIntArray* dims = aclCreateIntArray(dimsData.data(), dimsData.size());
  bool keepDims = false;
  uint64_t ws=0; aclOpExecutor* exe;
  ACL_CALL(aclnnReduceSumGetWorkspaceSize(x, dims, keepDims, aclDataType::{dt['acl']}, o, &ws, &exe));
  void* wsAddr=nullptr; if (ws>0) ACL_CALL(aclrtMalloc(&wsAddr, ws, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CALL(aclnnReduceSum(wsAddr, ws, exe, stream));
  ACL_CALL(aclrtSynchronizeStream(stream));
  if (ws>0) aclrtFree(wsAddr); aclrtFree(xAddr); aclrtFree(oAddr);
  aclDestroyIntArray(dims); aclDestroyTensor(x); aclDestroyTensor(o);
"""


DRIVERS = {
    "abs": {"header": "aclnnop/aclnn_abs.h", "body": _abs_body},
    "add": {"header": "aclnnop/aclnn_add.h", "body": _add_body},
    "reduce_sum": {"header": "aclnnop/aclnn_reduce_sum.h", "body": _reduce_sum_body},
}


def _driver_source(op: str, dtype: str, shape: list[int]) -> str:
    spec = DRIVERS.get(op)
    if spec is None:
        raise RefError(f"no driver registered for op '{op}' (add it to DRIVERS)")
    dt = DTYPES[dtype]
    body = spec["body"](dt, shape)
    return f"""// AUTO-GENERATED perf driver (Phase 11). Canonical ops-math {op} op;
// only shape/dtype are pinned for the comparability contract.
#include <iostream>
#include <vector>
#include <cstdint>
#include "acl/acl.h"
#include "{spec['header']}"
#define ACL_CALL(e) do {{ auto _r=(e); if(_r!=ACL_SUCCESS){{ printf("FAIL %s=%d %s\\n", #e, _r, aclGetRecentErrMsg()); return 1; }} }} while(0)
int64_t ShapeSize(const std::vector<int64_t>& s){{int64_t r=1;for(auto i:s)r*=i;return r;}}
template<typename T> int CreateAclTensor(const std::vector<T>& h, const std::vector<int64_t>& s,
    void** d, aclDataType dt, aclTensor** t){{
  auto sz=ShapeSize(s)*sizeof(T);
  if(aclrtMalloc(d,sz,ACL_MEM_MALLOC_HUGE_FIRST)!=ACL_SUCCESS) return 1;
  if(aclrtMemcpy(*d,sz,h.data(),sz,ACL_MEMCPY_HOST_TO_DEVICE)!=ACL_SUCCESS) return 1;
  std::vector<int64_t> st(s.size(),1);
  for(int64_t i=s.size()-2;i>=0;i--) st[i]=s[i+1]*st[i+1];
  *t=aclCreateTensor(s.data(),s.size(),dt,st.data(),0,aclFormat::ACL_FORMAT_ND,s.data(),s.size(),*d);
  return 0;
}}
int main(){{
  int32_t dev=0; aclrtStream stream;
  ACL_CALL(aclInit(nullptr)); ACL_CALL(aclrtSetDevice(dev)); ACL_CALL(aclrtCreateStream(&stream));
  {body}
  printf("REF_DRIVER_DONE op={op} dtype={dtype}\\n");
  aclrtDestroyStream(stream); aclrtResetDevice(dev); aclFinalize();
  return 0;
}}
"""


# --------------------------------------------------------------------------- #
# Build / install / fix-up
# --------------------------------------------------------------------------- #
def _bash(cmd: str, *, cwd: Path | None = None, timeout: int = 1200) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", "-c", cmd],
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _ensure_cmake_shim() -> Path:
    """Expose /usr/bin/cmake under a dir we can put first on PATH, so the real
    cmake wins over a broken ~/.local/bin/cmake shim WITHOUT shadowing CANN's
    ld.lld (which must stay ahead of the system ld.lld-14)."""
    shim = BUILD_CACHE / "cmake-shim"
    shim.mkdir(parents=True, exist_ok=True)
    for tool in ("cmake", "ctest", "cpack"):
        real = shutil.which(tool, path="/usr/bin")
        link = shim / tool
        if real and not link.exists():
            link.symlink_to(real)
    return shim


def _ensure_third_party_patches(ops_math: Path) -> None:
    """ops-math's abseil/protobuf third-party build expects two patch files
    under cmake/third_party/build/modules/patch/ that are shipped elsewhere in
    the tree.  Create the dir + copy them (third-party dep scaffolding only)."""
    dst = ops_math / "cmake" / "third_party" / "build" / "modules" / "patch"
    src = ops_math / "third_party" / "opbase" / "cmake" / "third_party"
    needed = ["protobuf-hide_absl_symbols.patch", "protobuf_25.1_change_version.patch"]
    if all((dst / n).exists() for n in needed):
        return
    dst.mkdir(parents=True, exist_ok=True)
    for n in needed:
        s = src / n
        if s.exists() and not (dst / n).exists():
            shutil.copy2(s, dst / n)


def _installed_opp(ops_math: Path) -> Path:
    return BUILD_CACHE / "installed_opp"


def _vendor_dir(ops_math: Path) -> Path:
    return _installed_opp(ops_math) / "vendors" / "custom_math"


def _op_is_installed(ops_math: Path, op: str) -> bool:
    cfg = (
        _vendor_dir(ops_math)
        / "op_impl/ai_core/tbe/kernel/config/ascend950"
        / f"{op}.json"
    )
    return cfg.exists()


def build_and_install(ops_math: Path, ascend: Path, op: str, *, verbose: bool) -> Path:
    """Build the canonical op package and install it to the writable cache.
    Returns the vendor dir.  Cached on the per-op <op>.json sentinel."""
    vendor = _vendor_dir(ops_math)
    if _op_is_installed(ops_math, op):
        if verbose:
            print(f"[cache] {op}: already built+installed at {vendor}")
        _post_install_fixups(ops_math, ascend, op)
        return vendor

    shim = _ensure_cmake_shim()
    _ensure_third_party_patches(ops_math)
    setenv = ascend / "set_env.sh"
    if not setenv.exists():
        raise RefError(f"CANN set_env.sh not found: {setenv}")

    build_cmd = (
        f"source {setenv}; export PATH={shim}:$PATH; "
        f"bash build.sh --pkg --soc=ascend950 --ops={op}"
    )
    if verbose:
        print(f"[build] {op}: build.sh --pkg --soc=ascend950 --ops={op} (may take ~1-2 min)")
    r = _bash(build_cmd, cwd=ops_math, timeout=1800)
    if r.returncode != 0:
        tail = "\n".join((r.stdout + r.stderr).splitlines()[-25:])
        raise RefError(f"ops-math build failed for {op} (exit {r.returncode}):\n{tail}")

    runs = sorted((ops_math / "build_out").glob("cann-ops-math-custom_*.run"))
    runs = [p for p in runs if p.parent.name == "build_out"]
    if not runs:
        raise RefError("no .run package produced by build.sh --pkg")
    run_pkg = runs[0]

    dest = _installed_opp(ops_math)
    install_cmd = f"source {setenv}; bash {run_pkg} --quiet --install-path={dest}"
    if verbose:
        print(f"[install] {op}: {run_pkg.name} -> {dest}")
    r = _bash(install_cmd, cwd=run_pkg.parent, timeout=600)
    if "SUCCESS" not in (r.stdout + r.stderr):
        tail = "\n".join((r.stdout + r.stderr).splitlines()[-15:])
        raise RefError(f"install failed for {op}:\n{tail}")

    _post_install_fixups(ops_math, ascend, op)
    if not _op_is_installed(ops_math, op):
        raise RefError(f"{op}.json not present after install+fixup; kernel config missing")
    return vendor


def _post_install_fixups(ops_math: Path, ascend: Path, op: str) -> None:
    vendor = _vendor_dir(ops_math)
    # vendor registration
    cfg_ini = _installed_opp(ops_math) / "vendors" / "config.ini"
    if not cfg_ini.exists():
        cfg_ini.write_text("load_priority=custom_math\n")
    # aclnnop include shim so the example's #include "aclnnop/aclnn_<op>.h" resolves
    inc = vendor / "op_api" / "include"
    aclnnop = inc / "aclnnop"
    if inc.is_dir() and not aclnnop.exists():
        try:
            aclnnop.symlink_to(".")
        except OSError:
            pass
    # per-op dynamic config filename: runtime wants <op>.json, build emits <op>_apt.json
    cfg_dir = vendor / "op_impl/ai_core/tbe/kernel/config/ascend950"
    target = cfg_dir / f"{op}.json"
    if not target.exists():
        candidates = [p for p in cfg_dir.glob("*_apt.json")] + [
            p for p in cfg_dir.glob("*.json")
            if p.name not in ("binary_info_config.json", f"{op}.json")
        ]
        if candidates:
            shutil.copy2(candidates[0], target)


def _sim_shadow(ascend: Path) -> Path:
    """Writable dir with libruntime.so / libascend_hal.so -> camodel libs."""
    sim = ascend / "tools" / "simulator" / REF_SIM_CHIP / "lib"
    shadow = BUILD_CACHE / "sim-shadow"
    shadow.mkdir(parents=True, exist_ok=True)
    pairs = [
        ("libruntime.so", "libruntime_camodel.so"),
        ("libascend_hal.so", "libnpu_drv_camodel.so"),
    ]
    for link_name, real_name in pairs:
        link = shadow / link_name
        real = sim / real_name
        if real.exists() and not link.exists():
            link.symlink_to(real)
    return shadow


# --------------------------------------------------------------------------- #
# Compile + run the perf driver
# --------------------------------------------------------------------------- #
def compile_driver(ops_math: Path, ascend: Path, op: str, dtype: str, shape: list[int], *, verbose: bool) -> Path:
    vendor = _vendor_dir(ops_math)
    src = BUILD_CACHE / "drivers" / f"perf_{op}_{dtype}_{'x'.join(map(str, shape))}.cpp"
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_text(_driver_source(op, dtype, shape))
    exe = src.with_suffix("")

    sim = ascend / "tools" / "simulator" / REF_SIM_CHIP / "lib"
    aclnn_inc = ascend / "aarch64-linux" / "include"
    setenv = ascend / "set_env.sh"
    cmd = (
        f"source {setenv}; "
        f"g++ {src} "
        f"-I {vendor}/op_api/include -I {ascend}/include "
        f"-I {aclnn_inc} -I {aclnn_inc}/aclnnop "
        f"-L {vendor}/op_api/lib -L {ascend}/lib64 "
        f"-lcust_opapi -lascendcl -lnnopbase "
        f"-L {sim} -lruntime_camodel -lnpu_drv_camodel "
        f"-o {exe} -Wl,-rpath={vendor}/op_api/lib:{sim}"
    )
    if verbose:
        print(f"[compile] {src.name}")
    r = _bash(cmd, cwd=ops_math, timeout=300)
    if r.returncode != 0 or not exe.exists():
        raise RefError(f"driver compile failed:\n{r.stdout}\n{r.stderr}")
    return exe


_TICK_RE = re.compile(r"Total tick:\s*(\d+)")
_WALL_RE = re.compile(r"Model RUN TIME:\s*([\d.]+)")


def run_driver(exe: Path, ops_math: Path, ascend: Path, *, runs: int, verbose: bool) -> tuple[list[int], float, Path]:
    vendor = _vendor_dir(ops_math)
    shadow = _sim_shadow(ascend)
    sim = ascend / "tools" / "simulator" / REF_SIM_CHIP / "lib"
    setenv = ascend / "set_env.sh"
    log_dir = BUILD_CACHE / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    ticks: list[int] = []
    wall = 0.0
    last_log = log_dir / f"{exe.name}.log"
    for i in range(runs):
        log = log_dir / f"{exe.name}.run{i + 1}.log"
        cmd = (
            f"source {setenv}; "
            f"export ASCEND_CUSTOM_OPP_PATH={vendor}; "
            f"export ASCEND_SLOG_PRINT_TO_STDOUT=1 ASCEND_GLOBAL_LOG_LEVEL=3; "
            f"export LD_LIBRARY_PATH={shadow}:{sim}:{vendor}/op_api/lib:{ascend}/lib64:$LD_LIBRARY_PATH; "
            f"{exe} > {log} 2>&1; echo EXIT=$?"
        )
        r = _bash(cmd, cwd=ops_math, timeout=600)
        text = log.read_text(errors="replace") if log.exists() else ""
        m = _TICK_RE.search(text)
        if "REF_DRIVER_DONE" not in text or m is None:
            tail = "\n".join(text.splitlines()[-20:])
            raise RefError(f"reference run {i + 1} did not complete cleanly:\n{tail}")
        ticks.append(int(m.group(1)))
        w = _WALL_RE.search(text)
        if w:
            wall = float(w.group(1))
        last_log = log
        if verbose:
            print(f"[run {i + 1}/{runs}] Total tick = {ticks[-1]}")
    return ticks, wall, last_log


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def measure(op: str, dtype: str, shape: list[int], *, ops_math: Path, ascend: Path,
            runs: int, verbose: bool) -> dict:
    build_and_install(ops_math, ascend, op, verbose=verbose)
    exe = compile_driver(ops_math, ascend, op, dtype, shape, verbose=verbose)
    ticks, wall, log = run_driver(exe, ops_math, ascend, runs=runs, verbose=verbose)
    n = 1
    for d in shape:
        n *= d
    median = int(statistics.median(ticks))
    spread = (max(ticks) - min(ticks)) / median if median else 0.0
    return {
        "cell": f"{op}/{DTYPES[dtype]['name']}",
        "op": op,
        "dtype": DTYPES[dtype]["name"],
        "shape": shape,
        "n_elements": n,
        "arch": ARCH_PIN,
        "simulator_chip": REF_SIM_CHIP,
        "metric": "camodel_total_tick",
        "ref_ticks": median,
        "ref_ticks_runs": ticks,
        "ref_ticks_method": f"median_of_{runs}",
        "ref_ticks_spread": round(spread, 4),
        "wall_ms_last": wall,
        "reference_kind": "canonical_only",
        "reference_source": f"ops-math canonical operator (math/{op}), build.sh --pkg --soc=ascend950",
        "build_cmd": f"bash build.sh --pkg --soc=ascend950 --ops={op}",
        "camodel_log": str(log),
    }


def _parse_shape(s: str) -> list[int]:
    return [int(x) for x in re.split(r"[x,]", s) if x]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Phase 11 AscendC reference runner")
    ap.add_argument("--cell", help="e.g. abs/float16 (overrides --op/--dtype)")
    ap.add_argument("--op", choices=sorted(DRIVERS))
    ap.add_argument("--dtype", choices=sorted(DTYPES))
    ap.add_argument("--shape", default="32x4096", help="e.g. 32x4096")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--ops-math-root", default=str(DEFAULT_OPS_MATH))
    ap.add_argument("--ascend-home", default=str(DEFAULT_ASCEND))
    ap.add_argument("--out", help="evidence JSON path (default: evidence/perf/ascendc-ref/...)")
    ap.add_argument("--json", action="store_true", help="print result JSON to stdout")
    ap.add_argument("-q", "--quiet", action="store_true")
    args = ap.parse_args(argv)

    if args.cell:
        if args.cell not in CELL_TO_OP_DTYPE:
            print(f"unknown cell '{args.cell}'", file=sys.stderr)
            return 2
        op, dtype = CELL_TO_OP_DTYPE[args.cell]
    elif args.op and args.dtype:
        op, dtype = args.op, args.dtype
    else:
        print("provide --cell OR (--op and --dtype)", file=sys.stderr)
        return 2

    shape = _parse_shape(args.shape)
    ops_math = Path(args.ops_math_root)
    ascend = Path(args.ascend_home)
    verbose = not args.quiet

    try:
        result = measure(op, dtype, shape, ops_math=ops_math, ascend=ascend,
                         runs=args.runs, verbose=verbose)
    except RefError as exc:
        print(f"[BLOCKED] {exc}", file=sys.stderr)
        return 1

    shape_s = "x".join(map(str, shape))
    out = Path(args.out) if args.out else (
        REF_EVIDENCE / f"{op}-{dtype}-ascend950-{shape_s}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    if verbose or args.json:
        print(json.dumps(result, indent=2))
    print(f"[ref] {result['cell']} @ {shape_s}: ref_ticks={result['ref_ticks']} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
