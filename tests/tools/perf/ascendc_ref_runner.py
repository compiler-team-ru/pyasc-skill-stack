#!/usr/bin/env python3
"""Phase 11 / 12 — AscendC reference runner.

Builds the *canonical* hand-written operator (no hand-rolled fallback) from one
of the cloned reference repos (``ops-math`` or ``ops-nn``), runs its aclnn
example on the ascend950 camodel, and reports the camodel ``Total tick:`` as
``ref_ticks`` for a cell (op + dtype + shape).

Pipeline (all cached under ``evidence/perf/_build_cache/``; the cache is
gitignored, the parsed tick JSON under ``evidence/perf/ascendc-ref/`` is not):

1. ``build.sh --pkg --soc=ascend950 --ops=<op>`` in the op's reference repo,
   producing a self-contained ``cann-ops-*-custom_*.run`` package (kernel bin +
   opapi). ops-math and ops-nn share this exact build interface.
2. Install the ``.run`` to a writable, per-repo custom-opp path (no root needed).
3. Apply three env fix-ups required to run a custom op on the camodel without
   root (none touch the operator/kernel source):
     * a writable sim shadow dir providing ``libruntime.so`` /
       ``libascend_hal.so`` -> the camodel libs (CANN sim dir is root-owned);
     * ``vendors/config.ini`` (``load_priority=<vendor>``) + an ``aclnnop``
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
    python ascendc_ref_runner.py --cell rms_norm/float16 --shape 8x256 --json
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
DEFAULT_OPS_NN = Path(
    os.environ.get("OPS_NN_HOME", "/home/aloschilov/workspace/ops-nn")
)
DEFAULT_ASCEND = Path(
    os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/cann-9.0.0")
)
BUILD_CACHE = REPO_ROOT / "evidence" / "perf" / "_build_cache"
REF_EVIDENCE = REPO_ROOT / "evidence" / "perf" / "ascendc-ref"

# camodel chip dir for soc=ascend950 (see build.sh get_simulator_chip_version)
REF_SIM_CHIP = "dav_3510"
ARCH_PIN = "Ascend950PR_9599"

# dtype -> (aclDataType, host C type, byte-pattern initialiser)
DTYPES = {
    "f16": {"acl": "ACL_FLOAT16", "ctype": "uint16_t", "init": "0xBC00", "name": "float16"},
    "f32": {"acl": "ACL_FLOAT", "ctype": "float", "init": "-1.0f", "name": "float32"},
}


class RefError(RuntimeError):
    pass


# --------------------------------------------------------------------------- #
# Reference repo registry.  ops-math and ops-nn share the same build.sh
# interface (build.sh --pkg --soc=ascend950 --ops=<op>) and the same custom-opp
# ``.run`` install flow; they only differ in repo root, the produced ``.run``
# name, and the installed vendor sub-dir name (discovered by globbing so we
# never hardcode the vendor string).  ops-cv / ops-transformer would slot in
# here too, but none of the five demo ops live there.
# --------------------------------------------------------------------------- #
class Repo:
    def __init__(self, name: str, root: Path, install_subdir: str, run_glob: str,
                 needs_third_party_patch: bool):
        self.name = name
        self.root = root
        self.install_root = BUILD_CACHE / install_subdir
        self.run_glob = run_glob
        self.needs_third_party_patch = needs_third_party_patch

    def vendor_dir(self) -> Path:
        """First (and normally only) vendor dir under the install root.
        Discovered by glob so the repo's CUSTOM_OPP vendor name need not be
        hardcoded.  Falls back to a conventional name before first install."""
        vroot = self.install_root / "vendors"
        if vroot.is_dir():
            dirs = [d for d in sorted(vroot.iterdir()) if d.is_dir()]
            if dirs:
                return dirs[0]
        fallback = "custom_math" if self.name == "ops-math" else "customize"
        return vroot / fallback


def _make_repos(ops_math: Path, ops_nn: Path) -> dict[str, Repo]:
    return {
        "ops-math": Repo("ops-math", ops_math, "installed_opp",
                         "cann-ops-math-custom_*.run", needs_third_party_patch=True),
        "ops-nn": Repo("ops-nn", ops_nn, "installed_opp_nn",
                       "*custom_*.run", needs_third_party_patch=False),
    }


# --------------------------------------------------------------------------- #
# Per-op driver registry.  Each entry knows the source repo, the aclnn header
# and how to build the tensors + invoke the op.  Keeps the canonical kernel
# untouched; only the host driver shape/dtype change (the comparability
# contract, see plan R2).
# --------------------------------------------------------------------------- #
def _shape_literal(shape: list[int]) -> str:
    return ", ".join(str(d) for d in shape)


def _prod(shape: list[int]) -> int:
    n = 1
    for d in shape:
        n *= d
    return n


def _abs_body(dt: dict, shape: list[int]) -> str:
    n = _prod(shape)
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
    n = _prod(shape)
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


def _tanh_body(dt: dict, shape: list[int]) -> str:
    # aclnnTanh(self, out): unary elementwise, mirrors ops-math/math/tanh
    # examples/test_aclnn_tanh.cpp.
    n = _prod(shape)
    return f"""
  std::vector<int64_t> shape = {{ {_shape_literal(shape)} }};
  int64_t n = {n};
  void *xAddr=nullptr,*yAddr=nullptr; aclTensor *x=nullptr,*y=nullptr;
  std::vector<{dt['ctype']}> xh(n, {dt['init']}), yh(n, 0);
  if (CreateAclTensor(xh, shape, &xAddr, aclDataType::{dt['acl']}, &x)) return 1;
  if (CreateAclTensor(yh, shape, &yAddr, aclDataType::{dt['acl']}, &y)) return 1;
  uint64_t ws=0; aclOpExecutor* exe;
  ACL_CALL(aclnnTanhGetWorkspaceSize(x, y, &ws, &exe));
  void* wsAddr=nullptr; if (ws>0) ACL_CALL(aclrtMalloc(&wsAddr, ws, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CALL(aclnnTanh(wsAddr, ws, exe, stream));
  ACL_CALL(aclrtSynchronizeStream(stream));
  if (ws>0) aclrtFree(wsAddr); aclrtFree(xAddr); aclrtFree(yAddr);
  aclDestroyTensor(x); aclDestroyTensor(y);
"""


def _drop_out_do_mask_body(dt: dict, shape: list[int]) -> str:
    # aclnnDropoutDoMask(self, maskOut, p, out): self [..], maskOut UINT8
    # bit-packed (1 bit/element, byte-aligned -> ceil(n/8) bytes; we allocate a
    # generous n bytes, matching the >=1B/elem ratio in the ops-math example).
    n = _prod(shape)
    return f"""
  std::vector<int64_t> shape = {{ {_shape_literal(shape)} }};
  std::vector<int64_t> maskShape = {{ {n} }};
  int64_t n = {n};
  void *xAddr=nullptr,*mAddr=nullptr,*oAddr=nullptr;
  aclTensor *x=nullptr,*mask=nullptr,*o=nullptr;
  std::vector<{dt['ctype']}> xh(n, {dt['init']}), oh(n, 0);
  std::vector<uint8_t> mh(n, 0xFF);
  if (CreateAclTensor(xh, shape, &xAddr, aclDataType::{dt['acl']}, &x)) return 1;
  if (CreateAclTensor(mh, maskShape, &mAddr, aclDataType::ACL_UINT8, &mask)) return 1;
  if (CreateAclTensor(oh, shape, &oAddr, aclDataType::{dt['acl']}, &o)) return 1;
  double p = 0.5;
  uint64_t ws=0; aclOpExecutor* exe;
  ACL_CALL(aclnnDropoutDoMaskGetWorkspaceSize(x, mask, p, o, &ws, &exe));
  void* wsAddr=nullptr; if (ws>0) ACL_CALL(aclrtMalloc(&wsAddr, ws, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CALL(aclnnDropoutDoMask(wsAddr, ws, exe, stream));
  ACL_CALL(aclrtSynchronizeStream(stream));
  if (ws>0) aclrtFree(wsAddr); aclrtFree(xAddr); aclrtFree(mAddr); aclrtFree(oAddr);
  aclDestroyTensor(x); aclDestroyTensor(mask); aclDestroyTensor(o);
"""


def _rms_norm_body(dt: dict, shape: list[int]) -> str:
    # aclnnRmsNorm(x, gamma, eps, y, rstd): x [R,C], gamma [C], y [R,C],
    # rstd [R,1] (always fp32).  Mirrors ops-nn/norm/rms_norm example.
    if len(shape) != 2:
        raise RefError("rms_norm driver expects a 2D shape [num_rows, num_cols]")
    R, C = shape
    n = R * C
    return f"""
  std::vector<int64_t> xShape = {{ {R}, {C} }};
  std::vector<int64_t> gammaShape = {{ {C} }};
  std::vector<int64_t> rstdShape = {{ {R}, 1 }};
  void *xAddr=nullptr,*gAddr=nullptr,*yAddr=nullptr,*rAddr=nullptr;
  aclTensor *x=nullptr,*g=nullptr,*y=nullptr,*rstd=nullptr;
  std::vector<{dt['ctype']}> xh({n}, {dt['init']}), gh({C}, {dt['init']}), yh({n}, 0);
  std::vector<float> rh({R}, 0);
  if (CreateAclTensor(xh, xShape, &xAddr, aclDataType::{dt['acl']}, &x)) return 1;
  if (CreateAclTensor(gh, gammaShape, &gAddr, aclDataType::{dt['acl']}, &g)) return 1;
  if (CreateAclTensor(yh, xShape, &yAddr, aclDataType::{dt['acl']}, &y)) return 1;
  if (CreateAclTensor(rh, rstdShape, &rAddr, aclDataType::ACL_FLOAT, &rstd)) return 1;
  double eps = 1e-6;
  uint64_t ws=0; aclOpExecutor* exe;
  ACL_CALL(aclnnRmsNormGetWorkspaceSize(x, g, eps, y, rstd, &ws, &exe));
  void* wsAddr=nullptr; if (ws>0) ACL_CALL(aclrtMalloc(&wsAddr, ws, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CALL(aclnnRmsNorm(wsAddr, ws, exe, stream));
  ACL_CALL(aclrtSynchronizeStream(stream));
  if (ws>0) aclrtFree(wsAddr); aclrtFree(xAddr); aclrtFree(gAddr); aclrtFree(yAddr); aclrtFree(rAddr);
  aclDestroyTensor(x); aclDestroyTensor(g); aclDestroyTensor(y); aclDestroyTensor(rstd);
"""


def _batch_norm_v3_body(dt: dict, shape: list[int]) -> str:
    # aclnnBatchNorm(self, weight, bias, rMean, rVar, training, momentum, eps,
    #                out, mean, var): self [N,C,L]; per-channel params [C].
    # Affine params + running stats are fp32 (standard BN), self/out match the
    # cell dtype.  Mirrors ops-nn/norm/batch_norm_v3 example.
    if len(shape) != 3:
        raise RefError("batch_norm_v3 driver expects a 3D shape [N, C, L]")
    N, C, L = shape
    n = N * C * L
    return f"""
  std::vector<int64_t> selfShape = {{ {N}, {C}, {L} }};
  std::vector<int64_t> chShape = {{ {C} }};
  void *xAddr=nullptr,*wAddr=nullptr,*bAddr=nullptr,*rmAddr=nullptr,*rvAddr=nullptr;
  void *oAddr=nullptr,*mAddr=nullptr,*vAddr=nullptr;
  aclTensor *x=nullptr,*w=nullptr,*b=nullptr,*rm=nullptr,*rv=nullptr,*o=nullptr,*mean=nullptr,*var=nullptr;
  std::vector<{dt['ctype']}> xh({n}, {dt['init']}), oh({n}, 0);
  std::vector<float> wh({C}, 1.0f), bh({C}, 0.0f), rmh({C}, 0.0f), rvh({C}, 1.0f), mh({C}, 0.0f), vh({C}, 1.0f);
  if (CreateAclTensor(xh, selfShape, &xAddr, aclDataType::{dt['acl']}, &x)) return 1;
  if (CreateAclTensor(wh, chShape, &wAddr, aclDataType::ACL_FLOAT, &w)) return 1;
  if (CreateAclTensor(bh, chShape, &bAddr, aclDataType::ACL_FLOAT, &b)) return 1;
  if (CreateAclTensor(rmh, chShape, &rmAddr, aclDataType::ACL_FLOAT, &rm)) return 1;
  if (CreateAclTensor(rvh, chShape, &rvAddr, aclDataType::ACL_FLOAT, &rv)) return 1;
  if (CreateAclTensor(oh, selfShape, &oAddr, aclDataType::{dt['acl']}, &o)) return 1;
  if (CreateAclTensor(mh, chShape, &mAddr, aclDataType::ACL_FLOAT, &mean)) return 1;
  if (CreateAclTensor(vh, chShape, &vAddr, aclDataType::ACL_FLOAT, &var)) return 1;
  bool training = true; double momentum = 0.1; double eps = 1e-5;
  uint64_t ws=0; aclOpExecutor* exe;
  ACL_CALL(aclnnBatchNormGetWorkspaceSize(x, w, b, rm, rv, training, momentum, eps, o, mean, var, &ws, &exe));
  void* wsAddr=nullptr; if (ws>0) ACL_CALL(aclrtMalloc(&wsAddr, ws, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CALL(aclnnBatchNorm(wsAddr, ws, exe, stream));
  ACL_CALL(aclrtSynchronizeStream(stream));
  if (ws>0) aclrtFree(wsAddr);
  aclrtFree(xAddr); aclrtFree(wAddr); aclrtFree(bAddr); aclrtFree(rmAddr); aclrtFree(rvAddr);
  aclrtFree(oAddr); aclrtFree(mAddr); aclrtFree(vAddr);
  aclDestroyTensor(x); aclDestroyTensor(w); aclDestroyTensor(b); aclDestroyTensor(rm);
  aclDestroyTensor(rv); aclDestroyTensor(o); aclDestroyTensor(mean); aclDestroyTensor(var);
"""


def _apply_adam_body(dt: dict, shape: list[int]) -> str:
    # aclnnApplyAdam(var, m, v, beta1Power, beta2Power, lr, beta1, beta2,
    #                epsilon, grad, useLocking, useNesterov): in-place optimizer.
    # var/m/v/grad are [R,C]; the 6 hyper-params are [1] tensors.  fp32 (Adam
    # optimizer states are fp32).  Mirrors ops-nn/optim/apply_adam example.
    if len(shape) != 2:
        raise RefError("apply_adam driver expects a 2D shape [rows, cols]")
    R, C = shape
    n = R * C
    return f"""
  std::vector<int64_t> tShape = {{ {R}, {C} }};
  std::vector<int64_t> sShape = {{ 1 }};
  void *varAddr=nullptr,*mAddr=nullptr,*vAddr=nullptr,*gAddr=nullptr;
  void *b1pAddr=nullptr,*b2pAddr=nullptr,*lrAddr=nullptr,*b1Addr=nullptr,*b2Addr=nullptr,*epsAddr=nullptr;
  aclTensor *var=nullptr,*m=nullptr,*v=nullptr,*grad=nullptr;
  aclTensor *b1p=nullptr,*b2p=nullptr,*lr=nullptr,*b1=nullptr,*b2=nullptr,*eps=nullptr;
  std::vector<{dt['ctype']}> varh({n}, {dt['init']}), mh({n}, 0), vh({n}, 0), gh({n}, {dt['init']});
  std::vector<{dt['ctype']}> b1ph(1, {dt['init']}), b2ph(1, {dt['init']}), lrh(1, {dt['init']});
  std::vector<{dt['ctype']}> b1h(1, {dt['init']}), b2h(1, {dt['init']}), epsh(1, {dt['init']});
  if (CreateAclTensor(varh, tShape, &varAddr, aclDataType::{dt['acl']}, &var)) return 1;
  if (CreateAclTensor(mh, tShape, &mAddr, aclDataType::{dt['acl']}, &m)) return 1;
  if (CreateAclTensor(vh, tShape, &vAddr, aclDataType::{dt['acl']}, &v)) return 1;
  if (CreateAclTensor(b1ph, sShape, &b1pAddr, aclDataType::{dt['acl']}, &b1p)) return 1;
  if (CreateAclTensor(b2ph, sShape, &b2pAddr, aclDataType::{dt['acl']}, &b2p)) return 1;
  if (CreateAclTensor(lrh, sShape, &lrAddr, aclDataType::{dt['acl']}, &lr)) return 1;
  if (CreateAclTensor(b1h, sShape, &b1Addr, aclDataType::{dt['acl']}, &b1)) return 1;
  if (CreateAclTensor(b2h, sShape, &b2Addr, aclDataType::{dt['acl']}, &b2)) return 1;
  if (CreateAclTensor(epsh, sShape, &epsAddr, aclDataType::{dt['acl']}, &eps)) return 1;
  if (CreateAclTensor(gh, tShape, &gAddr, aclDataType::{dt['acl']}, &grad)) return 1;
  bool useLocking = false; bool useNesterov = false;
  uint64_t ws=0; aclOpExecutor* exe;
  ACL_CALL(aclnnApplyAdamGetWorkspaceSize(var, m, v, b1p, b2p, lr, b1, b2, eps, grad, useLocking, useNesterov, &ws, &exe));
  void* wsAddr=nullptr; if (ws>0) ACL_CALL(aclrtMalloc(&wsAddr, ws, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CALL(aclnnApplyAdam(wsAddr, ws, exe, stream));
  ACL_CALL(aclrtSynchronizeStream(stream));
  if (ws>0) aclrtFree(wsAddr);
  aclrtFree(varAddr); aclrtFree(mAddr); aclrtFree(vAddr); aclrtFree(gAddr);
  aclrtFree(b1pAddr); aclrtFree(b2pAddr); aclrtFree(lrAddr); aclrtFree(b1Addr); aclrtFree(b2Addr); aclrtFree(epsAddr);
  aclDestroyTensor(var); aclDestroyTensor(m); aclDestroyTensor(v); aclDestroyTensor(grad);
  aclDestroyTensor(b1p); aclDestroyTensor(b2p); aclDestroyTensor(lr);
  aclDestroyTensor(b1); aclDestroyTensor(b2); aclDestroyTensor(eps);
"""


# op -> {repo, build_op (--ops= name + <op>.json sentinel), header, body}.
# build_op is the snake op-type the kernel config is keyed by; it can differ
# from the registry key when a cell maps onto a sibling reference op (e.g. the
# Adam cell uses the apply_adam reference, since pure ApplyAdamD has no aclnn).
OP_SPECS = {
    "abs": {"repo": "ops-math", "build_op": "abs",
            "header": "aclnnop/aclnn_abs.h", "body": _abs_body},
    "add": {"repo": "ops-math", "build_op": "add",
            "header": "aclnnop/aclnn_add.h", "body": _add_body},
    "reduce_sum": {"repo": "ops-math", "build_op": "reduce_sum",
                   "header": "aclnnop/aclnn_reduce_sum.h", "body": _reduce_sum_body},
    "tanh": {"repo": "ops-math", "build_op": "tanh",
             "header": "aclnnop/aclnn_tanh.h", "body": _tanh_body},
    "drop_out_do_mask": {"repo": "ops-math", "build_op": "drop_out_do_mask",
                         "header": "aclnnop/aclnn_dropout_do_mask.h",
                         "body": _drop_out_do_mask_body},
    "rms_norm": {"repo": "ops-nn", "build_op": "rms_norm",
                 "header": "aclnnop/aclnn_rms_norm.h", "body": _rms_norm_body},
    "batch_norm_v3": {"repo": "ops-nn", "build_op": "batch_norm_v3",
                      "header": "aclnnop/aclnn_batch_norm.h", "body": _batch_norm_v3_body},
    "apply_adam": {"repo": "ops-nn", "build_op": "apply_adam",
                   "header": "aclnnop/aclnn_apply_adam.h", "body": _apply_adam_body},
}

# Demo cell -> (op spec key, dtype).
CELL_TO_OP_DTYPE = {
    "abs/float16": ("abs", "f16"),
    "add/float16": ("add", "f16"),
    "reduce_sum/float32": ("reduce_sum", "f32"),
    "tanh/float16": ("tanh", "f16"),
    "drop_out_do_mask/float16": ("drop_out_do_mask", "f16"),
    "rms_norm/float16": ("rms_norm", "f16"),
    "rms_norm/float32": ("rms_norm", "f32"),
    "batch_norm_v3/float32": ("batch_norm_v3", "f32"),
    "apply_adam/float32": ("apply_adam", "f32"),
}


def _driver_source(op: str, dtype: str, shape: list[int]) -> str:
    spec = OP_SPECS.get(op)
    if spec is None:
        raise RefError(f"no driver registered for op '{op}' (add it to OP_SPECS)")
    dt = DTYPES[dtype]
    body = spec["body"](dt, shape)
    return f"""// AUTO-GENERATED perf driver (Phase 11/12). Canonical {spec['repo']} {op} op;
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


def _ensure_third_party_patches(repo: Repo) -> None:
    """ops-math's abseil/protobuf third-party build expects two patch files
    under cmake/third_party/build/modules/patch/ that are shipped elsewhere in
    the tree.  Create the dir + copy them (third-party dep scaffolding only)."""
    if not repo.needs_third_party_patch:
        return
    dst = repo.root / "cmake" / "third_party" / "build" / "modules" / "patch"
    src = repo.root / "third_party" / "opbase" / "cmake" / "third_party"
    needed = ["protobuf-hide_absl_symbols.patch", "protobuf_25.1_change_version.patch"]
    if all((dst / n).exists() for n in needed):
        return
    dst.mkdir(parents=True, exist_ok=True)
    for n in needed:
        s = src / n
        if s.exists() and not (dst / n).exists():
            shutil.copy2(s, dst / n)


def _op_config_path(repo: Repo, op: str) -> Path | None:
    base = repo.install_root / "vendors"
    if not base.is_dir():
        return None
    for vendor in sorted(base.iterdir()):
        if not vendor.is_dir():
            continue
        cfg = vendor / "op_impl/ai_core/tbe/kernel/config/ascend950" / f"{op}.json"
        if cfg.exists():
            return cfg
    return None


def _op_is_installed(repo: Repo, op: str) -> bool:
    return _op_config_path(repo, op) is not None


def build_and_install(repo: Repo, ascend: Path, op: str, *, verbose: bool) -> Path:
    """Build the canonical op package and install it to the writable cache.
    Returns the vendor dir.  Cached on the per-op <op>.json sentinel."""
    if _op_is_installed(repo, op):
        vendor = repo.vendor_dir()
        if verbose:
            print(f"[cache] {repo.name}:{op}: already built+installed at {vendor}")
        _post_install_fixups(repo, ascend, op)
        return vendor

    shim = _ensure_cmake_shim()
    _ensure_third_party_patches(repo)
    setenv = ascend / "set_env.sh"
    if not setenv.exists():
        raise RefError(f"CANN set_env.sh not found: {setenv}")

    build_cmd = (
        f"source {setenv}; export PATH={shim}:$PATH; "
        f"bash build.sh --pkg --soc=ascend950 --ops={op}"
    )
    if verbose:
        print(f"[build] {repo.name}:{op}: build.sh --pkg --soc=ascend950 --ops={op} (may take ~1-3 min)")
    r = _bash(build_cmd, cwd=repo.root, timeout=2400)
    if r.returncode != 0:
        tail = "\n".join((r.stdout + r.stderr).splitlines()[-25:])
        raise RefError(f"{repo.name} build failed for {op} (exit {r.returncode}):\n{tail}")

    runs = sorted((repo.root / "build_out").glob(repo.run_glob))
    runs = [p for p in runs if p.parent.name == "build_out"]
    if not runs:
        raise RefError(f"no .run package produced by build.sh --pkg (glob {repo.run_glob})")
    run_pkg = runs[-1]

    dest = repo.install_root
    dest.mkdir(parents=True, exist_ok=True)
    install_cmd = f"source {setenv}; bash {run_pkg} --quiet --install-path={dest}"
    if verbose:
        print(f"[install] {repo.name}:{op}: {run_pkg.name} -> {dest}")
    r = _bash(install_cmd, cwd=run_pkg.parent, timeout=600)
    if "SUCCESS" not in (r.stdout + r.stderr):
        tail = "\n".join((r.stdout + r.stderr).splitlines()[-15:])
        raise RefError(f"install failed for {op}:\n{tail}")

    _post_install_fixups(repo, ascend, op)
    if not _op_is_installed(repo, op):
        raise RefError(f"{op}.json not present after install+fixup; kernel config missing")
    return repo.vendor_dir()


def _post_install_fixups(repo: Repo, ascend: Path, op: str) -> None:
    vendor = repo.vendor_dir()
    # vendor registration
    cfg_ini = repo.install_root / "vendors" / "config.ini"
    if not cfg_ini.exists():
        cfg_ini.write_text(f"load_priority={vendor.name}\n")
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
    if cfg_dir.is_dir() and not target.exists():
        candidates = [p for p in cfg_dir.glob(f"{op}_apt.json")] + [
            p for p in cfg_dir.glob("*_apt.json")
        ] + [
            p for p in cfg_dir.glob("*.json")
            if p.name not in ("binary_info_config.json", f"{op}.json")
        ]
        if candidates:
            shutil.copy2(candidates[0], target)


def _opapi_math_shim(repos: dict[str, Repo], ascend: Path, *, verbose: bool) -> Path | None:
    """ops-nn's ``libcust_opapi.so`` resolves its base ``l0op::*`` ops (Cast,
    Reshape, Transpose, ...) through a ``DT_NEEDED libopapi_math.so``; only a
    build-time *stub* of that lib exists.  The real definitions live in the
    ops-math vendor ``libcust_opapi.so`` (472 l0op symbols).  Expose it under
    the name ``libopapi_math.so`` so both the driver link and the runtime
    loader resolve against the real ops-math opapi.  Returns the shim dir."""
    math_repo = repos["ops-math"]
    math_lib = math_repo.vendor_dir() / "op_api" / "lib" / "libcust_opapi.so"
    if not math_lib.exists():
        # Any ops-math op build produces the shared opapi lib; bring up `abs`.
        if verbose:
            print("[opapi-shim] ops-math opapi lib missing; building abs to produce it")
        build_and_install(math_repo, ascend, "abs", verbose=verbose)
        math_lib = math_repo.vendor_dir() / "op_api" / "lib" / "libcust_opapi.so"
    if not math_lib.exists():
        return None
    shim = BUILD_CACHE / "opapi-shim"
    shim.mkdir(parents=True, exist_ok=True)
    link = shim / "libopapi_math.so"
    if not link.exists():
        link.symlink_to(math_lib)
    return shim


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
def compile_driver(repo: Repo, ascend: Path, op: str, dtype: str, shape: list[int], *,
                   verbose: bool, repos: dict[str, Repo]) -> Path:
    vendor = repo.vendor_dir()
    src = BUILD_CACHE / "drivers" / f"perf_{op}_{dtype}_{'x'.join(map(str, shape))}.cpp"
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_text(_driver_source(op, dtype, shape))
    exe = src.with_suffix("")

    sim = ascend / "tools" / "simulator" / REF_SIM_CHIP / "lib"
    aclnn_inc = ascend / "aarch64-linux" / "include"
    setenv = ascend / "set_env.sh"

    # ops-nn opapi pulls base l0op ops from the ops-math opapi via a
    # ``libopapi_math.so`` DT_NEEDED; provide the real lib through a shim.
    extra_L, extra_l, extra_rpath, extra_ld_flags = "", "", "", ""
    if repo.name != "ops-math":
        shim = _opapi_math_shim(repos, ascend, verbose=verbose)
        if shim is not None:
            extra_L = f"-L {shim} "
            extra_l = "-lopapi_math "
            extra_rpath = f":{shim}"
            # The l0op::* base ops are internal to the (ops-nn) cust_opapi and
            # resolve at runtime via DT_NEEDED libopapi_math.so (-> ops-math
            # opapi on the rpath/shim); trust runtime resolution at link.
            extra_ld_flags = "-Wl,--allow-shlib-undefined "

    cmd = (
        f"source {setenv}; "
        f"g++ {src} "
        f"-I {vendor}/op_api/include -I {ascend}/include "
        f"-I {aclnn_inc} -I {aclnn_inc}/aclnnop "
        f"-L {vendor}/op_api/lib {extra_L}-L {ascend}/lib64 "
        f"-lcust_opapi {extra_l}-lascendcl -lnnopbase "
        f"-L {sim} -lruntime_camodel -lnpu_drv_camodel "
        f"{extra_ld_flags}-o {exe} -Wl,-rpath={vendor}/op_api/lib:{sim}{extra_rpath}"
    )
    if verbose:
        print(f"[compile] {src.name}")
    r = _bash(cmd, cwd=repo.root, timeout=300)
    if r.returncode != 0 or not exe.exists():
        raise RefError(f"driver compile failed:\n{r.stdout}\n{r.stderr}")
    return exe


_TICK_RE = re.compile(r"Total tick:\s*(\d+)")
_WALL_RE = re.compile(r"Model RUN TIME:\s*([\d.]+)")


def run_driver(exe: Path, repo: Repo, ascend: Path, *, runs: int, verbose: bool,
               repos: dict[str, Repo]) -> tuple[list[int], float, Path]:
    vendor = repo.vendor_dir()
    shadow = _sim_shadow(ascend)
    sim = ascend / "tools" / "simulator" / REF_SIM_CHIP / "lib"
    setenv = ascend / "set_env.sh"
    extra_ld = ""
    if repo.name != "ops-math":
        shim = _opapi_math_shim(repos, ascend, verbose=False)
        if shim is not None:
            extra_ld = f"{shim}:"
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
            f"export LD_LIBRARY_PATH={shadow}:{sim}:{extra_ld}{vendor}/op_api/lib:{ascend}/lib64:$LD_LIBRARY_PATH; "
            f"{exe} > {log} 2>&1; echo EXIT=$?"
        )
        r = _bash(cmd, cwd=repo.root, timeout=600)
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
def measure(op: str, dtype: str, shape: list[int], *, ascend: Path,
            runs: int, verbose: bool, ops_math: Path | None = None,
            ops_nn: Path | None = None, repos: dict[str, Repo] | None = None) -> dict:
    spec = OP_SPECS.get(op)
    if spec is None:
        raise RefError(f"no driver registered for op '{op}' (add it to OP_SPECS)")
    if repos is None:
        repos = _make_repos(ops_math or DEFAULT_OPS_MATH, ops_nn or DEFAULT_OPS_NN)
    repo = repos[spec["repo"]]
    build_op = spec.get("build_op", op)

    build_and_install(repo, ascend, build_op, verbose=verbose)
    exe = compile_driver(repo, ascend, op, dtype, shape, verbose=verbose, repos=repos)
    ticks, wall, log = run_driver(exe, repo, ascend, runs=runs, verbose=verbose, repos=repos)
    n = _prod(shape)
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
        "reference_repo": repo.name,
        "reference_source": f"{repo.name} canonical operator ({build_op}, {spec['header']}), build.sh --pkg --soc=ascend950",
        "build_cmd": f"bash build.sh --pkg --soc=ascend950 --ops={build_op}",
        "camodel_log": str(log),
    }


def _parse_shape(s: str) -> list[int]:
    return [int(x) for x in re.split(r"[x,]", s) if x]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Phase 11/12 AscendC reference runner")
    ap.add_argument("--cell", help="e.g. rms_norm/float16 (overrides --op/--dtype)")
    ap.add_argument("--op", choices=sorted(OP_SPECS))
    ap.add_argument("--dtype", choices=sorted(DTYPES))
    ap.add_argument("--shape", default="32x4096", help="e.g. 32x4096 or 8x256")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--ops-math-root", default=str(DEFAULT_OPS_MATH))
    ap.add_argument("--ops-nn-root", default=str(DEFAULT_OPS_NN))
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
    ascend = Path(args.ascend_home)
    repos = _make_repos(Path(args.ops_math_root), Path(args.ops_nn_root))
    verbose = not args.quiet

    try:
        result = measure(op, dtype, shape, ascend=ascend, runs=args.runs,
                         verbose=verbose, repos=repos)
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
