#!/usr/bin/env python3
"""Fixture tests for the Phase 10 host-import policy gates in score_kernel.

Each test materializes a synthetic kernel source, runs score_kernel.score(),
and asserts the gating outcome (failure_category, accepted, gating_detail).

Run:

    python3 tests/tools/test_score_kernel_bans.py

Exits 0 on success, 1 on first failure.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

# Allow direct import of score_kernel when run from anywhere.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from score_kernel import score  # type: ignore  # noqa: E402

THRESHOLD = 12


CLEAN_KERNEL = """\
import asc
import asc.runtime.config as config
import asc2


@asc2.jit(always_compile=True)
def clean_kernel(x_ptr: asc.GlobalAddress, out_ptr: asc.GlobalAddress,
                 size: int, tile_size: asc.ConstExpr[int],
                 tile_per_block: asc.ConstExpr[int]):
    x_gm = asc2.tensor(x_ptr, [size])
    out_gm = asc2.tensor(out_ptr, [size])
    base = asc2.block_idx() * tile_size * tile_per_block
    for i in asc2.range(tile_per_block, unroll_factor=2, parallel=True):
        off = base + i * tile_size
        x = asc2.load(x_gm, [tile_size], offsets=[off])
        out = asc2.exp(x)
        asc2.store(out, out_gm, offsets=[off])
"""


LEGACY_TIK_KERNEL = """\
import asc
import asc2
import tik  # banned legacy framework


@asc2.jit
def f(x_ptr: asc.GlobalAddress, out_ptr: asc.GlobalAddress, size: int):
    x_gm = asc2.tensor(x_ptr, [size])
    asc2.store(x_gm, asc2.tensor(out_ptr, [size]), offsets=[0])
"""


LEGACY_ASCENDCL_KERNEL = """\
import asc
import asc2
from ascendcl import acl_init


@asc2.jit
def f(x_ptr: asc.GlobalAddress, out_ptr: asc.GlobalAddress, size: int):
    x_gm = asc2.tensor(x_ptr, [size])
    asc2.store(x_gm, asc2.tensor(out_ptr, [size]), offsets=[0])
"""


LEGACY_TBE_KERNEL = """\
import asc
import asc2
from tbe.dsl import build


@asc2.jit
def f(x_ptr: asc.GlobalAddress, out_ptr: asc.GlobalAddress, size: int):
    x_gm = asc2.tensor(x_ptr, [size])
    asc2.store(x_gm, asc2.tensor(out_ptr, [size]), offsets=[0])
"""


TPOSITION_KERNEL = """\
import asc
import asc2

# Uses TPosition from the v1 stack
TPOSITION_UB = TPosition.UB


@asc2.jit
def f(x_ptr: asc.GlobalAddress, out_ptr: asc.GlobalAddress, size: int):
    x_gm = asc2.tensor(x_ptr, [size])
    asc2.store(x_gm, asc2.tensor(out_ptr, [size]), offsets=[0])
"""


MISSING_ASC2_IMPORT_KERNEL = """\
import asc
# NOTE: missing 'import asc2' but uses asc2.* below


@asc2.jit
def f(x_ptr: asc.GlobalAddress, out_ptr: asc.GlobalAddress, size: int):
    x_gm = asc2.tensor(x_ptr, [size])
    asc2.store(x_gm, asc2.tensor(out_ptr, [size]), offsets=[0])
"""


ALIASED_NPU_KERNEL = """\
import asc
import asc2
import torch


@asc2.jit
def f(x_ptr: asc.GlobalAddress, out_ptr: asc.GlobalAddress, size: int):
    x_gm = asc2.tensor(x_ptr, [size])
    asc2.store(x_gm, asc2.tensor(out_ptr, [size]), offsets=[0])


def launch(x):
    x = torch.zeros(8).npu()  # banned: .npu() dispatch marker
    return x
"""


def _run(source: str, **kwargs):
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(source)
        path = f.name
    try:
        return score(path, **kwargs)
    finally:
        Path(path).unlink(missing_ok=True)


def test_clean_kernel_no_gate():
    r = _run(CLEAN_KERNEL)
    total = sum(1 for c in r.checks if c.passed)
    assert r.failure_category is None, f"unexpected gate: {r.failure_category}"
    assert total >= THRESHOLD, f"clean kernel scored too low: {total}"
    assert r.gating_detail["legacy_api_findings"] == []
    assert r.gating_detail["missing_asc2_import"] is False
    return "clean_kernel_no_gate OK"


def test_legacy_tik_gates():
    r = _run(LEGACY_TIK_KERNEL)
    assert r.failure_category == "F_legacy_api_import", \
        f"expected F_legacy_api_import; got {r.failure_category}"
    assert "import tik" in r.gating_detail["legacy_api_findings"]
    return "legacy_tik_gates OK"


def test_legacy_ascendcl_gates():
    r = _run(LEGACY_ASCENDCL_KERNEL)
    assert r.failure_category == "F_legacy_api_import"
    assert "import ascendcl" in r.gating_detail["legacy_api_findings"]
    return "legacy_ascendcl_gates OK"


def test_legacy_tbe_gates():
    r = _run(LEGACY_TBE_KERNEL)
    assert r.failure_category == "F_legacy_api_import"
    assert "import tbe" in r.gating_detail["legacy_api_findings"]
    return "legacy_tbe_gates OK"


def test_tposition_gates():
    r = _run(TPOSITION_KERNEL)
    assert r.failure_category == "F_legacy_api_import"
    findings = r.gating_detail["legacy_api_findings"]
    assert any("TPosition" in f for f in findings), f"got: {findings}"
    return "tposition_gates OK"


def test_npu_dispatch_gates():
    r = _run(ALIASED_NPU_KERNEL)
    assert r.failure_category == "F_legacy_api_import"
    findings = r.gating_detail["legacy_api_findings"]
    assert any(".npu" in f for f in findings), f"got: {findings}"
    return "npu_dispatch_gates OK"


def test_missing_asc2_import_gates():
    r = _run(MISSING_ASC2_IMPORT_KERNEL)
    assert r.failure_category == "F_missing_asc2_import", \
        f"expected F_missing_asc2_import; got {r.failure_category}"
    assert r.gating_detail["missing_asc2_import"] is True
    return "missing_asc2_import_gates OK"


def test_allow_legacy_apis_opt_out():
    r = _run(LEGACY_TIK_KERNEL, allow_legacy_apis=True)
    assert r.failure_category is None, \
        f"opt-out should suppress F_legacy_api_import, got {r.failure_category}"
    assert r.gating_detail["allow_legacy_apis"] is True
    return "allow_legacy_apis_opt_out OK"


TESTS = [
    test_clean_kernel_no_gate,
    test_legacy_tik_gates,
    test_legacy_ascendcl_gates,
    test_legacy_tbe_gates,
    test_tposition_gates,
    test_npu_dispatch_gates,
    test_missing_asc2_import_gates,
    test_allow_legacy_apis_opt_out,
]


def main() -> int:
    failures = []
    for t in TESTS:
        try:
            msg = t()
        except AssertionError as exc:
            failures.append((t.__name__, str(exc)))
            print(f"  [FAIL] {t.__name__}: {exc}")
        else:
            print(f"  [PASS] {msg}")
    if failures:
        print(f"\n{len(failures)} failure(s)")
        return 1
    print(f"\nall {len(TESTS)} tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
