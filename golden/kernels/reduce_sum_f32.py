#!/usr/bin/env python3.10
"""Golden kernel: reduce_sum/float32

Row-wise sum reduction for float32 tensors:
``[num_rows, num_cols] -> [num_rows]``.
Verified on CANN simulator with Ascend950PR_9599 platform.

Cell metadata (mirrors capabilities.yaml; do not drift):
  - shape_regime: fixed
  - reduce_axis: -1
  - output_shape: [num_rows]      # logically [32] for the verified shape
  - accumulator_dtype: float32
  - identity: "0"
  - tail_behavior: host_pad       # OUT_PAD=8 for 32-byte alignment
  - padding: 8
  - partitioning: row_per_core
  - unsupported_regimes:
      [non_last_axis, multi_axis_reduction, dynamic_num_rows]

Non-obvious constraints (Phase 1.5):
  - Alignment: ``OUT_PAD = 8`` is the minimum last-axis element
    count for 32-byte alignment on float32 (8 * 4 bytes = 32);
    the kernel writes ``[1, OUT_PAD]`` per row and the host slices
    column 0 back into a [num_rows] vector.
  - UB/L1/L0 placement: one row tile is loaded into UB,
    ``asc2.reduce_sum`` reduces it to a scalar, and the host-pad
    is constructed in UB via ``asc2.full([1, OUT_PAD], s,
    dtype=row.dtype)`` before storing back to GM. No L0 / cube
    involvement.
  - Padding: 8 elements (float32) on the output last axis;
    ``tail_behavior=host_pad`` records this.
  - Tail behavior: host_pad — the kernel itself does not handle
    partial input tiles (``num_cols`` is a ConstExpr), but the
    output is host-padded for 32-byte alignment.
  - Accumulator dtype: ``float32``. ``asc2.reduce_sum`` accumulates
    in the input dtype which is already float32 here.
  - Reduction axis: -1 (last axis). Reducing along a non-last axis
    or along multiple axes is out of scope
    (``non_last_axis``, ``multi_axis_reduction`` in
    ``unsupported_regimes``).
  - ``num_rows`` is a runtime ``int`` but the kernel does not
    branch on it (each block iterates rows via ``asc2.range``);
    so the cell metadata records ``shape_regime: fixed`` because
    the only verified shape is ``[32, 4096]``. Truly dynamic
    ``num_rows`` is an unsupported regime.
  - Simulator/platform assumptions: ``Ascend950PR_9599`` (C310);
    numpy is safe here (no cube involvement, no torch dependency).
"""

import logging
import argparse
import numpy as np

import asc
import asc.runtime.config as config
import asc2

CORE_NUM = 16
OUT_PAD = 8  # min last-dim for 32-byte alignment with float32

logging.basicConfig(level=logging.INFO)


@asc2.jit(always_compile=True)
def reduce_sum_kernel(x_ptr: asc.GlobalAddress, out_ptr: asc.GlobalAddress,
                      num_rows: int, num_cols: asc.ConstExpr[int],
                      out_pad: asc.ConstExpr[int]):
    x_gm = asc2.tensor(x_ptr, [num_rows, num_cols])
    out_gm = asc2.tensor(out_ptr, [num_rows, out_pad])
    for i in asc2.range(asc2.block_idx(), num_rows, asc2.block_num(),
                        unroll_factor=2, parallel=True):
        row = asc2.load(x_gm, [1, num_cols], offsets=[i, 0])
        s = asc2.reduce_sum(row)
        result = asc2.full([1, out_pad], s, dtype=row.dtype)
        asc2.store(result, out_gm, offsets=[i, 0])


def reduce_sum_launch(x: np.ndarray) -> np.ndarray:
    num_rows, num_cols = x.shape
    out = np.zeros((num_rows, OUT_PAD), dtype=x.dtype)
    reduce_sum_kernel[CORE_NUM](x, out, num_rows, num_cols, OUT_PAD)
    return out[:, 0]


def run_kernel(backend: config.Backend, platform: config.Platform):
    config.set_platform(backend, platform)

    rng = np.random.default_rng(seed=2026)
    x = rng.random((32, 4096), dtype=np.float32)
    out = reduce_sum_launch(x)
    expected = x.sum(axis=1)
    np.testing.assert_allclose(out, expected, atol=1e-2, rtol=1e-3)
    logging.info("[PASS] reduce_sum verified for shape (32, 4096).")


def test_reduce_sum_f32(backend: config.Backend, platform: config.Platform):
    """pytest entry point."""
    run_kernel(backend, platform)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", type=str, default="Model", help="backend: Model or NPU")
    parser.add_argument("-v", type=str, default="Ascend950PR_9599", help="platform/SoC version")
    args = parser.parse_args()
    backend = args.r
    platform = args.v
    if backend not in config.Backend.__members__:
        raise ValueError(f"Unsupported Backend! Supported: {list(config.Backend.__members__.keys())}")
    backend = config.Backend(backend)
    if platform is not None:
        platform_values = [p.value for p in config.Platform]
        if platform not in platform_values:
            raise ValueError(f"Unsupported Platform! Supported: {platform_values}")
        platform = config.Platform(platform)
    logging.info(f"[INFO] Running kernel with backend={backend}, platform={platform}")
    run_kernel(backend, platform)
    logging.info("[INFO] Kernel run complete.")
