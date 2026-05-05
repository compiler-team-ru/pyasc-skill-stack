#!/usr/bin/env python3.10
"""
Golden reference: reduce_sum_f32 kernel (asc2 API)
Row-wise sum reduction for float32 tensors: [num_rows, num_cols] -> [num_rows].
Verified on CANN simulator with Ascend910B1 platform.
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
    parser.add_argument("-v", type=str, default=None, help="platform/SoC version")
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
