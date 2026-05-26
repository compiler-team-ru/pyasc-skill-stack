#!/usr/bin/env python3.10
"""Softmax kernel: float16, row-wise, axis=-1, shape [32, 4096].

Row-wise softmax using the asc2.softmax builtin for float16 tensors.
Each row is independently normalized so that it sums to 1.

Specifications:
  - Input shape: [32, 4096] ONLY (full-row regime)
  - Output shape: same_as_input
  - Dtype: float16
  - Layout: contiguous
  - Axis: -1 (last axis)
  - Tiling: row_per_core via block-of-rows distribution
  - Tail: aligned_only (split_row and long_rows_exceeding_UB unsupported)
  - Accumulator: float32 (softmax exp accumulates in fp32 internally)
  - Tolerance: atol=5e-2, rtol=5e-2
  - Platform: Ascend950PR_9599
"""

import logging
import argparse
import numpy as np

import asc
import asc.runtime.config as config
import asc2

CORE_NUM = 16

logging.basicConfig(level=logging.INFO)


@asc2.jit(always_compile=True)
def softmax_kernel(x_ptr: asc.GlobalAddress, out_ptr: asc.GlobalAddress,
                   num_rows: int, num_cols: asc.ConstExpr[int],
                   block_size: asc.ConstExpr[int]):
    x_gm = asc2.tensor(x_ptr, [num_rows, num_cols])
    out_gm = asc2.tensor(out_ptr, [num_rows, num_cols])
    start_row = asc2.block_idx() * block_size
    rows = asc2.load(x_gm, [block_size, num_cols], offsets=[start_row, 0])
    out = asc2.softmax(rows)
    asc2.store(out, out_gm, offsets=[start_row, 0])


def softmax_launch(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x)
    num_rows, num_cols = x.shape
    block_size = asc.ceildiv(num_rows, CORE_NUM)
    softmax_kernel[CORE_NUM](x, out, num_rows, num_cols, block_size)
    return out


def softmax_numpy(x: np.ndarray) -> np.ndarray:
    x_f32 = x.astype(np.float32)
    shifted = x_f32 - x_f32.max(axis=1, keepdims=True)
    exp_x = np.exp(shifted)
    return (exp_x / exp_x.sum(axis=1, keepdims=True)).astype(x.dtype)


def run_kernel(backend: config.Backend, platform: config.Platform):
    config.set_platform(backend, platform)

    rng = np.random.default_rng(seed=2026)
    x = (rng.random((32, 4096), dtype=np.float32) * 4 - 2).astype(np.float16)
    out = softmax_launch(x)
    expected = softmax_numpy(x)
    np.testing.assert_allclose(out.astype(np.float32),
                               expected.astype(np.float32),
                               atol=5e-2, rtol=5e-2)
    logging.info("[PASS] softmax verified for shape (32, 4096).")


def test_softmax_f16(backend: config.Backend, platform: config.Platform):
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