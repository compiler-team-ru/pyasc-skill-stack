#!/usr/bin/env python3.11
"""
pyasc kernel: reduce_sum_float32

Operation: Reduce sum along the last axis (axis=-1)
Input: 2D tensor [num_rows, num_cols] of float32
Output: 1D tensor [num_rows] of float32 (stored as [num_rows, out_pad] with padding)

Usage:
    python3.11 kernel.py -r Model -v Ascend950PR_9599   # Run with simulator
    python3.11 kernel.py -r NPU                         # Run with NPU hardware
    pytest kernel.py --backend Model --platform Ascend950PR_9599
"""

import logging
import argparse
import numpy as np

import asc
import asc.runtime.config as config
import asc2

CORE_NUM = 16
OUT_PAD = 8  # 8 float32 elements = 32 bytes for alignment

logging.basicConfig(level=logging.INFO)


@asc2.jit(always_compile=True)
def reduce_sum_kernel(x_ptr: asc.GlobalAddress, out_ptr: asc.GlobalAddress,
                      num_rows: int, num_cols: asc.ConstExpr[int],
                      out_pad: asc.ConstExpr[int]):
    """
    Reduce sum along the last axis (axis=-1).
    
    Each core processes rows starting from block_idx() with stride block_num().
    For each row, load the full row, reduce_sum, and store as a padded scalar.
    """
    x_gm = asc2.tensor(x_ptr, [num_rows, num_cols])
    out_gm = asc2.tensor(out_ptr, [num_rows, out_pad])
    
    for i in asc2.range(asc2.block_idx(), num_rows, asc2.block_num(),
                        unroll_factor=2, parallel=True):
        row = asc2.load(x_gm, [1, num_cols], offsets=[i, 0])
        s = asc2.reduce_sum(row)
        result = asc2.full([1, out_pad], s, dtype=row.dtype)
        asc2.store(result, out_gm, offsets=[i, 0])


def reduce_sum_launch(x: np.ndarray, out_pad: int = OUT_PAD) -> np.ndarray:
    """
    Launch the reduce_sum kernel.
    
    Args:
        x: Input 2D array of shape [num_rows, num_cols]
        out_pad: Padding for output alignment (default 8 for float32)
    
    Returns:
        1D array of shape [num_rows] containing the sum of each row
    """
    num_rows, num_cols = x.shape
    out_padded = np.zeros((num_rows, out_pad), dtype=x.dtype)
    reduce_sum_kernel[CORE_NUM](x, out_padded, num_rows, num_cols, out_pad)
    return out_padded[:, 0]


def run_kernel(backend: config.Backend, platform: config.Platform):
    """Run kernel verification with test shapes."""
    config.set_platform(backend, platform)
    
    test_shapes = [
        (32, 4096),  # Primary test shape from task
        (16, 4096),  # Fewer rows than cores
    ]
    rng = np.random.default_rng(seed=2026)
    
    for num_rows, num_cols in test_shapes:
        x = (rng.random((num_rows, num_cols), dtype=np.float32) * 10 - 5).astype(np.float32)
        out = reduce_sum_launch(x)
        expected = np.sum(x.astype(np.float32), axis=1)
        np.testing.assert_allclose(out, expected, atol=1e-2, rtol=1e-3)
        logging.info(f"[PASS] Kernel output verified for shape ({num_rows}, {num_cols}).")


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