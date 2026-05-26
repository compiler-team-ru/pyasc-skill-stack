#!/usr/bin/env python3.10
"""
pyasc kernel: reduce_sum-float16

Operation: Row-wise sum reduction along the last axis.
For input x with shape [num_rows, num_cols], computes:
    out[i] = sum_j x[i, j]    for i in 0..num_rows-1

Usage:
    python3.10 kernel.py -r Model -v Ascend950PR_9599   # Run with simulator
    python3.10 kernel.py -r NPU                         # Run with NPU hardware
    pytest kernel.py --backend Model --platform Ascend950PR_9599
"""

import logging
import argparse
import numpy as np

import asc
import asc.runtime.config as config
import asc2

OUT_PAD = 16
CORE_NUM = 16

logging.basicConfig(level=logging.INFO)


@asc2.jit(always_compile=True)
def reduce_sum_kernel(x_ptr: asc.GlobalAddress, out_ptr: asc.GlobalAddress,
                     num_rows: int, num_cols: asc.ConstExpr[int],
                     out_pad: asc.ConstExpr[int]):
    x_gm = asc2.tensor(x_ptr, [num_rows, num_cols])
    out_gm = asc2.tensor(out_ptr, [num_rows, out_pad])
    
    for r in asc2.range(asc2.block_idx(), num_rows, asc2.block_num(),
                        unroll_factor=2, parallel=True):
        row = asc2.load(x_gm, [1, num_cols], offsets=[r, 0])
        row_f32 = row.to(asc.float32)
        s = asc2.reduce_sum(row_f32)
        result = asc2.full([1, out_pad], s, dtype=row.dtype)
        asc2.store(result, out_gm, offsets=[r, 0])


def reduce_sum_launch(x: np.ndarray) -> np.ndarray:
    num_rows, num_cols = x.shape
    out = np.zeros((num_rows, OUT_PAD), dtype=x.dtype)
    reduce_sum_kernel[CORE_NUM](x, out, num_rows, num_cols, OUT_PAD)
    return out[:, 0]


def run_kernel(backend: config.Backend, platform: config.Platform):
    config.set_platform(backend, platform)
    
    test_shapes = [(32, 4096)]
    rng = np.random.default_rng(seed=2026)
    
    for shape in test_shapes:
        x = rng.random(shape, dtype=np.float32).astype(np.float16)
        out = reduce_sum_launch(x)
        expected = np.sum(x.astype(np.float32), axis=1)
        np.testing.assert_allclose(out.astype(np.float32), expected, atol=2.0, rtol=5e-2)
        logging.info(f"[PASS] Kernel output verified for shape {shape}.")


def test_reduce_sum_f16(backend: config.Backend, platform: config.Platform):
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