#!/usr/bin/env python3.10
"""
pyasc kernel: abs_float16

Operation: Element-wise absolute value (out = |x|)
Input shapes: [1, 128], [4, 2048], [32, 4096] (flattened to 1D)
Output shape: same as input
Dtype: float16

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

TILE_SIZE = 128
CORE_NUM = 16

logging.basicConfig(level=logging.INFO)


@asc2.jit(always_compile=True)
def abs_kernel(x_ptr: asc.GlobalAddress, out_ptr: asc.GlobalAddress,
               size: int, tile_size: asc.ConstExpr[int]):
    x_gm = asc2.tensor(x_ptr, [size])
    out_gm = asc2.tensor(out_ptr, [size])
    num_tiles = asc.ceildiv(size, tile_size)
    for tile_idx in asc2.range(asc2.block_idx(), num_tiles, asc2.block_num(),
                               unroll_factor=2, parallel=True):
        offset = tile_idx * tile_size
        x = asc2.load(x_gm, [tile_size], offsets=[offset])
        out = asc2.abs(x)
        asc2.store(out, out_gm, offsets=[offset])


def abs_launch(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x)
    size = out.size
    abs_kernel[CORE_NUM](x, out, size, TILE_SIZE)
    return out


def run_kernel(backend: config.Backend, platform: config.Platform):
    config.set_platform(backend, platform)
    test_shapes = [[1, 128], [4, 2048], [32, 4096]]
    rng = np.random.default_rng(seed=2026)
    for shape in test_shapes:
        size = shape[0] * shape[1]
        x = (rng.random(size, dtype=np.float32) * 10 - 5).astype(np.float16)
        out = abs_launch(x)
        expected = np.abs(x)
        np.testing.assert_allclose(out, expected, atol=1e-3, rtol=1e-3)
        logging.info(f"[PASS] Kernel output verified for shape {shape} (size={size}).")


def test_abs_f16(backend: config.Backend, platform: config.Platform):
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