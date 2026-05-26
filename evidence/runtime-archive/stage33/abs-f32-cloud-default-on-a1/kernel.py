#!/usr/bin/env python3.10
"""
pyasc kernel: abs_float32

Operation: out = |x| (absolute value)
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
               size: int, tile_size: asc.ConstExpr[int],
               num_tiles: asc.ConstExpr[int]):
    x_gm = asc2.tensor(x_ptr, [size])
    out_gm = asc2.tensor(out_ptr, [size])
    for tile_id in asc2.range(asc2.block_idx(), num_tiles, asc2.block_num(),
                              unroll_factor=2, parallel=True):
        tile_offset = tile_id * tile_size
        x_tile = asc2.load(x_gm, [tile_size], offsets=[tile_offset])
        out_tile = asc2.abs(x_tile)
        asc2.store(out_tile, out_gm, offsets=[tile_offset])


def kernel_launch(x: np.ndarray) -> np.ndarray:
    x_flat = x.reshape(-1)
    out_flat = np.empty_like(x_flat)
    size = x_flat.size
    num_tiles = asc.ceildiv(size, TILE_SIZE)
    abs_kernel[CORE_NUM](x_flat, out_flat, size, TILE_SIZE, num_tiles)
    return out_flat.reshape(x.shape)


def run_kernel(backend: config.Backend, platform: config.Platform):
    config.set_platform(backend, platform)
    test_shapes = [[1, 128], [4, 2048], [32, 4096]]
    rng = np.random.default_rng(seed=2026)
    for shape in test_shapes:
        total_size = np.prod(shape)
        x = rng.random(total_size, dtype=np.float32) * 20 - 10
        x = x.reshape(shape)
        out = kernel_launch(x)
        expected = np.abs(x)
        np.testing.assert_allclose(out, expected, atol=1e-3, rtol=1e-3)
        logging.info(f"[PASS] Kernel output verified for shape {shape}.")


def test_kernel(backend: config.Backend, platform: config.Platform):
    """pytest entry point — uses conftest.py fixtures for backend/platform."""
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