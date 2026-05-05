#!/usr/bin/env python3.10
"""
Golden reference: leaky_relu_f16 kernel (asc2 API)
Conditional operation using asc2.where: out = where(x >= 0, x, alpha * x).
Verified on CANN simulator with Ascend910B1 platform.
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
def leaky_relu_kernel(x_ptr: asc.GlobalAddress, alpha: float,
                      out_ptr: asc.GlobalAddress, size: int,
                      tile_size: asc.ConstExpr[int],
                      tile_per_block: asc.ConstExpr[int]):
    x_gm = asc2.tensor(x_ptr, [size])
    out_gm = asc2.tensor(out_ptr, [size])
    base_offset = asc2.block_idx() * tile_size * tile_per_block
    for i in asc2.range(tile_per_block, unroll_factor=2, parallel=True):
        tile_offset = base_offset + i * tile_size
        x = asc2.load(x_gm, [tile_size], offsets=[tile_offset])
        out = asc2.where(x >= 0, x, x * alpha)
        asc2.store(out, out_gm, offsets=[tile_offset])


def leaky_relu_launch(x: np.ndarray, alpha: float) -> np.ndarray:
    out = np.empty_like(x)
    size = out.size
    num_tiles = asc.ceildiv(size, TILE_SIZE)
    leaky_relu_kernel[CORE_NUM](x, alpha, out, size, TILE_SIZE,
                                asc.ceildiv(num_tiles, CORE_NUM))
    return out


def run_kernel(backend: config.Backend, platform: config.Platform):
    config.set_platform(backend, platform)

    test_sizes = [128, 8192, 131072]
    alpha = 0.01
    rng = np.random.default_rng(seed=2026)

    for size in test_sizes:
        x = (rng.random(size, dtype=np.float32) * 4 - 2).astype(np.float16)
        out = leaky_relu_launch(x, alpha)
        expected = np.where(x >= 0, x, x * alpha)
        np.testing.assert_allclose(out.astype(np.float32),
                                   expected.astype(np.float32),
                                   atol=1e-3, rtol=1e-3)
        logging.info(f"[PASS] leaky_relu verified for size {size}.")


def test_leaky_relu_f16(backend: config.Backend, platform: config.Platform):
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
