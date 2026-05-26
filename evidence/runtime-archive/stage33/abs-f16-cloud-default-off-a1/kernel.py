#!/usr/bin/env python3.10
"""Kernel: abs/float16

Element-wise absolute value for float16 tensors.
Verified on CANN simulator with Ascend950PR_9599 platform.

Cell metadata:
  - shape_regime: fixed
  - reduce_axis: null
  - output_shape: same_as_input
  - accumulator_dtype: null
  - identity: null
  - tail_behavior: aligned_only
  - padding: null
  - partitioning: tile_per_core
  - unsupported_regimes: []

Test shapes: [1, 128], [4, 2048], [32, 4096]
TILE_SIZE: 128 (divides all shapes evenly)
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
def abs_kernel(x_ptr: asc.GlobalAddress, out_ptr: asc.GlobalAddress, size: int,
               tile_size: asc.ConstExpr[int]):
    x_gm = asc2.tensor(x_ptr, [size])
    out_gm = asc2.tensor(out_ptr, [size])
    num_tiles = asc2.num_tiles(x_gm, axis=0, shape=[tile_size])
    for tile_id in asc2.range(asc2.block_idx(), num_tiles, asc2.block_num()):
        x = asc2.load(x_gm, [tile_size], tile_id=[tile_id])
        out = asc2.abs(x)
        asc2.store(out, out_gm, tile_id=[tile_id])


def abs_launch(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x)
    size = out.size
    abs_kernel[CORE_NUM](x, out, size, TILE_SIZE)
    return out


def run_kernel(backend: config.Backend, platform: config.Platform):
    config.set_platform(backend, platform)

    test_shapes = [(1, 128), (4, 2048), (32, 4096)]
    rng = np.random.default_rng(seed=2026)

    for shape in test_shapes:
        x = (rng.random(shape, dtype=np.float32) * 10 - 5).astype(np.float16)
        out = abs_launch(x)
        expected = np.abs(x)
        np.testing.assert_allclose(out, expected, atol=1e-3, rtol=1e-3)
        logging.info(f"[PASS] Kernel output verified for shape {shape}.")


def test_abs_f16(backend: config.Backend, platform: config.Platform):
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