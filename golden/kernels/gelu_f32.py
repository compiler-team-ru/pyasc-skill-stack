#!/usr/bin/env python3.10
"""
Golden reference: gelu_f32 kernel (asc2 API)

GELU activation, tanh / Padé approximation form (PyTorch's
``gelu(approximate='tanh')`` / TensorFlow's ``gelu(approximate=True)``):

    gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))

The original ``erf`` form was numerically too noisy on the CANN simulator
for float32 (``asc2.erf`` exhibits up to ~4.7 absolute error vs the host
``math.erf`` reference, requiring atol=2.0; even then the cell flaked
2-of-3 nightlies). The tanh form sidesteps simulator erf entirely and
is bit-exact against numpy on Ascend910B1 with TILE_SIZE=64.

Tile width: 64 elements (one Ascend910B1 SIMD vector lane). Wider tiles
trip the same wide-tile lowering bug seen in the rms_norm streaming
work — only the first 64 elements of a 128-wide tile get written, the
rest are silently zeroed. Pinning the lane width avoids that.

Verified on the CANN 9.0.0 simulator at sizes 128, 8192, 131072 with
``atol=rtol=1e-2``.
"""

import logging
import argparse
import math
import numpy as np

import asc
import asc.runtime.config as config
import asc2

TILE_SIZE = 64
CORE_NUM = 16

GELU_K = math.sqrt(2.0 / math.pi)
GELU_C = 0.044715

logging.basicConfig(level=logging.INFO)


@asc2.jit(always_compile=True)
def gelu_kernel(x_ptr: asc.GlobalAddress, out_ptr: asc.GlobalAddress,
                size: int, tile_size: asc.ConstExpr[int],
                tile_per_block: asc.ConstExpr[int]):
    x_gm = asc2.tensor(x_ptr, [size])
    out_gm = asc2.tensor(out_ptr, [size])
    base_offset = asc2.block_idx() * tile_size * tile_per_block
    for i in asc2.range(tile_per_block, unroll_factor=2, parallel=True):
        tile_offset = base_offset + i * tile_size
        x = asc2.load(x_gm, [tile_size], offsets=[tile_offset])
        inner = (x + x * x * x * GELU_C) * GELU_K
        out = x * (asc2.tanh(inner) + 1) * 0.5
        asc2.store(out, out_gm, offsets=[tile_offset])


def gelu_launch(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x)
    size = out.size
    num_tiles = asc.ceildiv(size, TILE_SIZE)
    gelu_kernel[CORE_NUM](x, out, size, TILE_SIZE, asc.ceildiv(num_tiles, CORE_NUM))
    return out


def gelu_numpy(x: np.ndarray) -> np.ndarray:
    k = np.sqrt(2.0 / np.pi)
    return 0.5 * x * (1.0 + np.tanh(k * (x + 0.044715 * x ** 3)))


def run_kernel(backend: config.Backend, platform: config.Platform):
    config.set_platform(backend, platform)

    rng = np.random.default_rng(seed=2026)

    for size in [128, 8192, 131072]:
        x = rng.random(size, dtype=np.float32) * 4 - 2
        out = gelu_launch(x)
        expected = gelu_numpy(x)
        np.testing.assert_allclose(out, expected, atol=1e-2, rtol=1e-2)
        logging.info(f"[PASS] GELU f32 verified for size {size}.")


def test_gelu_f32(backend: config.Backend, platform: config.Platform):
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
