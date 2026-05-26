#!/usr/bin/env python3.10
"""Golden kernel: gelu/float16

GELU activation composed from erf, mul, add -- no single asc2.gelu builtin.
  gelu(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
Verified on CANN simulator with Ascend950PR_9599 platform.

Cell metadata (mirrors capabilities.yaml; do not drift):
  - shape_regime: fixed
  - reduce_axis: null
  - output_shape: same_as_input
  - accumulator_dtype: null
  - identity: null
  - tail_behavior: aligned_only
  - padding: null
  - partitioning: tile_per_core
  - unsupported_regimes: []

Non-obvious constraints (Phase 1.5):
  - Alignment: input ``size`` must be a multiple of
    ``TILE_SIZE * CORE_NUM = 128 * 16 = 2048``; smaller inputs trip
    C310's stricter MTE GDMA burst alignment check (test sizes
    start at 8192 = ``TILE_SIZE * CORE_NUM * 4``).
  - UB/L1/L0 placement: every per-tile op (``asc2.erf``, the two
    multiplies, the add) runs in UB.  No L0 / cube involvement.
  - Padding: none.
  - Tail behavior: aligned_only — the kernel does not handle
    partial tiles.
  - Accumulator dtype: null. The composition is purely elementwise
    in float16 throughout.
  - Numerical note: ``asc2.erf`` on float16 is exercised here. On
    float32 it has up to ~4.7 absolute error vs ``math.erf`` and is
    forbidden — see ``gelu_f32.py``, which uses the tanh/Padé form.
  - Simulator/platform assumptions: ``Ascend950PR_9599`` (C310);
    numpy buffers are safe for this elementwise UB-only path.
"""

import logging
import argparse
import math
import numpy as np

import asc
import asc.runtime.config as config
import asc2

TILE_SIZE = 128
CORE_NUM = 16

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
        k = asc2.sqrt(0.5)
        out = x * (asc2.erf(x * k) + 1) * 0.5
        asc2.store(out, out_gm, offsets=[tile_offset])


def gelu_launch(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x)
    size = out.size
    num_tiles = asc.ceildiv(size, TILE_SIZE)
    gelu_kernel[CORE_NUM](x, out, size, TILE_SIZE, asc.ceildiv(num_tiles, CORE_NUM))
    return out


_verf = np.vectorize(math.erf)

def gelu_numpy(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + _verf(x / np.sqrt(2.0)))


def run_kernel(backend: config.Backend, platform: config.Platform):
    config.set_platform(backend, platform)

    test_sizes = [8192, 131072]
    rng = np.random.default_rng(seed=2026)

    for size in test_sizes:
        x = (rng.random(size, dtype=np.float32) * 4 - 2).astype(np.float16)
        out = gelu_launch(x)
        expected = gelu_numpy(x.astype(np.float32)).astype(np.float16)
        np.testing.assert_allclose(out.astype(np.float32),
                                   expected.astype(np.float32),
                                   atol=5e-2, rtol=5e-2)
        logging.info(f"[PASS] GELU verified for size {size}.")


def test_gelu_f16(backend: config.Backend, platform: config.Platform):
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
