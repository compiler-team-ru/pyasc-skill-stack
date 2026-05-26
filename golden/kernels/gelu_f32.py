#!/usr/bin/env python3.10
"""Golden kernel: gelu/float32

GELU activation, tanh / Padé approximation form (PyTorch's
``gelu(approximate='tanh')`` / TensorFlow's ``gelu(approximate=True)``):

    gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))

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
  - regime_note: tanh/Padé approximation form; do not use asc2.erf
    on float32 (simulator erf is too noisy).

Non-obvious constraints (Phase 1.5):
  - Numerical: the original ``erf`` form was too noisy on the CANN
    simulator (``asc2.erf`` exhibits up to ~4.7 absolute error vs
    the host ``math.erf`` on float32 and flaked 2-of-3 nightlies
    even with atol=2.0). The tanh/Padé form sidesteps simulator
    erf entirely and is bit-exact against numpy at TILE_SIZE=64.
  - Tile width: ``TILE_SIZE = 64`` (conservative single-lane width
    for C310). Wider tiles can hit a lowering bug that only writes
    the first 64 elements — leaving the tile at 64 keeps the
    correctness contract while the perf retune is filed as a
    follow-up.
  - Alignment: input ``size`` must be a multiple of
    ``TILE_SIZE * CORE_NUM = 64 * 16 = 1024``; smaller inputs trip
    C310's stricter MTE GDMA burst alignment check (size=128 is
    excluded from the test set for this reason).
  - UB/L1/L0 placement: every op (``asc2.tanh``, the multiplies,
    the additions) runs in UB. No L0 / cube involvement.
  - Padding: none.
  - Tail behavior: aligned_only.
  - Accumulator dtype: null. Composition is elementwise in float32
    throughout.
  - CRITICAL ordering rule: inside the ``@asc2.jit`` body, every
    scalar-times-Tile multiplication MUST put the Tile on the left
    (``x * GELU_K``, NOT ``GELU_K * x``). The asc2 Tile class lacks
    ``__rmul__``, so scalar-on-left fails at codegen with
    ``AttributeError: 'Tile' object has no attribute '__rmul__'``.
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

    for size in [8192, 131072]:
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
