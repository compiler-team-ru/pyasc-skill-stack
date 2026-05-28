#!/usr/bin/env python3.10
"""Golden kernel: add/float16

Element-wise add (``out = x + y``) for float16 tensors.

Vendored from pyasc-v2-eval@7b85554a:
  python/test/asc2/target/test_vadd.py
    -- 1D-flatten tiling skeleton (Pattern A in pyasc-api-patterns/SKILL.md);
       upstream parametrizes 8 (core_num, unroll_factor, input_shape, dtype)
       tuples with cache-line aligned tail handling. We adopt the first
       three shapes for our golden test (8192, 9216, 87768).
  python/test/asc2/kernels/test_vadd.py
    -- aligned 1D launch wrapper without tail handling (matches Pattern A).
numpy / numpy.testing.assert_allclose adapted from
torch / torch.testing.assert_close (see docs/golden-upstream-map.md).

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

Non-obvious constraints (Phase 9):
  - Rank-consistent tiling: 1D tensor + 1D load shape + 1D offsets.
    The kernel never sees the multi-dim test shape; the host flattens
    to ``size`` before launch. Mixing ranks (2D tensor with a 1D
    ``[tile_size]`` load and 2D ``[row_idx, col_idx]`` offsets) raises
    ``RuntimeError: rank of 'tensor_shape' must match rank of 'shape'``
    on v2.
  - Alignment: ``size`` is rounded up to a multiple of
    ``TILE_SIZE * CORE_NUM = 128 * 16 = 2048`` for the test path; the
    production Pattern B skeleton (cache-line aligned tail) is
    documented in SKILL.md for cells with ``tail_behavior: padded``.
  - UB/L1/L0 placement: each tile loaded into UB, ``+`` operates
    in-place, result stored straight back to GM. No L0 / cube
    involvement.
  - Tolerance: ``atol=rtol=1e-3`` matches the tolerance contract in
    ``capabilities.yaml`` for ``add/float16``. Upstream
    ``target/test_vadd.py`` uses ``atol=rtol=1e-3`` for float32.
  - Simulator/platform assumptions: ``Ascend950PR_9599`` (C310);
    numpy buffers are safe for this elementwise UB-only path.
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
def add_kernel(x_ptr: asc.GlobalAddress, y_ptr: asc.GlobalAddress, out_ptr: asc.GlobalAddress,
               size: int, tile_size: asc.ConstExpr[int],
               tile_per_block: asc.ConstExpr[int]):
    x_gm = asc2.tensor(x_ptr, [size])
    y_gm = asc2.tensor(y_ptr, [size])
    out_gm = asc2.tensor(out_ptr, [size])
    base_offset = asc2.block_idx() * tile_size * tile_per_block
    for i in asc2.range(tile_per_block, unroll_factor=2, parallel=True):
        tile_offset = base_offset + i * tile_size
        x = asc2.load(x_gm, [tile_size], offsets=[tile_offset])
        y = asc2.load(y_gm, [tile_size], offsets=[tile_offset])
        out = x + y
        asc2.store(out, out_gm, offsets=[tile_offset])


def add_launch(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    out = np.empty_like(x)
    size = out.size
    num_tiles = asc.ceildiv(size, TILE_SIZE)
    add_kernel[CORE_NUM](x, y, out, size, TILE_SIZE, asc.ceildiv(num_tiles, CORE_NUM))
    return out


def run_kernel(backend: config.Backend, platform: config.Platform):
    config.set_platform(backend, platform)

    # Upstream target/test_vadd.py parametrizes 8 cases; we exercise three
    # that fit our aligned_only contract (multiples of TILE_SIZE * CORE_NUM
    # = 2048): 8192 from kernels/test_vadd.py; 16384 and 131072 from the
    # target multi-shape spread.
    test_sizes = [8192, 16384, 131072]
    rng = np.random.default_rng(seed=2026)

    for size in test_sizes:
        x = rng.random(size, dtype=np.float32).astype(np.float16)
        y = rng.random(size, dtype=np.float32).astype(np.float16)
        out = add_launch(x, y)
        expected = x + y
        np.testing.assert_allclose(out, expected, atol=1e-3, rtol=1e-3)
        logging.info(f"[PASS] Kernel output verified for size {size}.")


def test_add_f16(backend: config.Backend, platform: config.Platform):
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
