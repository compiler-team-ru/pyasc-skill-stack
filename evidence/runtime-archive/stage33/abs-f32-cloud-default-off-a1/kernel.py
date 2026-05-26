#!/usr/bin/env python3.10
"""Golden kernel: abs/float32

Element-wise absolute value for float32 tensors.
Verified on CANN simulators with Ascend950PR_9599 platform.

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
    ``TILE_SIZE * CORE_NUM = 128 * 16 = 2048`` for CORE_NUM=16; smaller
    inputs may require fewer cores. Test sizes: [1, 128] uses 1 core,
    [4, 2048] and [32, 4096] use 16 cores.
  - UB/L1/L0 placement: each tile is loaded into UB (default
    location of ``asc2.load`` with no ``location=`` arg); the abs
    is computed in-place in UB and stored straight back to GM. No
    L0 / cube involvement.
  - Padding: none. ``tail_behavior=aligned_only`` — the kernel
    does not handle partial tiles; the dispatch shape (``size``)
    must be an exact multiple of ``TILE_SIZE``.
  - Tail behavior: aligned_only. Non-multiple shapes are out of
    scope (see ``unsupported_regimes=[]`` — i.e. no opt-in regime
    extends this cell).
  - Accumulator dtype: null. ``asc2.abs`` is a pure elementwise op
    with no accumulation.
  - Simulator/platform assumptions: ``Ascend950PR_9599`` (C310);
    numpy ``np.empty_like`` for the output is fine here because
    this is an elementwise op without cube/matmul involvement
    (the numpy-zeroing hazard documented in matmul/rms_norm does
    not apply to elementwise UB-only kernels).
"""

import logging
import argparse
import numpy as np

import asc
import asc.runtime.config as config
import asc2

TILE_SIZE = 128

logging.basicConfig(level=logging.INFO)


@asc2.jit(always_compile=True)
def abs_kernel(x_ptr: asc.GlobalAddress, out_ptr: asc.GlobalAddress,
              size: int, tile_size: asc.ConstExpr[int], tile_per_block: asc.ConstExpr[int]):
    x_gm = asc2.tensor(x_ptr, [size])
    out_gm = asc2.tensor(out_ptr, [size])
    base_offset = asc2.block_idx() * tile_size * tile_per_block
    for i in asc2.range(tile_per_block, unroll_factor=2, parallel=True):
        tile_offset = base_offset + i * tile_size
        x = asc2.load(x_gm, [tile_size], offsets=[tile_offset])
        out = asc2.abs(x)
        asc2.store(out, out_gm, offsets=[tile_offset])


def abs_launch(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x)
    size = out.size
    num_tiles = asc.ceildiv(size, TILE_SIZE)
    core_num = min(16, num_tiles)
    abs_kernel[core_num](x, out, size, TILE_SIZE, asc.ceildiv(num_tiles, core_num))
    return out


def run_kernel(backend: config.Backend, platform: config.Platform):
    config.set_platform(backend, platform)

    test_shapes = [(1, 128), (4, 2048), (32, 4096)]
    rng = np.random.default_rng(seed=2026)

    for shape in test_shapes:
        x = (rng.random(shape, dtype=np.float32) * 10 - 5).astype(np.float32)
        out = abs_launch(x)
        expected = np.abs(x)
        np.testing.assert_allclose(out, expected, atol=1e-3, rtol=1e-3)
        logging.info(f"[PASS] Kernel output verified for shape {shape}.")


def test_abs_f32(backend: config.Backend, platform: config.Platform):
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