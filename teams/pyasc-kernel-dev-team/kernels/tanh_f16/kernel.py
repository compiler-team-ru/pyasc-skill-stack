#!/usr/bin/env python3.11
"""
pyasc kernel: tanh_f16

Operation: Element-wise hyperbolic tangent for float16 tensors (out = tanh(x))
Usage:
    python3.11 kernel.py -r Model -v Ascend950PR_9599   # Run with simulator
    python3.11 kernel.py -r NPU                         # Run with NPU hardware
    pytest kernel.py --backend Model --platform Ascend950PR_9599

Performance note:
    TILE_SIZE=2048 (perf-optimized for the AIV vector pipeline), mirroring the
    abs_f16 cell. A standalone single-`asc2.tanh` tile stays well within the
    150s sim budget (the budget concern documented on the gelu cell is specific
    to composing tanh inside the full GELU polynomial, not a lone tanh op).

    Alignment requirement: input size must be a multiple of
    TILE_SIZE * CORE_NUM = 32768.
"""

import logging
import argparse
import numpy as np

import asc
import asc.runtime.config as config
import asc2

TILE_SIZE = 2048
CORE_NUM = 16

logging.basicConfig(level=logging.INFO)


@asc2.jit(always_compile=True)
def tanh_kernel(x_ptr: asc.GlobalAddress, out_ptr: asc.GlobalAddress,
                size: int, tile_size: asc.ConstExpr[int], tile_per_block: asc.ConstExpr[int]):
    x_gm = asc2.tensor(x_ptr, [size])
    out_gm = asc2.tensor(out_ptr, [size])
    base_offset = asc2.block_idx() * tile_size * tile_per_block
    for i in asc2.range(tile_per_block, unroll_factor=2, parallel=True):
        tile_offset = base_offset + i * tile_size
        x = asc2.load(x_gm, [tile_size], offsets=[tile_offset])
        out = asc2.tanh(x)
        asc2.store(out, out_gm, offsets=[tile_offset])


def tanh_launch(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x)
    size = out.size
    num_tiles = asc.ceildiv(size, TILE_SIZE)
    tanh_kernel[CORE_NUM](x, out, size, TILE_SIZE, asc.ceildiv(num_tiles, CORE_NUM))
    return out


def run_kernel(backend: config.Backend, platform: config.Platform):
    config.set_platform(backend, platform)

    test_shapes = [[1, 128], [4, 2048], [32, 4096]]
    rng = np.random.default_rng(seed=2026)

    for shape in test_shapes:
        flat_size = int(np.prod(shape))
        if flat_size % (TILE_SIZE * CORE_NUM) != 0:
            padded_size = ((flat_size + TILE_SIZE * CORE_NUM - 1) // (TILE_SIZE * CORE_NUM)) * (TILE_SIZE * CORE_NUM)
            logging.info(f"[INFO] Shape {shape} (size={flat_size}) padded to {padded_size} for alignment")
        else:
            padded_size = flat_size

        x_flat = (rng.random(padded_size, dtype=np.float32) * 10 - 5).astype(np.float16)
        out_flat = tanh_launch(x_flat)

        x_original = x_flat[:flat_size]
        out_original = out_flat[:flat_size]
        expected = np.tanh(x_original.astype(np.float32)).astype(np.float16)

        np.testing.assert_allclose(out_original.astype(np.float32),
                                   expected.astype(np.float32), atol=1e-2, rtol=1e-2)
        logging.info(f"[PASS] Kernel output verified for shape {shape} (flat_size={flat_size}).")


def test_tanh_f16(backend: config.Backend, platform: config.Platform):
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
