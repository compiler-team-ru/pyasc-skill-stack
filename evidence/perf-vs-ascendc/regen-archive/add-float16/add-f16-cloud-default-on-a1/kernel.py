#!/usr/bin/env python3.11
"""
pyasc kernel: add_f16

Operation: Element-wise binary addition (out = x + y)
Usage:
    python3.11 kernel.py -r Model -v Ascend950PR_9599   # Run with simulator
    python3.11 kernel.py -r NPU                         # Run with NPU hardware
    pytest kernel.py --backend Model --platform Ascend950PR_9599
"""

import logging
import argparse
import numpy as np

import asc
import asc.runtime.config as config
import asc2

TILE_SIZE = 2048
CORE_NUM = 16
ALIGNMENT = TILE_SIZE * CORE_NUM  # 32768 elements

logging.basicConfig(level=logging.INFO)


@asc2.jit(always_compile=True)
def add_kernel(x_ptr: asc.GlobalAddress, y_ptr: asc.GlobalAddress, out_ptr: asc.GlobalAddress,
               size: int, tile_size: asc.ConstExpr[int], tile_per_block: asc.ConstExpr[int]):
    """Element-wise addition kernel: out = x + y
    
    Args:
        x_ptr: Global memory pointer for input x
        y_ptr: Global memory pointer for input y
        out_ptr: Global memory pointer for output
        size: Total number of elements (must be aligned to TILE_SIZE * CORE_NUM)
        tile_size: Elements per tile (compile-time constant)
        tile_per_block: Tiles per core (compile-time constant)
    """
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
    """Host launcher for add kernel.
    
    Args:
        x: Input tensor (will be flattened to 1D)
        y: Input tensor (will be flattened to 1D)
    
    Returns:
        Output tensor with same shape as input
    """
    original_shape = x.shape
    x_flat = x.flatten()
    y_flat = y.flatten()
    size = x_flat.size
    
    # Aligned-only: pad to multiple of ALIGNMENT
    if size % ALIGNMENT != 0:
        padded_size = ((size + ALIGNMENT - 1) // ALIGNMENT) * ALIGNMENT
        x_padded = np.zeros(padded_size, dtype=x.dtype)
        y_padded = np.zeros(padded_size, dtype=y.dtype)
        x_padded[:size] = x_flat
        y_padded[:size] = y_flat
        out_padded = np.empty_like(x_padded)
        
        num_tiles = asc.ceildiv(padded_size, TILE_SIZE)
        add_kernel[CORE_NUM](x_padded, y_padded, out_padded, padded_size, 
                           TILE_SIZE, asc.ceildiv(num_tiles, CORE_NUM))
        
        out_flat = out_padded[:size]
    else:
        out_flat = np.empty_like(x_flat)
        num_tiles = asc.ceildiv(size, TILE_SIZE)
        add_kernel[CORE_NUM](x_flat, y_flat, out_flat, size, 
                           TILE_SIZE, asc.ceildiv(num_tiles, CORE_NUM))
    
    return out_flat.reshape(original_shape)


def run_kernel(backend: config.Backend, platform: config.Platform):
    """Run kernel verification with test shapes."""
    config.set_platform(backend, platform)
    test_shapes = [(1, 128), (4, 2048), (32, 4096)]
    rng = np.random.default_rng(seed=2026)
    
    for shape in test_shapes:
        # Generate test data as float32, then cast to float16
        x = rng.random(shape, dtype=np.float32).astype(np.float16)
        y = rng.random(shape, dtype=np.float32).astype(np.float16)
        out = add_launch(x, y)
        expected = x + y
        np.testing.assert_allclose(out, expected, atol=1e-3, rtol=1e-3)
        logging.info(f"[PASS] Kernel output verified for shape {shape}.")


def test_add_f16(backend: config.Backend, platform: config.Platform):
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