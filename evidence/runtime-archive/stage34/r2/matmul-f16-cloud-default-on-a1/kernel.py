#!/usr/bin/env python3.10
"""
pyasc kernel: matmul_float16

Operation: C = A @ B (matrix multiplication)
- Inputs: A (M×K, float16), B (K×N, float16)
- Output: C (M×N, float32) — cube accumulator always produces float32
- Tiling: block_grid with L0A/L0B locations
- Platform: Ascend950PR_9599 (cube unit required)

Usage:
    python3.10 kernel.py -r Model -v Ascend950PR_9599
"""

import logging
import argparse
import torch

import asc
import asc.runtime.config as config
import asc2

CORE_NUM = 1

logging.basicConfig(level=logging.INFO)


@asc2.jit(always_compile=True)
def matmul_kernel(a_ptr: asc.GlobalAddress, b_ptr: asc.GlobalAddress, c_ptr: asc.GlobalAddress,
                  a_shape: asc.ConstExpr[tuple], b_shape: asc.ConstExpr[tuple], c_shape: asc.ConstExpr[tuple],
                  m_tile: asc.ConstExpr[int], m_tiles_per_block: asc.ConstExpr[int],
                  n_tile: asc.ConstExpr[int], n_tiles_per_block: asc.ConstExpr[int]):
    a_gm = asc2.tensor(a_ptr, a_shape)
    b_gm = asc2.tensor(b_ptr, b_shape)
    c_gm = asc2.tensor(c_ptr, c_shape)

    block_id = asc2.block_idx()
    m_elems_per_block = m_tile * m_tiles_per_block
    m_base_off = (m_elems_per_block * block_id) % a_shape[0]
    n_base_off = ((m_elems_per_block * block_id) // a_shape[0]) * (n_tile * n_tiles_per_block)

    for j in range(n_tiles_per_block):
        b_offset = n_base_off + j * n_tile
        b_j = asc2.load(b_gm, [b_shape[0], n_tile], offsets=[0, b_offset],
                        location=asc2.TileLocation.L0B)
        for i in range(m_tiles_per_block):
            a_offset = m_base_off + i * m_tile
            a_i = asc2.load(a_gm, [m_tile, a_shape[1]], offsets=[a_offset, 0],
                            location=asc2.TileLocation.L0A)
            c_ij = a_i @ b_j
            asc2.store(c_ij, c_gm, offsets=[a_offset, b_offset])


def matmul_launch(a: torch.Tensor, b: torch.Tensor, m_tile: int, n_tile: int,
                  core_num: int = CORE_NUM) -> torch.Tensor:
    m, k = a.shape
    k2, n = b.shape
    assert k == k2, f"K dimension mismatch: {k} vs {k2}"
    
    c = torch.zeros((m, n), dtype=torch.float32)
    
    m_tiles = m // m_tile
    n_tiles = n // n_tile
    num_tiles = m_tiles * n_tiles
    m_tiles_per_block = m_tiles
    n_tiles_per_block = n_tiles
    
    matmul_kernel[core_num](a, b, c, a.shape, b.shape, c.shape,
                            m_tile, m_tiles_per_block,
                            n_tile, n_tiles_per_block)
    return c


def run_kernel(backend: config.Backend, platform: config.Platform):
    config.set_platform(backend, platform)

    test_cases = [
        (16, 16, 16, 16, 16),
        (32, 16, 32, 16, 16),
    ]

    for m, k, n, m_tile, n_tile in test_cases:
        a = torch.rand((m, k), dtype=torch.float16)
        b = torch.rand((k, n), dtype=torch.float16)
        
        c = matmul_launch(a, b, m_tile, n_tile, core_num=CORE_NUM)
        
        c_ref = a.to(torch.float32) @ b.to(torch.float32)
        torch.testing.assert_close(c, c_ref, atol=1e-2, rtol=1e-2)
        logging.info(f"[PASS] Kernel output verified for shape ({m}, {k}, {n}).")


def test_matmul_f16(backend: config.Backend, platform: config.Platform):
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