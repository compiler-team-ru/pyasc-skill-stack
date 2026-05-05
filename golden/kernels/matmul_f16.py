#!/usr/bin/env python3.10
"""
Golden reference: matmul_f16 kernel (asc2 API)
Tiled matrix multiplication for float16 inputs, float32 output.

Pattern derived from `python/test/asc2/kernels/test_matmul_mnblock.py` in the
pyasc v2 reference implementation, which was verified to pass on the
Ascend950PR_9599 simulator.

Notes
-----
* asc2.matmul / `@` require torch.Tensor inputs on the simulator.
  numpy arrays are silently lowered to zero, so torch is mandatory here.
* Platform must be Ascend950PR_9599 (cube unit). Ascend910B1 does not
  expose the cube ops needed by asc2.matmul.
* Memory hierarchy: GM -> L0A (left) / L0B (right) -> MAC (cube) -> GM.
* Result tile dtype is always float32 (cube accumulator).
"""

import logging
import argparse
import torch

import asc
import asc.runtime.config as config
import asc2

logging.basicConfig(level=logging.INFO)


@asc2.jit(always_compile=True)
def matmul_kernel(a_ptr: asc.GlobalAddress, b_ptr: asc.GlobalAddress, c_ptr: asc.GlobalAddress,
                  a_shape: asc.ConstExpr, b_shape: asc.ConstExpr, c_shape: asc.ConstExpr,
                  m_tile: asc.ConstExpr[int], m_tiles_per_block: asc.ConstExpr[int],
                  n_tile: asc.ConstExpr[int], n_tiles_per_block: asc.ConstExpr[int]):
    a_gm = asc2.tensor(a_ptr, a_shape)
    b_gm = asc2.tensor(b_ptr, b_shape)
    c_gm = asc2.tensor(c_ptr, c_shape)
    block_id = asc2.block_idx()
    m_elems_per_block = m_tile * m_tiles_per_block
    m_base_off = (m_elems_per_block * block_id) % a_shape[0]
    n_base_off = ((m_elems_per_block * block_id) // a_shape[0]) * (n_tile * n_tiles_per_block)
    # Plain Python `range` over `asc.ConstExpr[int]` trip counts: both loops are
    # fully traced/unrolled at JIT time, so the PR 190 `asc2.range(unroll_factor=2)`
    # default does not apply -- wrapping these would emit a runtime ForOp instead.
    for j in range(n_tiles_per_block):
        b_offset = n_base_off + j * n_tile
        b_j = asc2.load(b_gm, [b_shape[0], n_tile], offsets=[0, b_offset], location=asc2.TileLocation.L0B)
        for i in range(m_tiles_per_block):
            a_offset = m_base_off + i * m_tile
            a_i = asc2.load(a_gm, [m_tile, a_shape[1]], offsets=[a_offset, 0], location=asc2.TileLocation.L0A)
            c_ij = a_i @ b_j
            asc2.store(c_ij, c_gm, offsets=[a_offset, b_offset])


def matmul_launch(a: torch.Tensor, b: torch.Tensor, core_num: int, m_tile: int,
                  n_tile: int, m_tiles_per_block: int, n_tiles_per_block: int) -> torch.Tensor:
    c = torch.zeros((a.shape[0], b.shape[1]), dtype=torch.float32)
    matmul_kernel[core_num](a, b, c, a.shape, b.shape, c.shape,
                            m_tile, m_tiles_per_block, n_tile, n_tiles_per_block)
    return c


def run_kernel(backend: config.Backend, platform: config.Platform):
    config.set_platform(backend, platform)

    test_cases = [
        # (m,  k,  n,  core_num, m_tile, n_tile, m_tiles_per_block, n_tiles_per_block)
        (16, 16, 16, 1, 16, 16, 1, 1),
        (32, 16, 32, 2, 16, 16, 1, 2),
    ]
    dtype = torch.float16
    torch.manual_seed(2026)

    for m, k, n, core_num, m_tile, n_tile, mtpb, ntpb in test_cases:
        a = torch.rand((m, k), dtype=dtype)
        b = torch.rand((k, n), dtype=dtype)
        c = matmul_launch(a, b, core_num, m_tile, n_tile, mtpb, ntpb)
        c_ref = a.to(torch.float32) @ b.to(torch.float32)
        torch.testing.assert_close(c, c_ref, atol=1e-2, rtol=1e-2)
        logging.info(f"[PASS] Kernel output verified for shape ({m},{k})x({k},{n}), core_num={core_num}.")


def test_matmul_f16(backend: config.Backend, platform: config.Platform):
    """pytest entry point."""
    run_kernel(backend, platform)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", type=str, default="Model", help="backend: Model or NPU")
    parser.add_argument("-v", type=str, default="Ascend950PR_9599",
                        help="platform/SoC version (matmul requires Ascend950PR_9599)")
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
