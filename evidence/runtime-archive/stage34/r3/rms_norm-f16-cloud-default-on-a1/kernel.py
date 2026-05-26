#!/usr/bin/env python3.10
"""
pyasc kernel: RMSNorm

Operation: y = x * gamma * rsqrt(mean(x^2, axis=-1, keepdims=True) + eps)

This kernel implements RMSNorm with a host-side dispatcher that selects between
two @asc2.jit kernels based on the runtime shape:
1. full_row kernel: when the row fits in UB (~64KB) and num_cols is aligned
2. split_d kernel: when the row exceeds UB, streaming along D in 64-element tiles

Usage:
    python3.10 kernel.py -r Model -v Ascend950PR_9599   # Run with simulator
    python3.10 kernel.py -r NPU                         # Run with NPU hardware
    pytest kernel.py --backend Model --platform Ascend950PR_9599
"""

import logging
import argparse
import math

import torch
import numpy as np

import asc
import asc.runtime.config as config
import asc2

logging.basicConfig(level=logging.INFO)

UB_BUDGET_BYTES = 64 * 1024
TILE_COLS = 64
CORE_NUM = 8


@asc2.jit(always_compile=True)
def rms_norm_full_row_kernel(
    x_ptr: asc.GlobalAddress,
    gamma_ptr: asc.GlobalAddress,
    out_ptr: asc.GlobalAddress,
    num_rows: int,
    num_cols: asc.ConstExpr[int],
    epsilon: asc.ConstExpr[float],
):
    """
    RMSNorm kernel for rows that fit entirely in UB.
    
    This kernel processes one full row at a time, casting to float32 for
    the sum of squares computation to maintain precision.
    
    Args:
        x_ptr: Input tensor [num_rows, num_cols]
        gamma_ptr: Per-column scaling factor [num_cols]
        out_ptr: Output tensor [num_rows, num_cols]
        num_rows: Number of rows (runtime int)
        num_cols: Number of columns (compile-time ConstExpr for tile shape)
        epsilon: Small constant for numerical stability (compile-time ConstExpr)
    """
    x_gm = asc2.tensor(x_ptr, [num_rows, num_cols])
    gamma_gm_2d = asc2.tensor(gamma_ptr, [1, num_cols])
    out_gm = asc2.tensor(out_ptr, [num_rows, num_cols])
    
    for row in asc2.range(
        asc2.block_idx(), num_rows, asc2.block_num(),
        unroll_factor=2, parallel=True
    ):
        x_row = asc2.load(x_gm, [1, num_cols], offsets=[row, 0])
        x_row_f32 = x_row.to(asc.float32)
        
        sum_sq = asc2.reduce_sum(x_row_f32 * x_row_f32)
        inv_rms = 1.0 / asc2.sqrt(sum_sq / num_cols + epsilon)
        
        gamma_row = asc2.load(gamma_gm_2d, [1, num_cols], offsets=[0, 0])
        gamma_row_f32 = gamma_row.to(asc.float32)
        
        out_f32 = x_row_f32 * gamma_row_f32 * inv_rms
        asc2.store(out_f32.to(x_row.dtype), out_gm, offsets=[row, 0])


@asc2.jit(always_compile=True)
def rms_norm_split_d_kernel(
    x_ptr: asc.GlobalAddress,
    gamma_ptr: asc.GlobalAddress,
    out_ptr: asc.GlobalAddress,
    num_rows: int,
    num_cols: int,
    padded_cols: int,
    num_tiles: int,
    tile_cols: asc.ConstExpr[int],
    epsilon: asc.ConstExpr[float],
):
    """
    RMSNorm kernel for large rows that must be streamed in tiles.
    
    This kernel processes each row in multiple tiles along the D dimension,
    accumulating the sum of squares across all tiles before computing the
    normalization. Host-side padding ensures all tiles are full.
    
    Args:
        x_ptr: Padded input tensor [num_rows, padded_cols]
        gamma_ptr: Padded gamma [1, padded_cols]
        out_ptr: Padded output tensor [num_rows, padded_cols]
        num_rows: Number of rows (runtime int)
        num_cols: Real number of columns before padding (runtime int)
        padded_cols: Padded columns = ceil(num_cols / tile_cols) * tile_cols (runtime int)
        num_tiles: Number of tiles per row = padded_cols // tile_cols (runtime int)
        tile_cols: Tile size along D (compile-time ConstExpr)
        epsilon: Small constant for numerical stability (compile-time ConstExpr)
    """
    x_gm = asc2.tensor(x_ptr, [num_rows, padded_cols])
    gamma_gm_2d = asc2.tensor(gamma_ptr, [1, padded_cols])
    out_gm = asc2.tensor(out_ptr, [num_rows, padded_cols])
    
    for row in asc2.range(
        asc2.block_idx(), num_rows, asc2.block_num(),
        unroll_factor=2, parallel=True
    ):
        zero_seed = asc2.full([1, tile_cols], 0.0, dtype=asc.float32)
        sum_sq = asc2.reduce_sum(zero_seed)
        
        for tile_id in asc2.range(num_tiles, unroll_factor=2):
            col = tile_id * tile_cols
            x = asc2.load(x_gm, [1, tile_cols], offsets=[row, col])
            x_f32 = x.to(asc.float32)
            sum_sq = sum_sq + asc2.reduce_sum(x_f32 * x_f32)
        
        inv_rms = 1.0 / asc2.sqrt(sum_sq / num_cols + epsilon)
        
        for tile_id in asc2.range(num_tiles, unroll_factor=2, parallel=True):
            col = tile_id * tile_cols
            x = asc2.load(x_gm, [1, tile_cols], offsets=[row, col])
            gamma = asc2.load(gamma_gm_2d, [1, tile_cols], offsets=[0, col])
            x_f32 = x.to(asc.float32)
            gamma_f32 = gamma.to(asc.float32)
            out_f32 = x_f32 * gamma_f32 * inv_rms
            asc2.store(out_f32.to(x.dtype), out_gm, offsets=[row, col])


def _full_row_launch(
    x: torch.Tensor,
    gamma: torch.Tensor,
    eps: float,
    core_num: int
) -> torch.Tensor:
    """Launch full_row kernel for aligned rows that fit in UB."""
    num_rows, num_cols = x.shape
    out = torch.empty_like(x)
    
    rms_norm_full_row_kernel[core_num](
        x, gamma, out,
        num_rows, num_cols, eps
    )
    
    return out


def _split_d_launch(
    x: torch.Tensor,
    gamma: torch.Tensor,
    eps: float,
    core_num: int
) -> torch.Tensor:
    """Launch split_d kernel for large rows with host-side padding."""
    num_rows, num_cols = x.shape
    padded_cols = ((num_cols + TILE_COLS - 1) // TILE_COLS) * TILE_COLS
    num_tiles = padded_cols // TILE_COLS
    
    x_padded = torch.zeros((num_rows, padded_cols), dtype=x.dtype)
    x_padded[:, :num_cols] = x
    
    gamma_padded = torch.zeros((padded_cols,), dtype=gamma.dtype)
    gamma_padded[:num_cols] = gamma
    
    out_padded = torch.zeros((num_rows, padded_cols), dtype=x.dtype)
    
    rms_norm_split_d_kernel[core_num](
        x_padded, gamma_padded, out_padded,
        num_rows, num_cols, padded_cols,
        num_tiles, TILE_COLS, eps
    )
    
    return out_padded[:, :num_cols].clone()


def rms_norm_launch(
    x: torch.Tensor,
    gamma: torch.Tensor,
    eps: float = 1e-5,
    core_num: int = CORE_NUM
) -> torch.Tensor:
    """
    Host-side dispatcher for RMSNorm.
    
    Selects between full_row and split_d kernels based on row size:
    - full_row: when row fits in UB budget AND num_cols is aligned to 8
    - split_d: otherwise (streams along D with host-side padding)
    
    Args:
        x: Input tensor [num_rows, num_cols], dtype float16
        gamma: Per-column scaling factor [num_cols], dtype float16
        eps: Small constant for numerical stability
        core_num: Number of cores to use
        
    Returns:
        Normalized output tensor [num_rows, num_cols], dtype float16
    """
    num_rows, num_cols = x.shape
    row_bytes = num_cols * x.element_size()
    
    if row_bytes <= UB_BUDGET_BYTES and num_cols % 8 == 0:
        return _full_row_launch(x, gamma, eps, core_num)
    return _split_d_launch(x, gamma, eps, core_num)


def torch_rms_norm(x: torch.Tensor, gamma: torch.Tensor, eps: float) -> torch.Tensor:
    """Reference implementation using torch operations."""
    x32 = x.to(torch.float32)
    mean_sq = torch.mean(x32 * x32, dim=-1, keepdim=True)
    return (x32 * torch.rsqrt(mean_sq + eps) * gamma.to(torch.float32)).to(x.dtype)


def run_kernel(backend: config.Backend, platform: config.Platform):
    """Run RMSNorm kernel tests for both full_row and split_d regimes."""
    config.set_platform(backend, platform)
    
    rng = torch.Generator().manual_seed(2026)
    
    x_256 = (torch.rand((8, 256), generator=rng, dtype=torch.float32) * 10 - 5).to(torch.float16)
    gamma_256 = (torch.rand((256,), generator=rng, dtype=torch.float32) * 2 - 1).to(torch.float16)
    eps = 1e-5
    
    out_256 = rms_norm_launch(x_256, gamma_256, eps)
    expected_256 = torch_rms_norm(x_256, gamma_256, eps)
    
    np.testing.assert_allclose(
        out_256.to(torch.float32).numpy(),
        expected_256.to(torch.float32).numpy(),
        atol=2e-2,
        rtol=2e-2
    )
    logging.info("[PASS] full_row kernel verified for shape (8, 256)")
    
    x_1055 = (torch.rand((8, 1055), generator=rng, dtype=torch.float32) * 10 - 5).to(torch.float16)
    gamma_1055 = (torch.rand((1055,), generator=rng, dtype=torch.float32) * 2 - 1).to(torch.float16)
    
    out_1055 = rms_norm_launch(x_1055, gamma_1055, eps)
    expected_1055 = torch_rms_norm(x_1055, gamma_1055, eps)
    
    np.testing.assert_allclose(
        out_1055.to(torch.float32).numpy(),
        expected_1055.to(torch.float32).numpy(),
        atol=5e-2,
        rtol=5e-2
    )
    logging.info("[PASS] split_d kernel verified for shape (8, 1055)")


def test_kernel(backend: config.Backend, platform: config.Platform):
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
    
    logging.info(f"[INFO] Running RMSNorm kernel with backend={backend}, platform={platform}")
    run_kernel(backend, platform)
    logging.info("[INFO] RMSNorm kernel run complete.")