#!/usr/bin/env python3.11
"""RMSNorm float16 operator with full_row and split_d kernels.

Implements: y = x * gamma * rsqrt(mean(x^2, axis=-1, keepdims=True) + eps)

Two @asc2.jit kernels with host-side dispatcher:
  - full_row: num_cols is ConstExpr, row fits in UB (64KB)
  - split_d: both dims are runtime int, streams along D in tile_cols=64 tiles

Test shapes: (8, 256) -> full_row, (8, 1055) -> split_d
Platform: Ascend950PR_9599
"""

import logging
import argparse
import torch

import asc
import asc.runtime.config as config
import asc2

CORE_NUM = 8
TILE_COLS = 64
EPS = 1e-5
UB_BUDGET_BYTES = 64 * 1024

logging.basicConfig(level=logging.INFO)


@asc2.jit(always_compile=True)
def rms_norm_full_row_kernel(
    x_ptr: asc.GlobalAddress,
    gamma_ptr: asc.GlobalAddress,
    out_ptr: asc.GlobalAddress,
    num_rows: int,
    num_cols: asc.ConstExpr[int],
    epsilon: asc.ConstExpr[float],
):
    x_gm = asc2.tensor(x_ptr, [num_rows, num_cols])
    gamma_gm_2d = asc2.tensor(gamma_ptr, [1, num_cols])
    out_gm = asc2.tensor(out_ptr, [num_rows, num_cols])

    for row in asc2.range(
        asc2.block_idx(), num_rows, asc2.block_num(), unroll_factor=2, parallel=True
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
    x_gm = asc2.tensor(x_ptr, [num_rows, padded_cols])
    gamma_gm_2d = asc2.tensor(gamma_ptr, [1, padded_cols])
    out_gm = asc2.tensor(out_ptr, [num_rows, padded_cols])

    for row in asc2.range(
        asc2.block_idx(), num_rows, asc2.block_num(), unroll_factor=2, parallel=True
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
    x: torch.Tensor, gamma: torch.Tensor, eps: float, core_num: int
) -> torch.Tensor:
    num_rows, num_cols = x.shape
    out = torch.empty_like(x)
    rms_norm_full_row_kernel[core_num](x, gamma, out, num_rows, num_cols, eps)
    return out


def _split_d_launch(
    x: torch.Tensor,
    gamma: torch.Tensor,
    eps: float,
    core_num: int,
    tile_cols: int = TILE_COLS,
) -> torch.Tensor:
    num_rows, num_cols = x.shape
    padded_cols = ((num_cols + tile_cols - 1) // tile_cols) * tile_cols

    if padded_cols == num_cols:
        x_padded = x
        gamma_padded = gamma
    else:
        x_padded = torch.zeros((num_rows, padded_cols), dtype=x.dtype)
        x_padded[:, :num_cols] = x
        gamma_padded = torch.zeros((padded_cols,), dtype=gamma.dtype)
        gamma_padded[:num_cols] = gamma

    out_padded = torch.zeros((num_rows, padded_cols), dtype=x.dtype)
    num_tiles = padded_cols // tile_cols
    rms_norm_split_d_kernel[core_num](
        x_padded, gamma_padded, out_padded, num_rows, num_cols, padded_cols, num_tiles, tile_cols, eps
    )
    return out_padded[:, :num_cols].clone()


def rms_norm_launch(
    x: torch.Tensor, gamma: torch.Tensor, eps: float = EPS, core_num: int = CORE_NUM
) -> torch.Tensor:
    num_rows, num_cols = x.shape
    dtype_bytes = x.element_size()
    row_bytes = num_cols * dtype_bytes
    if row_bytes <= UB_BUDGET_BYTES and num_cols % 8 == 0:
        return _full_row_launch(x, gamma, eps, core_num)
    return _split_d_launch(x, gamma, eps, core_num)


def torch_rms_norm(x: torch.Tensor, gamma: torch.Tensor, eps: float) -> torch.Tensor:
    x32 = x.to(torch.float32)
    mean_sq = torch.mean(x32 * x32, dim=-1, keepdim=True)
    return (x32 * torch.rsqrt(mean_sq + eps) * gamma.to(torch.float32)).to(x.dtype)


def run_kernel(backend: config.Backend, platform: config.Platform):
    config.set_platform(backend, platform)

    rng = torch.Generator().manual_seed(2026)

    num_rows, num_cols = 8, 256
    x = torch.randn((num_rows, num_cols), generator=rng, dtype=torch.float32).to(torch.float16)
    gamma = (torch.randn((num_cols,), generator=rng, dtype=torch.float32) * 0.5 + 1.0).to(torch.float16)
    out = rms_norm_launch(x, gamma, EPS)
    expected = torch_rms_norm(x, gamma, EPS)
    torch.testing.assert_close(out, expected, atol=2e-2, rtol=2e-2)
    logging.info(f"[PASS] full_row branch shape ({num_rows}, {num_cols})")

    num_rows, num_cols = 8, 1055
    x = torch.randn((num_rows, num_cols), generator=rng, dtype=torch.float32).to(torch.float16)
    gamma = (torch.randn((num_cols,), generator=rng, dtype=torch.float32) * 0.5 + 1.0).to(torch.float16)
    out = rms_norm_launch(x, gamma, EPS)
    expected = torch_rms_norm(x, gamma, EPS)
    torch.testing.assert_close(out, expected, atol=5e-2, rtol=5e-2)
    logging.info(f"[PASS] split_d branch shape ({num_rows}, {num_cols})")


def test_rms_norm_f16(backend: config.Backend, platform: config.Platform):
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