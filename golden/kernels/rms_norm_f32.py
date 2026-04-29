#!/usr/bin/env python3.10
"""
Golden reference: rms_norm_f32 kernel (asc2 API).

Truly runtime-dynamic `num_cols` RMSNorm via single-vector-lane tile
streaming. Both `num_rows` and `num_cols` are runtime `int`; the only
compile-time constant in the inner loop is `tile_cols = 64` (= one
hardware vector lane on Ascend910B1, 64 floats = 256 bytes), which keeps
all tile arithmetic within a single SIMD lane and avoids the wide-tile
fill / scalar-broadcast bugs in the pinned MR-85 simulator build.

Why not the column-loop in `pyasc-fork/docs/e2e-rms-norm-column-loop-en.md`?
  Pure scalar `asc2.store(plain_value, gm, offsets=[r, c])` lowers to
  `SetValueOp`, which is dropped on even-indexed blocks in MR-85 multi-
  core launches. We verified this with a minimal 8-row probe: rows
  processed by blocks 0, 2, 4, 6 came back as zero. The streaming form
  here uses tile stores instead and works reliably across all cores.

Shape contract:
  Host pads x and gamma so the row length becomes
  `padded_cols = ceil(num_cols / tile_cols) * tile_cols`. Padded zeros
  do not contribute to `sum_sq`, so mean-square is normalized by the
  REAL `num_cols`. The output is sliced back to `[:num_cols]` before
  return. This satisfies the user-visible "dynamic string lengths"
  prompt: any `num_cols` works without recompile of the kernel.

To run the doc-equivalent (64, 100003) shape, change only CORE_NUM and
the `num_rows, num_cols` line in `run_kernel`. The kernel itself does
not need any code changes.
"""

import logging
import argparse
import numpy as np

import asc
import asc.runtime.config as config
import asc2

CORE_NUM = 8
TILE_COLS = 64
EPS = 1e-5

logging.basicConfig(level=logging.INFO)


@asc2.jit(always_compile=True)
def rms_norm_streaming_kernel(x_ptr: asc.GlobalAddress, gamma_ptr: asc.GlobalAddress,
                              out_ptr: asc.GlobalAddress,
                              num_rows: int, num_cols: int, padded_cols: int,
                              num_tiles: int,
                              tile_cols: asc.ConstExpr[int],
                              epsilon: asc.ConstExpr[float]):
    x_gm = asc2.tensor(x_ptr, [num_rows, padded_cols])
    gamma_gm_2d = asc2.tensor(gamma_ptr, [1, padded_cols])
    out_gm = asc2.tensor(out_ptr, [num_rows, padded_cols])

    for row in asc2.range(asc2.block_idx(), num_rows, asc2.block_num()):
        zero_seed = asc2.full([1, tile_cols], 0.0, dtype=asc.float32)
        sum_sq = asc2.reduce_sum(zero_seed)

        for tile_id in asc2.range(num_tiles):
            col = tile_id * tile_cols
            x = asc2.load(x_gm, [1, tile_cols], offsets=[row, col])
            sum_sq = sum_sq + asc2.reduce_sum(x * x)

        inv_rms = 1.0 / asc2.sqrt(sum_sq / num_cols + epsilon)

        for tile_id in asc2.range(num_tiles):
            col = tile_id * tile_cols
            x = asc2.load(x_gm, [1, tile_cols], offsets=[row, col])
            gamma = asc2.load(gamma_gm_2d, [1, tile_cols], offsets=[0, col])
            out = x * gamma * inv_rms
            asc2.store(out, out_gm, offsets=[row, col])


def rms_norm_launch(x: np.ndarray, gamma: np.ndarray,
                    eps: float = EPS,
                    tile_cols: int = TILE_COLS,
                    core_num: int = CORE_NUM) -> np.ndarray:
    num_rows, num_cols = x.shape
    padded_cols = ((num_cols + tile_cols - 1) // tile_cols) * tile_cols

    if padded_cols == num_cols:
        x_padded = x
        gamma_padded = gamma
    else:
        x_padded = np.zeros((num_rows, padded_cols), dtype=x.dtype)
        x_padded[:, :num_cols] = x
        gamma_padded = np.zeros((padded_cols,), dtype=gamma.dtype)
        gamma_padded[:num_cols] = gamma

    out_padded = np.zeros((num_rows, padded_cols), dtype=x.dtype)
    num_tiles = padded_cols // tile_cols
    rms_norm_streaming_kernel[core_num](x_padded, gamma_padded, out_padded,
                                        num_rows, num_cols, padded_cols,
                                        num_tiles, tile_cols, eps)
    return out_padded[:, :num_cols].copy()


def rms_norm_numpy(x: np.ndarray, gamma: np.ndarray, eps: float) -> np.ndarray:
    x32 = x.astype(np.float32)
    mean_sq = (x32 * x32).mean(axis=-1, keepdims=True)
    return (x32 / np.sqrt(mean_sq + eps) * gamma.astype(np.float32)).astype(x.dtype)


def run_kernel(backend: config.Backend, platform: config.Platform):
    config.set_platform(backend, platform)

    rng = np.random.default_rng(seed=2026)
    num_rows, num_cols = 8, 1055
    x = rng.standard_normal((num_rows, num_cols), dtype=np.float32)
    gamma = rng.standard_normal((num_cols,), dtype=np.float32)
    out = rms_norm_launch(x, gamma, EPS)
    expected = rms_norm_numpy(x, gamma, EPS)
    np.testing.assert_allclose(out, expected, atol=1e-4, rtol=1e-4)
    logging.info(
        f"[PASS] streaming rms_norm verified for shape "
        f"({num_rows}, {num_cols}) tile_cols={TILE_COLS}.")


def test_rms_norm_f32(backend: config.Backend, platform: config.Platform):
    """pytest entry point."""
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
