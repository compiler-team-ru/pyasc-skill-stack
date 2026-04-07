#!/usr/bin/env python3.10
"""
Golden reference: sub_f16 kernel
Element-wise subtraction for float16 tensors (z = x - y).
Verified on CANN 9.0.0 simulator with Ascend910B1 platform.
"""

import logging
import argparse
import torch
try:
    import torch_npu
except ModuleNotFoundError:
    pass

import asc
import asc.runtime.config as config
import asc.lib.runtime as rt

BUFFER_NUM = 2
MIN_TILE_LENGTH = 32

logging.basicConfig(level=logging.INFO)


def _compute_tiling(total_length: int) -> tuple:
    """Pick (core_num, tile_num) that keep tile_length >= MIN_TILE_LENGTH."""
    for cores in (8, 4, 2, 1):
        if total_length % cores != 0:
            continue
        block = total_length // cores
        for tiles in (8, 4, 2, 1):
            if block % (tiles * BUFFER_NUM) != 0:
                continue
            if block // tiles // BUFFER_NUM >= MIN_TILE_LENGTH:
                return cores, tiles
    return 1, 1


@asc.jit
def sub_kernel(x: asc.GlobalAddress, y: asc.GlobalAddress, z: asc.GlobalAddress,
               block_length: int, tile_num: asc.ConstExpr[int]):
    offset = asc.get_block_idx() * block_length
    x_gm = asc.GlobalTensor()
    y_gm = asc.GlobalTensor()
    z_gm = asc.GlobalTensor()
    x_gm.set_global_buffer(x + offset, block_length)
    y_gm.set_global_buffer(y + offset, block_length)
    z_gm.set_global_buffer(z + offset, block_length)

    tile_length = block_length // tile_num // BUFFER_NUM
    data_type = x.dtype
    buffer_size = tile_length * BUFFER_NUM * data_type.sizeof()

    x_local = asc.LocalTensor(data_type, asc.TPosition.VECIN, 0, tile_length * BUFFER_NUM)
    y_local = asc.LocalTensor(data_type, asc.TPosition.VECIN, buffer_size, tile_length * BUFFER_NUM)
    z_local = asc.LocalTensor(data_type, asc.TPosition.VECOUT, 2 * buffer_size, tile_length * BUFFER_NUM)

    for i in range(tile_num * BUFFER_NUM):
        buf_id = i % BUFFER_NUM

        asc.data_copy(x_local[buf_id * tile_length:], x_gm[i * tile_length:], tile_length)
        asc.data_copy(y_local[buf_id * tile_length:], y_gm[i * tile_length:], tile_length)

        asc.set_flag(asc.HardEvent.MTE2_V, buf_id)
        asc.wait_flag(asc.HardEvent.MTE2_V, buf_id)

        asc.sub(z_local[buf_id * tile_length:], x_local[buf_id * tile_length:], y_local[buf_id * tile_length:], tile_length)

        asc.set_flag(asc.HardEvent.V_MTE3, buf_id)
        asc.wait_flag(asc.HardEvent.V_MTE3, buf_id)

        asc.data_copy(z_gm[i * tile_length:], z_local[buf_id * tile_length:], tile_length)

        asc.set_flag(asc.HardEvent.MTE3_MTE2, buf_id)
        asc.wait_flag(asc.HardEvent.MTE3_MTE2, buf_id)


def sub_launch(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    z = torch.zeros_like(x)
    total_length = z.numel()
    core_num, tile_num = _compute_tiling(total_length)
    block_length = total_length // core_num
    sub_kernel[core_num, rt.current_stream()](x, y, z, block_length, tile_num)
    return z


def run_kernel(backend: config.Backend, platform: config.Platform):
    config.set_platform(backend, platform)
    device = "npu" if config.Backend(backend) == config.Backend.NPU else "cpu"

    test_shapes = [[1, 128], [4, 2048], [32, 4096]]

    for shape in test_shapes:
        x = torch.randn(shape, dtype=torch.float16, device=device)
        y = torch.randn(shape, dtype=torch.float16, device=device)
        z = sub_launch(x, y)
        expected = x - y
        assert torch.allclose(z, expected, atol=1e-3), f"Output mismatch for shape {shape}! Max diff: {(z - expected).abs().max()}"
        logging.info(f"[PASS] Kernel output verified for shape {shape}.")


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
