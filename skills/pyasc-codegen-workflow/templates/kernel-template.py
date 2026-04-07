#!/usr/bin/env python3.10
"""
pyasc kernel template: {kernel_name}

Operation: {describe operation}
Usage:
    python3.10 kernel.py -r Model -v Ascend910B1   # Run with simulator
    python3.10 kernel.py -r NPU                    # Run with NPU hardware
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
    """Pick (core_num, tile_num) that keep tile_length >= MIN_TILE_LENGTH.

    DMA transfers require a minimum data length per tile.  For small
    tensors the default tile_num=8 / core_num=8 can make tile_length
    too small.  This helper reduces core count and tile count until
    the constraint is satisfied.
    """
    for cores in (8, 4, 2, 1):
        if total_length % cores != 0:
            continue
        block = total_length // cores
        for tiles in (8, 4, 2, 1):
            if block % (tiles * BUFFER_NUM) != 0:
                continue
            tile_len = block // tiles // BUFFER_NUM
            if tile_len >= MIN_TILE_LENGTH:
                return cores, tiles
    return 1, 1


@asc.jit
def kernel_func(x: asc.GlobalAddress, y: asc.GlobalAddress, z: asc.GlobalAddress,
                block_length: int, tile_num: asc.ConstExpr[int]):
    """Replace this with your kernel implementation."""
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
    z_local = asc.LocalTensor(data_type, asc.TPosition.VECOUT, buffer_size + buffer_size, tile_length * BUFFER_NUM)

    for i in range(tile_num * BUFFER_NUM):
        buf_id = i % BUFFER_NUM

        asc.data_copy(x_local[buf_id * tile_length:], x_gm[i * tile_length:], tile_length)
        asc.data_copy(y_local[buf_id * tile_length:], y_gm[i * tile_length:], tile_length)

        asc.set_flag(asc.HardEvent.MTE2_V, buf_id)
        asc.wait_flag(asc.HardEvent.MTE2_V, buf_id)

        # TODO: Replace with your operation
        asc.add(z_local[buf_id * tile_length:], x_local[buf_id * tile_length:], y_local[buf_id * tile_length:],
                tile_length)

        asc.set_flag(asc.HardEvent.V_MTE3, buf_id)
        asc.wait_flag(asc.HardEvent.V_MTE3, buf_id)

        asc.data_copy(z_gm[i * tile_length:], z_local[buf_id * tile_length:], tile_length)

        asc.set_flag(asc.HardEvent.MTE3_MTE2, buf_id)
        asc.wait_flag(asc.HardEvent.MTE3_MTE2, buf_id)


def kernel_launch(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    z = torch.zeros_like(x)
    total_length = z.numel()
    core_num, tile_num = _compute_tiling(total_length)
    block_length = total_length // core_num
    kernel_func[core_num, rt.current_stream()](x, y, z, block_length, tile_num)
    return z


def run_kernel(backend: config.Backend, platform: config.Platform):
    config.set_platform(backend, platform)
    device = "npu" if config.Backend(backend) == config.Backend.NPU else "cpu"
    # TODO: Update shapes, dtype, and expected result for your operation
    test_shapes = [[4, 2048], [32, 4096]]
    for shape in test_shapes:
        x = torch.rand(shape, dtype=torch.float32, device=device)
        y = torch.rand(shape, dtype=torch.float32, device=device)
        z = kernel_launch(x, y)
        assert torch.allclose(z, x + y), f"Output mismatch for shape {shape}! Max diff: {(z - (x + y)).abs().max()}"
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
