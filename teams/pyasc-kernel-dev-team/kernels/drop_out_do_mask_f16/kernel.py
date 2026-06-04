#!/usr/bin/env python3.11
"""
pyasc kernel: drop_out_do_mask_f16

Operation: DropoutDoMask apply-phase (float16):

    out = data * mask * (1 / keep_prob)

where ``mask`` is a per-element {0., 1.} keep-mask and ``keep_prob`` is the
keep probability. This is the elementwise apply phase of CANN's DropoutDoMask
(matches aclnnDropoutDoMask's out = self * mask * scale). It runs entirely on
the AIV vector pipeline (vector-only).

Note on comparability: the canonical aclnnDropoutDoMask reference consumes a
*bit-packed* uint8 mask (1 bit/element) and unpacks it on-chip; this generated
kernel consumes a dense float16 keep-mask. The dominant cost on both sides is
the per-element multiply + scale over all N elements; the reference's bit
unpack is a small fixed addend. The shape/dtype/element-count are pinned to the
same comparability contract.

Usage:
    python3.11 kernel.py -r Model -v Ascend950PR_9599
    pytest kernel.py --backend Model --platform Ascend950PR_9599

Alignment requirement: element count must be a multiple of
TILE_SIZE * CORE_NUM = 32768 (aligned_only, matching the abs/add cells).
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
def drop_out_do_mask_kernel(data_ptr: asc.GlobalAddress, mask_ptr: asc.GlobalAddress,
                            out_ptr: asc.GlobalAddress, size: int,
                            tile_size: asc.ConstExpr[int], tile_per_block: asc.ConstExpr[int],
                            scale: asc.ConstExpr[float]):
    data_gm = asc2.tensor(data_ptr, [size])
    mask_gm = asc2.tensor(mask_ptr, [size])
    out_gm = asc2.tensor(out_ptr, [size])
    base_offset = asc2.block_idx() * tile_size * tile_per_block
    for i in asc2.range(tile_per_block, unroll_factor=2, parallel=True):
        off = base_offset + i * tile_size
        data_t = asc2.load(data_gm, [tile_size], offsets=[off])
        mask_t = asc2.load(mask_gm, [tile_size], offsets=[off])
        out = (data_t * mask_t) * scale
        asc2.store(out, out_gm, offsets=[off])


def drop_out_do_mask_launch(data: np.ndarray, mask: np.ndarray,
                            keep_prob: float = 0.5) -> np.ndarray:
    shape = data.shape
    data_f = np.ascontiguousarray(data.reshape(-1))
    mask_f = np.ascontiguousarray(mask.reshape(-1)).astype(data.dtype)
    size = data_f.size
    if size % ALIGNMENT != 0:
        raise ValueError(f"drop_out_do_mask is aligned_only; size {size} not a multiple of {ALIGNMENT}")
    scale = 1.0 / keep_prob
    out_f = np.empty_like(data_f)
    num_tiles = asc.ceildiv(size, TILE_SIZE)
    drop_out_do_mask_kernel[CORE_NUM](data_f, mask_f, out_f, size, TILE_SIZE,
                                      asc.ceildiv(num_tiles, CORE_NUM), scale)
    return out_f.reshape(shape)


def run_kernel(backend: config.Backend, platform: config.Platform):
    config.set_platform(backend, platform)
    test_shapes = [(16, 2048), (32, 4096)]
    rng = np.random.default_rng(seed=2026)
    keep_prob = 0.5
    for shape in test_shapes:
        data = (rng.random(shape, dtype=np.float32) * 2 - 1).astype(np.float16)
        mask = (rng.random(shape, dtype=np.float32) < keep_prob).astype(np.float16)
        out = drop_out_do_mask_launch(data, mask, keep_prob)
        expected = (data.astype(np.float32) * mask.astype(np.float32) * (1.0 / keep_prob)).astype(np.float16)
        np.testing.assert_allclose(out.astype(np.float32), expected.astype(np.float32), atol=1e-2, rtol=1e-2)
        logging.info(f"[PASS] Kernel output verified for shape {shape}.")


def test_drop_out_do_mask_f16(backend: config.Backend, platform: config.Platform):
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
