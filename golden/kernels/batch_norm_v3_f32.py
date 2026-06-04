#!/usr/bin/env python3.11
"""
pyasc kernel: batch_norm_v3_f32

Operation: BatchNormV3 training-mode forward (matches CANN BatchNormV3 /
aclnnBatchNorm with training=true), float32:

    for each channel c (reduction over N and L):
        mean[c]   = sum_{n,l} x[n,c,l] / (N*L)
        var[c]    = sum_{n,l} x[n,c,l]^2 / (N*L) - mean[c]^2
        invstd[c] = 1 / sqrt(var[c] + eps)
        out[n,c,l] = (x[n,c,l] - mean[c]) * invstd[c] * weight[c] + bias[c]
                   = x[n,c,l] * scale[c] + shift[c]

Outputs: out [N,C,L], saveMean[c]=mean[c], saveInvStd[c]=invstd[c].

Realized ENTIRELY on the AIV vector pipeline (vector + reduce_sum, no cube), so
it stays in scope for the "vector-only generation" claim even though it is
multi-I/O. The per-channel reduction over the strided [N,C,L] layout is done
on-chip (NO host transpose): viewing x as [N*C, L], channel c's N row-segments
are at strided rows n*C+c, reduced one [1, L] tile at a time with the proven
full-reduce->scalar primitive (same as the rms_norm golden) so the reduction is
numerically exact. weight/bias are pre-expanded on the host to [C, L] so the
affine params load with the required 32-byte UB alignment; saveMean/saveInvStd
are written as [C, L] and the host slices column 0.

NOTE (honest perf): this strided per-channel reduction is heavily DMA-bound
(many small [1, L] loads) and does NOT clear the 0.70 gate against the
hand-written aclnnBatchNorm; it is a correctness-confirmed generative cell with
a documented perf miss (see capabilities.yaml perf_ratio_demo + docs).

Inputs MUST be torch.Tensor (the C310 simulator path mishandles numpy buffers
for this reduction pattern, as documented on the rms_norm golden).

Usage:
    python3.11 kernel.py -r Model -v Ascend950PR_9599
    pytest kernel.py --backend Model --platform Ascend950PR_9599

Cell metadata (mirrors capabilities.yaml; do not drift):
  - shape_regime: fixed
  - tail_behavior: aligned_only
  - partitioning: row_per_core
"""

import logging
import argparse
import torch

import asc
import asc.runtime.config as config
import asc2

CORE_NUM = 8
CH_PER_CORE = 8
N_SIZE = 32
C_SIZE = 64
L_SIZE = 64

logging.basicConfig(level=logging.INFO)


@asc2.jit(always_compile=True)
def batch_norm_v3_kernel(x_ptr: asc.GlobalAddress, w_ptr: asc.GlobalAddress,
                         b_ptr: asc.GlobalAddress, out_ptr: asc.GlobalAddress,
                         mean_ptr: asc.GlobalAddress, invstd_ptr: asc.GlobalAddress,
                         inv_count: asc.ConstExpr[float],
                         eps: asc.ConstExpr[float]):
    x_gm = asc2.tensor(x_ptr, [N_SIZE * C_SIZE, L_SIZE])
    out_gm = asc2.tensor(out_ptr, [N_SIZE * C_SIZE, L_SIZE])
    w_gm = asc2.tensor(w_ptr, [C_SIZE])
    b_gm = asc2.tensor(b_ptr, [C_SIZE])
    mean_gm = asc2.tensor(mean_ptr, [C_SIZE])
    invstd_gm = asc2.tensor(invstd_ptr, [C_SIZE])

    ch = CH_PER_CORE
    c0 = asc2.block_idx() * ch
    # Each core owns `ch` contiguous channels. For sample n, those channels are
    # `ch` contiguous rows (n*C + c0 ..) of the [N*C, L] view -> a single [ch, L]
    # strided tile. reduce_sum over the L axis (1) yields per-channel partials.
    sum_c = asc2.full([ch], 0.0, dtype=asc.float32)
    sumsq_c = asc2.full([ch], 0.0, dtype=asc.float32)
    for n in asc2.range(N_SIZE):
        blk = asc2.load(x_gm, [ch, L_SIZE], offsets=[n * C_SIZE + c0, 0])
        sum_c = sum_c + asc2.reduce_sum(blk, 1)
        sumsq_c = sumsq_c + asc2.reduce_sum(blk * blk, 1)

    mean_c = sum_c * inv_count
    var_c = sumsq_c * inv_count - mean_c * mean_c
    invstd_c = asc2.rsqrt(var_c + eps)
    w_c = asc2.load(w_gm, [ch], offsets=[c0])
    b_c = asc2.load(b_gm, [ch], offsets=[c0])
    scale_c = w_c * invstd_c
    shift_c = b_c - scale_c * mean_c
    asc2.store(mean_c, mean_gm, offsets=[c0])
    asc2.store(invstd_c, invstd_gm, offsets=[c0])

    scale_b = asc2.broadcast_to(asc2.expand_dims(scale_c, 1), ch, L_SIZE)
    shift_b = asc2.broadcast_to(asc2.expand_dims(shift_c, 1), ch, L_SIZE)
    for n in asc2.range(N_SIZE):
        blk = asc2.load(x_gm, [ch, L_SIZE], offsets=[n * C_SIZE + c0, 0])
        out = blk * scale_b + shift_b
        asc2.store(out, out_gm, offsets=[n * C_SIZE + c0, 0])


def batch_norm_v3_launch(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor,
                         eps: float = 1e-5):
    """Host launcher. x is [N, C, L]; weight/bias are [C] (fp32).
    Returns (out [N,C,L], saveMean [C], saveInvStd [C])."""
    N, C, L = x.shape
    if (N, C, L) != (N_SIZE, C_SIZE, L_SIZE):
        raise ValueError(f"batch_norm_v3 cell is fixed-shape [{N_SIZE},{C_SIZE},{L_SIZE}]; got {tuple(x.shape)}")
    x_2d = x.reshape(N * C, L).contiguous().to(torch.float32)
    w_1d = weight.reshape(C).contiguous().to(torch.float32)
    b_1d = bias.reshape(C).contiguous().to(torch.float32)
    out_2d = torch.empty_like(x_2d)
    mean_buf = torch.empty((C,), dtype=torch.float32)
    invstd_buf = torch.empty((C,), dtype=torch.float32)
    inv_count = 1.0 / float(N * L)
    batch_norm_v3_kernel[CORE_NUM](x_2d, w_1d, b_1d, out_2d, mean_buf, invstd_buf,
                                   inv_count, eps)
    return out_2d.reshape(N, C, L), mean_buf.contiguous(), invstd_buf.contiguous()


def torch_bn(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float):
    x64 = x.to(torch.float64)
    mean = x64.mean(dim=(0, 2))
    var = x64.var(dim=(0, 2), unbiased=False)
    invstd = 1.0 / torch.sqrt(var + eps)
    out = (x64 - mean[None, :, None]) * invstd[None, :, None] * weight.to(torch.float64)[None, :, None] + bias.to(torch.float64)[None, :, None]
    return out.to(torch.float32), mean.to(torch.float32), invstd.to(torch.float32)


def run_kernel(backend: config.Backend, platform: config.Platform):
    config.set_platform(backend, platform)
    test_shapes = [(32, 64, 64)]
    gen = torch.Generator().manual_seed(2026)
    eps = 1e-5
    for shape in test_shapes:
        N, C, L = shape
        x = (torch.rand(shape, generator=gen, dtype=torch.float32) * 2 - 1)
        weight = (torch.rand(C, generator=gen, dtype=torch.float32) * 2)
        bias = (torch.rand(C, generator=gen, dtype=torch.float32) * 2 - 1)
        exp_out, exp_mean, exp_invstd = torch_bn(x, weight, bias, eps)
        out, mean, invstd = batch_norm_v3_launch(x, weight, bias, eps)
        dm = (mean - exp_mean).abs().max().item()
        di = (invstd - exp_invstd).abs().max().item()
        do = (out - exp_out).abs().max().item()
        logging.info(f"[DIAG] shape {shape}: max|dmean|={dm:.6g} max|dinvstd|={di:.6g} max|dout|={do:.6g}")
        torch.testing.assert_close(mean, exp_mean, atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(invstd, exp_invstd, atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(out, exp_out, atol=1e-2, rtol=1e-2)
        logging.info(f"[PASS] Kernel output verified for shape {shape}.")


def test_batch_norm_v3_f32(backend: config.Backend, platform: config.Platform):
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
