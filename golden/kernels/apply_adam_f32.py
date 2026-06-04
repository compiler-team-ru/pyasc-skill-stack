#!/usr/bin/env python3.11
"""
pyasc kernel: apply_adam_f32

Operation: In-place Adam optimizer update (matches CANN ApplyAdam / aclnnApplyAdam,
non-Nesterov), float32:

    alpha = lr * sqrt(1 - beta2_power) / (1 - beta1_power)
    m   <- m   + (1 - beta1) * (grad - m)
    v   <- v   + (1 - beta2) * (grad*grad - v)
    var <- var - alpha * m / (sqrt(v) + epsilon)

The six hyper-parameters are scalars; the per-element work runs entirely on the
AIV vector pipeline (vector-only). var / m / v are updated in place.

Usage:
    python3.11 kernel.py -r Model -v Ascend950PR_9599
    pytest kernel.py --backend Model --platform Ascend950PR_9599

Alignment requirement: element count must be a multiple of
TILE_SIZE * CORE_NUM = 32768 (aligned_only, matching the abs/add cells).

Cell metadata (mirrors capabilities.yaml; do not drift):
  - shape_regime: fixed
  - tail_behavior: aligned_only
  - partitioning: tile_per_core
"""

import logging
import argparse
import math
import numpy as np

import asc
import asc.runtime.config as config
import asc2

TILE_SIZE = 2048
CORE_NUM = 16
ALIGNMENT = TILE_SIZE * CORE_NUM  # 32768 elements

logging.basicConfig(level=logging.INFO)


@asc2.jit(always_compile=True)
def apply_adam_kernel(var_ptr: asc.GlobalAddress, m_ptr: asc.GlobalAddress,
                      v_ptr: asc.GlobalAddress, grad_ptr: asc.GlobalAddress,
                      size: int, tile_size: asc.ConstExpr[int],
                      tile_per_block: asc.ConstExpr[int],
                      one_minus_beta1: asc.ConstExpr[float],
                      one_minus_beta2: asc.ConstExpr[float],
                      alpha: asc.ConstExpr[float],
                      epsilon: asc.ConstExpr[float]):
    var_gm = asc2.tensor(var_ptr, [size])
    m_gm = asc2.tensor(m_ptr, [size])
    v_gm = asc2.tensor(v_ptr, [size])
    grad_gm = asc2.tensor(grad_ptr, [size])
    base_offset = asc2.block_idx() * tile_size * tile_per_block
    # 8 live tiles/iteration (4 loads + 4 temps); unroll=1 keeps UB within the
    # ~248KB budget at the 32x4096 measurement shape. NOTE: this op is
    # memory-bound (4 loads + 3 stores/element) and currently lands at ~0.46 of
    # the hand-written aclnnApplyAdam — an honest perf gap, not a correctness one.
    for i in asc2.range(tile_per_block, unroll_factor=1, parallel=True):
        off = base_offset + i * tile_size
        var_t = asc2.load(var_gm, [tile_size], offsets=[off])
        m_t = asc2.load(m_gm, [tile_size], offsets=[off])
        v_t = asc2.load(v_gm, [tile_size], offsets=[off])
        g_t = asc2.load(grad_gm, [tile_size], offsets=[off])

        m_new = m_t + (g_t - m_t) * one_minus_beta1
        v_new = v_t + (g_t * g_t - v_t) * one_minus_beta2
        denom = asc2.sqrt(v_new) + epsilon
        var_new = var_t - asc2.div(m_new * alpha, denom)

        asc2.store(m_new, m_gm, offsets=[off])
        asc2.store(v_new, v_gm, offsets=[off])
        asc2.store(var_new, var_gm, offsets=[off])


def apply_adam_launch(var: np.ndarray, m: np.ndarray, v: np.ndarray, grad: np.ndarray,
                      beta1_power: float = 0.9, beta2_power: float = 0.999,
                      lr: float = 0.001, beta1: float = 0.9, beta2: float = 0.999,
                      epsilon: float = 1e-8):
    """Host launcher; updates var/m/v in place and returns (var, m, v)."""
    shape = var.shape
    var_f = np.ascontiguousarray(var.reshape(-1))
    m_f = np.ascontiguousarray(m.reshape(-1))
    v_f = np.ascontiguousarray(v.reshape(-1))
    grad_f = np.ascontiguousarray(grad.reshape(-1))
    size = var_f.size
    if size % ALIGNMENT != 0:
        raise ValueError(f"apply_adam is aligned_only; size {size} not a multiple of {ALIGNMENT}")

    alpha = lr * math.sqrt(1.0 - beta2_power) / (1.0 - beta1_power)
    one_minus_beta1 = 1.0 - beta1
    one_minus_beta2 = 1.0 - beta2
    num_tiles = asc.ceildiv(size, TILE_SIZE)
    apply_adam_kernel[CORE_NUM](var_f, m_f, v_f, grad_f, size, TILE_SIZE,
                                asc.ceildiv(num_tiles, CORE_NUM),
                                one_minus_beta1, one_minus_beta2, alpha, epsilon)
    return var_f.reshape(shape), m_f.reshape(shape), v_f.reshape(shape)


def _numpy_apply_adam(var, m, v, grad, beta1_power, beta2_power, lr, beta1, beta2, epsilon):
    alpha = lr * math.sqrt(1.0 - beta2_power) / (1.0 - beta1_power)
    m_new = m + (1.0 - beta1) * (grad - m)
    v_new = v + (1.0 - beta2) * (grad * grad - v)
    var_new = var - alpha * m_new / (np.sqrt(v_new) + epsilon)
    return var_new, m_new, v_new


def run_kernel(backend: config.Backend, platform: config.Platform):
    config.set_platform(backend, platform)
    test_shapes = [(16, 2048), (32, 4096)]
    rng = np.random.default_rng(seed=2026)
    for shape in test_shapes:
        var = (rng.random(shape, dtype=np.float32) * 2 - 1).astype(np.float32)
        m = (rng.random(shape, dtype=np.float32) * 2 - 1).astype(np.float32)
        v = (rng.random(shape, dtype=np.float32)).astype(np.float32)  # second moment >= 0
        grad = (rng.random(shape, dtype=np.float32) * 2 - 1).astype(np.float32)
        params = dict(beta1_power=0.9, beta2_power=0.999, lr=0.001,
                      beta1=0.9, beta2=0.999, epsilon=1e-8)
        exp_var, exp_m, exp_v = _numpy_apply_adam(var.copy(), m.copy(), v.copy(), grad.copy(), **params)
        out_var, out_m, out_v = apply_adam_launch(var.copy(), m.copy(), v.copy(), grad.copy(), **params)
        np.testing.assert_allclose(out_m, exp_m, atol=1e-5, rtol=1e-4)
        np.testing.assert_allclose(out_v, exp_v, atol=1e-5, rtol=1e-4)
        np.testing.assert_allclose(out_var, exp_var, atol=1e-5, rtol=1e-4)
        logging.info(f"[PASS] Kernel output verified for shape {shape}.")


def test_apply_adam_f32(backend: config.Backend, platform: config.Platform):
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
