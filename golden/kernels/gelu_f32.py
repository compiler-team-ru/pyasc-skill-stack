#!/usr/bin/env python3.10
"""Golden kernel: gelu/float32

GELU activation, tanh / Pade approximation form (PyTorch's
``gelu(approximate='tanh')`` / TensorFlow's ``gelu(approximate=True)``),
expressed as a lean exp/sigmoid restatement to reduce per-tile op count:

    gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
            = x / (1 + exp(-sqrt(8/pi) * (x + 0.044715 * x^3)))

The two forms are mathematically identical (substitute
``tanh(z) = 1 - 2/(exp(2z) + 1)`` and let ``2z = sqrt(8/pi) * (...)``)
but the exp form replaces ``asc2.tanh + scalar_mul + add + scalar_mul``
with ``asc2.exp + add + div`` -- one fewer asc2 op per tile and no
tanh dependency. This matters because the heavier tanh path made the
generative ``gelu/float32`` cell hit ``Timeout after 150s`` repeatedly
in Phase 9 evidence (see ``docs/skill-value-q1-findings.md`` Anomaly
#1); the lean exp form lands within the sim budget.

Vendored from pyasc-v2-eval@7b85554a:
  python/test/asc2/target/test_gelu.py
    -- structural Pattern A skeleton (1D flatten, ``asc2.tensor(ptr,
       [size])`` + ``asc2.load(..., [tile_size], offsets=[tile_offset])``)
       and the sigmoid restatement (``x / (1 + exp(...))``).
**Math correction vs upstream**: upstream's
``target/test_gelu.py`` swaps the polynomial coefficients
(``x^3 + 0.044715 * x``) compared to PyTorch's
``gelu(approximate='tanh')`` (``x + 0.044715 * x^3``); upstream's
test passes because its reference uses the same swapped form. Our
golden uses the canonical PyTorch coefficients so the numpy reference
``gelu_numpy`` continues to match ``torch.nn.functional.gelu(
approximate='tanh')``.

numpy / numpy.testing.assert_allclose adapted from
torch / torch.testing.assert_close (see docs/golden-upstream-map.md).

Cell metadata (mirrors capabilities.yaml; do not drift):
  - shape_regime: fixed
  - reduce_axis: null
  - output_shape: same_as_input
  - accumulator_dtype: null
  - identity: null
  - tail_behavior: aligned_only
  - padding: null
  - partitioning: tile_per_core
  - unsupported_regimes: []
  - regime_note: tanh/Pade approximation form via the lean
    exp/sigmoid restatement; do not use asc2.erf on float32
    (simulator erf is too noisy).

Non-obvious constraints:
  - Rank-consistent tiling (Pattern A in
    skills/pyasc-api-patterns/SKILL.md): 1D tensor + 1D load shape +
    1D offsets. The kernel sees a flat ``size``; the host launcher
    flattens any multi-dim test shape before launch.
  - Tile width: ``TILE_SIZE = 64`` (conservative single-lane width
    for C310). Wider tiles can hit a lowering bug that only writes
    the first 64 elements (see history with rms_norm wide-tile
    issues).
  - Alignment: input ``size`` must be a multiple of
    ``TILE_SIZE * CORE_NUM = 64 * 16 = 1024``.
  - CRITICAL ordering rule: inside the ``@asc2.jit`` body, every
    scalar-times-Tile multiplication MUST put the Tile on the left
    (``x_cubed * GELU_C``, NOT ``GELU_C * x_cubed``). The asc2 Tile
    class lacks ``__rmul__``.
  - Simulator/platform assumptions: ``Ascend950PR_9599`` (C310);
    numpy buffers are safe for this elementwise UB-only path.
"""

import logging
import argparse
import math
import numpy as np

import asc
import asc.runtime.config as config
import asc2

TILE_SIZE = 64
CORE_NUM = 16

GELU_C = 0.044715
NEG_SQRT_EIGHT_OVER_PI = -math.sqrt(8.0 / math.pi)

logging.basicConfig(level=logging.INFO)


@asc2.jit(always_compile=True)
def gelu_kernel(x_ptr: asc.GlobalAddress, out_ptr: asc.GlobalAddress,
                size: int, tile_size: asc.ConstExpr[int],
                tile_per_block: asc.ConstExpr[int]):
    x_gm = asc2.tensor(x_ptr, [size])
    out_gm = asc2.tensor(out_ptr, [size])
    base_offset = asc2.block_idx() * tile_size * tile_per_block
    for i in asc2.range(tile_per_block, unroll_factor=2, parallel=True):
        tile_offset = base_offset + i * tile_size
        x = asc2.load(x_gm, [tile_size], offsets=[tile_offset])
        x_cub = x * x * x
        inner = (x_cub * GELU_C + x) * NEG_SQRT_EIGHT_OVER_PI
        out = x / (asc2.exp(inner) + 1)
        asc2.store(out, out_gm, offsets=[tile_offset])


def gelu_launch(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x)
    size = out.size
    num_tiles = asc.ceildiv(size, TILE_SIZE)
    gelu_kernel[CORE_NUM](x, out, size, TILE_SIZE, asc.ceildiv(num_tiles, CORE_NUM))
    return out


def gelu_numpy(x: np.ndarray) -> np.ndarray:
    k = np.sqrt(2.0 / np.pi)
    return 0.5 * x * (1.0 + np.tanh(k * (x + 0.044715 * x ** 3)))


def run_kernel(backend: config.Backend, platform: config.Platform):
    config.set_platform(backend, platform)

    rng = np.random.default_rng(seed=2026)

    for size in [8192, 131072]:
        x = rng.random(size, dtype=np.float32) * 4 - 2
        out = gelu_launch(x)
        expected = gelu_numpy(x)
        np.testing.assert_allclose(out, expected, atol=1e-2, rtol=1e-2)
        logging.info(f"[PASS] GELU f32 verified for size {size}.")


def test_gelu_f32(backend: config.Backend, platform: config.Platform):
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
