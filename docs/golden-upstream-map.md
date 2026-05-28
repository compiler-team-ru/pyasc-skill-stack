# Golden ↔ upstream test map

**Pinned upstream:** [`https://gitcode.com/compiler-team/pyasc`](https://gitcode.com/compiler-team/pyasc) @ `7b85554a2bcc67805f90c30b4e262a6209767839` (v2 branch)
**Local read-only clone:** `/home/aloschilov/workspace/pyasc-v2-eval`
**Upstream tree:** `python/test/asc2/{kernels,operations,target}/`

This document is the rosetta stone between this repository's `golden/kernels/*.py`
and the upstream `pyasc` test suite. It is consumed by:

- `skills/pyasc-api-patterns/SKILL.md` — when teaching the rank-consistent
  tiling rule and the three valid tiling patterns.
- `capabilities.yaml` — when authoring `prompt_variants.guided` / `oracle_guided`
  prompt slots that reference a canonical reference implementation.
- New golden creation (`abs_f32.py`, `add_f16.py`, future cells) — to pick an
  intended upstream source instead of inventing skeletons.
- Phase 4a (MatMul taxonomy) — the upstream `kernels/test_matmul_*` family +
  `target/test_matmul_k_tiled.py` already enumerate the six matmul variants;
  Phase 4a becomes a vendoring task.

If the upstream pin is bumped, this document **must** be refreshed in the same
commit. The pin lives in [`docker/Dockerfile`](docker/Dockerfile) (build-time
`PYASC_GIT_REV`) and `docs/cann-setup.md` (host-side `pyasc-v2-eval` recipe).

## Upstream test taxonomy

Upstream `python/test/asc2/` is split into three buckets:

| Bucket | Intent | Canonical example |
|---|---|---|
| `kernels/` | End-to-end kernels in one specific tiling pattern, aligned with LLVM-pass coverage. **Multiple files per op are common** (e.g. five `test_matmul_*` variants). Uses `torch.testing.assert_close`. | `kernels/test_vadd.py` (1D flatten), `kernels/test_gelu.py` (2D row-tiled, parametrized erf/tanh) |
| `operations/` | Public `asc2.*` API correctness. Single-core, fixed small shape (typically `[32]`), `torch.allclose(atol=1e-3)` for f32. Parametrized over op × dtype × format (VV/VS/SV) × mask. | `operations/test_unary_ops.py` (covers `asc2.abs`, `asc2.tanh`, `asc2.erf`, ...), `operations/test_binary_ops.py` (covers `asc2.add`, `asc2.mul`, ...) |
| `target/` | CANN-target-shape coverage with cache-line aligned tail handling. **Production tiling**. Parametrized over many shape / core_num / unroll_factor tuples. Uses `torch.testing.assert_close`. | `target/test_vadd.py` (8 shape cases), `target/test_gelu.py` (1D tanh, multi-dim shapes flattened, f16/f32/bf16) |

Our `golden/kernels/*.py` corresponds most closely to upstream `target/`
(production-grade tiling skeleton) but is currently *simpler*: numpy interface,
single-shape per file, no tail handling. The map below records each divergence.

## Mapping table

| Our golden | Op / dtype | Upstream `kernels/` | Upstream `operations/` | Upstream `target/` | Divergence summary |
|---|---|---|---|---|---|
| `golden/kernels/abs_f16.py` | `asc2.abs / float16` | *(none)* | `operations/test_unary_ops.py` (parametrized; f32 only with `torch.allclose(atol=1e-3)`) | *(none)* | numpy vs torch; we exercise f16 (upstream `operations/` exercises f32 only); we tile (upstream is single-core `[32]`); tolerance `atol=rtol=1e-3` matches the f32 contract from `operations/`. |
| `golden/kernels/abs_f32.py` *(new — Phase 9)* | `asc2.abs / float32` | *(none)* | `operations/test_unary_ops.py` (f32, `atol=1e-3`) | `target/test_vadd.py` (tiling skeleton) | Will adapt the single-input variant of the `target/test_vadd.py` skeleton with `asc2.abs` in place of `+`. Tolerance taken from upstream `operations/test_unary_ops.py` (`atol=1e-3`). numpy interface; cite both upstream sources in header. |
| *(no golden yet)* `golden/kernels/add_f16.py` *(new — Phase 9)* | `asc2.add / float16` | `kernels/test_vadd.py` (1D flat, f32, `np.testing.assert_allclose`) | `operations/test_binary_ops.py` (parametrized VV/VS/SV, bf16/f32/i32 — *no f16* upstream) | `target/test_vadd.py` (8 parametrized cases, f32, cache-line aligned tail) | Switch dtype to f16; carry over `ALIGNMENT_ELEMENTS = 32 // dtype.itemsize` from `target/`; parametrize at least 3 of the 8 upstream shapes (`[9216]`, `[87768]`, `[8192]`). Tolerance `atol=rtol=1e-3`. numpy interface. |
| `golden/kernels/gelu_f16.py` | `asc2.erf + mul + add / float16` | `kernels/test_gelu.py` (2D row-tiled `[num_rows, num_columns]`, parametrized `approximate=[True, False]`, f32 only, `rtol=1e-3, atol=1e-5`) | `operations/test_unary_ops.py` (`asc2.erf` API only) | `target/test_gelu.py` (1D flatten, tanh form, parametrized f16/f32/bf16 across 6 multi-dim shapes, `rtol=1e-3, atol=1e-3`) | We use 1D flatten + **erf form** + numpy + single shape pair `[8192, 131072]` + `atol=rtol=5e-2`. Upstream `target/` is **tanh form** for f16; reviving f16-erf in upstream `kernels/` is f32-only there. Replacing our golden with the upstream tanh form is out of scope for Phase 9 (deferred — contract change for the cell). |
| `golden/kernels/gelu_f32.py` *(rewritten — Phase 10)* | `asc2.exp + add + div / float32` | `kernels/test_gelu.py` (2D row-tiled, f32, erf+tanh parametrized, `rtol=1e-3, atol=1e-5`) | *(n/a; covered in test_unary_ops for the individual ops)* | `target/test_gelu.py` (1D flatten, **tanh form**, f32 at `rtol=1e-3, atol=1e-3`) | Phase 10 re-vendor: Pattern A 1D-flatten skeleton from upstream `target/test_gelu.py`, but with the lean exp/sigmoid restatement (`x / (1 + exp(-sqrt(8/pi)*(x + 0.044715*x^3)))`) replacing `asc2.tanh + scalar_mul + add + scalar_mul`. **Math correction**: upstream `target/test_gelu.py` uses swapped polynomial coefficients (`x^3 + 0.044715*x` vs PyTorch standard `x + 0.044715*x^3`); upstream's test passes because the reference uses the same swap. We use PyTorch standard coefficients so the cell's contract remains compatible with `torch.nn.functional.gelu(approximate='tanh')`. Tolerance `atol=rtol=1e-2`. numpy interface. |
| `golden/kernels/leaky_relu_f16.py` | `asc2.where / float16` | `kernels/test_leaky_relu.py` (1D flat, size=8192, torch.bfloat16, `alpha=0.1`, `torch.testing.assert_close` with default tol) | `operations/test_where.py` (`asc2.where` API) | *(none)* | We exercise f16 + numpy + tighter `atol=rtol=1e-3`; upstream `kernels/` uses bf16. Same 1D flatten skeleton; tile sizing differs. |
| `golden/kernels/matmul_f16.py` | `asc2.matmul / float16` | `kernels/test_matmul_fixpipe.py`, `test_matmul_l0c_to_l1.py`, `test_matmul_mnblock.py`, `test_matmul_tiled.py`, `test_matmul_transpose.py` | *(n/a)* | `target/test_matmul_k_tiled.py` | Upstream covers six matmul variants. Our single golden is the SplitMat-only baseline. Phase 4a (postponed) maps each variant to a `capabilities.yaml` cell. |
| `golden/kernels/reduce_sum_f16.py` | `asc2.reduce_sum / float16` | *(none specifically)* | `operations/test_reduce_ops.py` (parametrized reduce_{sum,max,min,...} over dtype × axis) | `target/test_reduce_sum.py` (very large parametrize: 22 cases over row/col reductions, f32 only, cache-line aligned tile_shape, `atol=rtol=1e-3`) | We exercise f16 + numpy + 2D `[num_rows, num_cols]` with row-wise sum; upstream `target/` is f32 with both row+col variants and dispatching to `reduce_sum_rows` / `reduce_sum_cols`. Skeleton roughly matches the row variant. |
| `golden/kernels/reduce_sum_f32.py` | `asc2.reduce_sum / float32` | *(none specifically)* | `operations/test_reduce_ops.py` | `target/test_reduce_sum.py` (matches dtype) | Same row-reduction skeleton; numpy vs torch; fewer parametrized shapes. |
| `golden/kernels/rms_norm_f16.py` | RMSNorm composition / float16 | *(none)* | *(none in asc2/)* | *(none)* | **No upstream `asc2/` test for RMSNorm**. Closest upstream coverage is `python/test/unit/language/adv/test_normalization.py`. Our golden is a from-scratch composition. The tactical `rms_norm_platform_fix_and_gelu_tanh` plan added `platform: Ascend950PR_9599`; v2-eval confirms stable. |
| `golden/kernels/rms_norm_f32.py` | RMSNorm composition / float32 | *(none)* | *(none in asc2/)* | *(none)* | Same as f16; no upstream `asc2/` test. |
| `golden/kernels/softmax_f16.py` | `asc2.softmax / float16` | `kernels/test_softmax.py`, `kernels/test_softmax_fused.py` | `operations/test_softmax.py` (`asc2.softmax` API) | `target/test_softmax.py`, `target/test_softmax_fused.py` | Upstream has dedicated kernels for fused/unfused softmax; we ship one variant. Phase 5 (tail/mask investigation) may revisit row-split semantics. |

## Recurring divergences

All goldens differ from upstream on at least these three axes. They are
intentional and recorded once here so individual golden headers can refer to
this section rather than restating it.

1. **Interface — `numpy` vs `torch`.** Our skill stack teaches numpy
   (`numpy.testing.assert_allclose`); upstream tests are written in
   `torch.testing.assert_close`. Verification semantics are equivalent for the
   tolerance contract; the agent never directly imports torch in our generated
   kernels. Decision: keep numpy in `golden/kernels/`. Out of scope to change.

2. **Shape coverage — single pair vs upstream parametrized.** Upstream
   `target/` tests typically parametrize over 6–22 shape / core_num /
   unroll_factor tuples to cover the CANN target-shape combinatorics. Our
   goldens exercise one or two sizes. New goldens added under Phase 9
   (`abs_f32.py`, `add_f16.py`) pick at least three upstream shapes.

3. **Tail handling — `aligned_only` vs upstream cache-line aligned padding.**
   Our goldens declare `tail_behavior: aligned_only` and require the dispatch
   shape to be an exact multiple of `TILE_SIZE * CORE_NUM`. Upstream `target/`
   computes `ALIGNMENT_ELEMENTS = 32 // dtype.itemsize`, pads input to that
   alignment, and runs a tail iteration. The new Phase 9 goldens adopt the
   upstream pattern (Pattern B in `skills/pyasc-api-patterns/SKILL.md`).

## Golden header convention (Phase 9 onward)

New or rewritten goldens carry a single-line upstream-source citation in the
module docstring:

```
# Vendored from pyasc-v2-eval@7b85554a:
#   python/test/asc2/operations/test_unary_ops.py  (asc2.abs API + atol=1e-3)
#   python/test/asc2/target/test_vadd.py           (1D-flatten tiling skeleton)
# numpy / numpy.testing.assert_allclose adapted from torch / torch.testing.assert_close.
```

This makes the upstream provenance grep-able and survives later refactors.

## Phase 4a implication (MatMul taxonomy)

The six matmul configurations enumerated in
[`.cursor/plans/phase_4a_matmul_taxonomy.plan.md`](.cursor/plans/phase_4a_matmul_taxonomy.plan.md)
(SplitMat-only, ABL1Full, AL1Full, BL1Full, tiled-MN, tiled-K) are already
expressed in upstream:

| Phase 4a variant | Upstream test |
|---|---|
| SplitMat-only baseline | already in our `golden/kernels/matmul_f16.py` |
| Tiled-K | `target/test_matmul_k_tiled.py` |
| MN-block tiled | `kernels/test_matmul_mnblock.py` |
| Tiled (MN, fixed-K) | `kernels/test_matmul_tiled.py` |
| L0C → L1 reuse | `kernels/test_matmul_l0c_to_l1.py` |
| Fixpipe-fused | `kernels/test_matmul_fixpipe.py` |
| Transposed inputs (orthogonal) | `kernels/test_matmul_transpose.py` |

When Phase 4a executes, it should vendor these in (one golden per variant)
with the same header convention as Phase 9 — not invent new skeletons.
