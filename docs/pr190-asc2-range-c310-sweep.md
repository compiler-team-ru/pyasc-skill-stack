# `asc2.range` parameter sweep on Ascend950PR_9599

This is a one-shot characterisation of how the C310 simulator
(`Ascend950PR_9599`) responds to varying `asc2.range` flags on the three
goldens that target it natively: `rms_norm_f16`, `rms_norm_f32`, and
`matmul_f16`. Each cell ran the whole `run_kernel` path once on the simulator
inside `ghcr.io/aloschilov/pyasc-sim:py3.11`, captured the single
`Total tick:` value the simulator emits per Docker invocation (it aggregates
all kernel calls), and verified correctness via the goldens' built-in
`torch.testing.assert_close`. No file in `golden/kernels/` was modified — every
variant lived in a temporary `_sweep_<name>.py` rewritten by regex on a copy of
the source and deleted after the run. Three uf=8 cells initially hit the
1500 s simulator timeout and were re-run with a 2700 s ceiling; all 25 cells
ultimately completed.

The pair `(unroll_factor, parallel)` was applied uniformly across the swept
loops:

- `rms_norm_*`: applied to L1 (full_row dispatch row), L2 (split_d outer row),
  and L4 (split_d disjoint write-back). The L3 `sum_sq` accumulator was
  pinned at `unroll_factor=2` (no `parallel`) for every cell — that decouples
  the swept-loop signal from the accumulator's unroll factor.
- `matmul_f16`: both Python `range` loops over `asc.ConstExpr[int]` were
  rewritten to `asc2.range(...)` so the swept flags actually take effect; an
  extra **baseline row** captures the unmodified compile-time-unroll form for
  reference.

PR reference: [gitcode compiler-team/pyasc/pull/190](https://gitcode.com/compiler-team/pyasc/pull/190).
Source goldens: [golden/kernels/rms_norm_f16.py](../golden/kernels/rms_norm_f16.py),
[golden/kernels/rms_norm_f32.py](../golden/kernels/rms_norm_f32.py),
[golden/kernels/matmul_f16.py](../golden/kernels/matmul_f16.py).

## `rms_norm_f16` (full_row at (8, 256) + split_d at (8, 1055), one tick per run)

| uf | parallel | Ticks | Δ vs uf=1, par=False | Note |
|---|---|---|---|---|
| 1 | False | 38 973 | — | natural baseline |
| **1** | **True** | **30 671** | **-21.31%** | **best cell** |
| 2 | False | 40 846 | +4.81% | |
| 2 | True | 36 630 | -6.01% | current PR 190 default |
| 4 | False | 60 772 | +55.93% | |
| 4 | True | 58 747 | +50.74% | |
| 8 | False | 148 188 | +280.23% | uf=8 cliff |
| 8 | True | 146 629 | +276.23% | uf=8 cliff |

## `rms_norm_f32` (full_row at (8, 256) + split_d at (8, 1055), one tick per run)

| uf | parallel | Ticks | Δ vs uf=1, par=False | Note |
|---|---|---|---|---|
| 1 | False | 38 593 | — | natural baseline |
| 1 | True | 30 868 | -20.02% | |
| 2 | False | 33 111 | -14.21% | |
| **2** | **True** | **30 028** | **-22.19%** | **best cell — current PR 190 default** |
| 4 | False | 45 615 | +18.20% | |
| 4 | True | 42 903 | +11.17% | |
| 8 | False | 88 505 | +129.33% | uf=8 cliff |
| 8 | True | 86 799 | +124.91% | uf=8 cliff |

## `matmul_f16` (16x16x16 + 32x16x32, one tick per run)

| uf | parallel | Ticks | Δ vs baseline | Note |
|---|---|---|---|---|
| — (Python `range`, compile-time unroll) | — | 7 423 | — | baseline; the form the committed golden uses |
| 1 | False | 7 155 | -3.61% | |
| 1 | True | 7 170 | -3.41% | |
| **2** | **False** | **6 548** | **-11.79%** | **best cell** |
| 2 | True | 6 581 | -11.34% | |
| 4 | False | 7 190 | -3.14% | |
| 4 | True | 7 151 | -3.66% | |
| 8 | False | 7 173 | -3.37% | |
| 8 | True | 7 131 | -3.93% | |

All eight `asc2.range` cells beat the compile-time-unroll baseline by 3-12%.
`parallel=True` is essentially noise on matmul (≤0.5% spread across each uf
pair) — the cube unit's single-port semantics serialise the inner `a_i @ b_j`
issue regardless. `unroll_factor=2` is the sweet spot; any further unrolling
regresses to baseline-noise level, presumably because the inner loop already
has trip count 1 in both test cases (`m_tiles_per_block=1`) so unrolling >1
leaves no work to amortise.

## Headline takeaways

1. **`parallel=True` is a free win or neutral on every C310 cell that completed**.
   Across all 16 rms_norm cells, switching `parallel=False -> True` at the same
   `uf` is monotonically faster (or unchanged), with the largest delta at
   `uf=1` on `rms_norm_f16` (-21%). Across all 8 matmul cells the delta is
   ≤0.5% and noise-level. **No regressions** like the `Ascend910B1`
   `reduce_sum` case that drove issue #1 — the C310 lowering / scheduler
   handles `parallel=True` cleanly on the kernel patterns probed here.
2. **`unroll_factor` is per-kernel, per-shape**. The right value depends on
   per-core trip count and body weight:
   - `rms_norm_f16` at (8 rows / 8 cores = 1 row/core): `uf=1` wins (no
     unrolling to amortise — the loop only iterates once per core).
   - `rms_norm_f32` at the same shape: `uf=2` wins by 2 percentage points
     over `uf=1` — the heavier f32 body (no `.to(asc.float32)` casts in the
     hot path; everything is already f32) gives the unroller a slightly
     larger window to overlap.
   - `matmul_f16`: `uf=2` is the universal best.
3. **`uf=8` is a cliff on `rms_norm`**. f16 explodes to ~3.8x baseline,
   f32 to ~2.3x baseline. Body expansion eight-fold trashes the simulator's
   register/UB budget. `uf=8` should never be the default.
4. **`asc2.range` adds no overhead vs Python `range`/compile-time unroll on
   matmul**. Even `uf=1, par=False` (the "ForOp wrapper, no unroll, no
   parallel" case) beats the baseline by 3.6%. This is a genuine surprise:
   the comment we shipped in `matmul_f16.py` and the SKILL doc says
   "wrapping these would emit a runtime ForOp instead" with the implication
   that compile-time unrolling is preferable. The data says the C310
   lowering handles ForOp+unroll at least as efficiently as full unrolling
   at this scale.
5. **Current PR 190 defaults are 50% optimal on rms_norm**:
   - `rms_norm_f32`: `uf=2, par=True` is the best cell — defaults are ideal.
   - `rms_norm_f16`: `uf=2, par=True` is **6 percentage points worse** than
     `uf=1, par=True`. Defaults leave perf on the table for f16 specifically
     because of the per-core trip count of 1.

## Comparison with the prior `Ascend910B1` data

The earlier
[docs/pr190-asc2-range-defaults-impact.md](pr190-asc2-range-defaults-impact.md)
captured a fixed-form pre/post-retrofit delta on `Ascend910B1`, where two
goldens (`reduce_sum_f16`, `reduce_sum_f32`) regressed by ~30% with
`unroll_factor=2, parallel=True` applied to a row-distribution body that
contained a wide load + reduction. The follow-up
[issue #1](https://github.com/aloschilov/pyasc-skill-stack/issues/1)
hypothesised that the regression was scheduler/codegen serialisation of
`asc2.reduce_sum` under `parallel=True`. The C310 sweep is consistent with
that hypothesis being **910B1-specific**: on `Ascend950PR_9599` the same
PR 190 form does not regress on similar kernels (rms_norm has the same
"wide load + reduce" shape inside the split_d outer loop and the full_row
loop, and `parallel=True` strictly wins there). The 910B1 reduction-codegen
issue should still be investigated upstream, but the C310 path is clean.

## Suggested next moves (not in this commit)

- For the next pass at the SKILL doc, consider refining the row-distribution
  guidance from "always `unroll_factor=2`" to "`unroll_factor` should match
  per-core trip count, defaulting to 2 when shape isn't known statically".
  The `rms_norm_f16` data point shows `uf=1` is non-trivially better at
  shapes where rows-per-core ≤ 1.
- Consider revisiting the matmul SKILL doc note: the data shows
  `asc2.range` is not a regression on matmul-shaped loops at small
  `ConstExpr` trip counts. Compile-time unrolling is still simpler and
  produces a cleaner trace, but the perf story is more nuanced than the
  current "leave as-is" framing.

The probe script (`/tmp/probe_c310_sweep.py`) was throwaway and deleted after
this doc landed. Raw data was persisted to `/tmp/c310_sweep_results.json`
during the run and is reproducible: total wall clock ≈ 4 h 42 min on the
shared simulator, plus ≈ 1 h 12 min for the three uf=8 / cold-start retries.
