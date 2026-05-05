# PR 190 `asc2.range` defaults — one-shot tick-delta impact

PR 190 ([gitcode compiler-team/pyasc/pull/190](https://gitcode.com/compiler-team/pyasc/pull/190))
adds `unroll_factor` and `parallel` knobs to `asc2.range`. The skill stack now
prescribes `unroll_factor=2` always and `parallel=True` whenever the loop body
has no read of a value written by a previous iteration; see
[skills/pyasc-api-patterns/SKILL.md](../skills/pyasc-api-patterns/SKILL.md)
for the full rule and the worked examples. This file is the throwaway evidence
that the rule moves the simulator-tick needle on the 10 goldens it was applied
to. The probe was a `Total tick:` grep over a single simulator run per golden,
captured pre- and post-retrofit on the platform each kernel uses in CI
(`Ascend910B1` for elementwise/reduction/softmax, `Ascend950PR_9599` for
`matmul_f16`, `rms_norm_f16`, `rms_norm_f32`).

| Golden | Platform | Pre ticks | Post ticks | Δ % | Loop pattern | Note |
|---|---|---|---|---|---|---|
| `abs_f16` | Ascend910B1 | 118 103 | 118 368 | +0.22% | elementwise tile loop | 1 tile / block — `parallel=True` cannot help; within run-to-run noise |
| `gelu_f16` | Ascend910B1 | 172 113 | 165 555 | -3.81% | elementwise tile loop | parallel disjoint stores pay off |
| `gelu_f32` | Ascend910B1 | 200 104 | 179 708 | -10.19% | elementwise tile loop | composed body, more overlap available |
| `leaky_relu_f16` | Ascend910B1 | 125 483 | 114 867 | -8.46% | elementwise tile loop | composed body (where + scale) |
| `matmul_f16` | Ascend950PR_9599 | 7 280 | 7 260 | -0.27% | comment-only (Python `range` over `ConstExpr`) | within noise; no IR change |
| `reduce_sum_f16` | Ascend910B1 | 5 007 | 6 566 | **+31.14%** | row distribution | regression — see follow-up below |
| `reduce_sum_f32` | Ascend910B1 | 5 156 | 6 605 | **+28.10%** | row distribution | regression — see follow-up below |
| `rms_norm_f16` | Ascend950PR_9599 | 38 322 | 36 477 | -4.81% | row dist + accumulator + write-back | three loops retrofitted; `sum_sq` carried, no `parallel` |
| `rms_norm_f32` | Ascend950PR_9599 | 38 725 | 30 177 | -22.07% | row dist + accumulator + write-back | three loops retrofitted; `sum_sq` carried, no `parallel` |
| `softmax_f16` | Ascend910B1 | 6 203 | 6 306 | +1.66% | n/a (no `asc2.range`) | unchanged kernel; within noise |

7 of the 10 deltas are perf-neutral or improvements — the largest wins are on
the composed elementwise/RMSNorm bodies, which is where `unroll_factor=2 +
parallel=True` exposes the most independent tiles per core. The two regressions
are both `reduce_sum` row-distribution loops, where the body is one
`asc2.load → asc2.reduce_sum → asc2.full → asc2.store` per row across 32 rows /
16 cores. In that exact shape the simulator schedules the unrolled-by-2 +
parallel form materially worse than the default — likely register-pressure
or memory-bank ordering inside the per-row `asc2.reduce_sum`, since per-row
output is genuinely disjoint and the rule's safety condition is satisfied.
Stability check (composed kernels run 3x): `gelu_f32` 179 708 / 177 239 /
178 446 (~1.4% spread), `rms_norm_f16` 36 477 / 36 954 / 36 345 (~1.7%),
`rms_norm_f32` 30 177 / 30 022 / 29 977 (~0.7%) — the wins are real, not noise.

## Decision

Ship the rule as-prescribed across all 10 goldens including `reduce_sum_*`.
The PR 190 documentation and our SKILL doc both teach the same canonical form;
having one of the two `reduce_sum` row-distribution goldens silently use a
different form purely because of one simulator's scheduler behaviour would be a
worse teaching signal than the +30% tick cost on a kernel that takes a few
microseconds in absolute terms (5 007 → 6 566 ticks, ~1.5 µs). The correctness
verification (`run_and_verify.py … --mode simulator`) and `score_kernel.py`
(15/16 retained) both stay green.

## Follow-up

A targeted post-nightly sweep (`Ascend910B1`, fixed `num_cols=4096`, varying
`num_rows`) refutes the row-count heuristic theory — the regression is
monotonic but stays positive even at 16 rows / core:

| `num_rows` | rows/core | Baseline ticks | PR 190 ticks | Δ % |
|---|---|---|---|---|
| 32 | 2 | 4 938 | 6 440 | +30.42% |
| 64 | 4 | 6 530 | 8 130 | +24.50% |
| 128 | 8 | 9 694 | 11 334 | +16.92% |
| 256 | 16 | 15 913 | 18 009 | +13.17% |

Translation: the regression is not "trip count too low for `parallel=True` to
amortise" — it is the body itself. Each iteration does a wide load
(`row = asc2.load(x_gm, [1, 4096])`, ~8 KB) followed by a heavy
`asc2.reduce_sum`. Doubling the in-flight bodies via `unroll_factor=2 +
parallel=True` causes memory-bank contention on the load side and SIMD
serialization on the reduce side; both costs grow with body weight, not trip
count. The rule's wins on elementwise / RMSNorm bodies hold because those
bodies are light enough that two in-flight iterations actually overlap.

Filed as
[issue #1](https://github.com/aloschilov/pyasc-skill-stack/issues/1) —
investigate whether `asc2.reduce_sum` codegen serialises the SIMD reduction
across `parallel=True` iterations, and whether the SKILL doc rule should add a
"wide-load + per-iteration reduction" caveat to the row-distribution row of
the pattern table.
