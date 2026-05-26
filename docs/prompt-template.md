# Capability-cell prompt template

The canonical 13-slot order every capability prompt MUST follow. The
slots are *positional* — same order, every time — and *labeled* — each
slot has a one-word header so a human and a model can find the same
information in the same place across every cell.

This template is consumed by [tests/tools/collect_generative_evidence.py](../tests/tools/collect_generative_evidence.py)
and by every per-cell prompt in [capabilities.yaml](../capabilities.yaml).
It is the *structural* contract; the *vocabulary* lives in
[docs/glossary.md](glossary.md); the *protocol semantics* live in
[docs/evaluation-methodology.md §"Comparisons of interest"](evaluation-methodology.md#comparisons-of-interest).

## Slot order

| # | Slot | Type | Reference |
|---|---|---|---|
| 1 | Operator semantics | one-sentence definition + formula | [glossary §1](glossary.md) |
| 2 | Input shapes | concrete list, or `runtime_size_only` | [glossary §1](glossary.md), `shape_regime` |
| 3 | Output shape | concrete shape, or `same_as_input` | [glossary §1](glossary.md), `output_shape` |
| 4 | Dtype | `float16 | float32`; output dtype if different | [glossary §1](glossary.md) |
| 5 | Layout | `contiguous` unless otherwise stated | [glossary §1](glossary.md) |
| 6 | Axis | for reductions; `null` otherwise | [glossary §4](glossary.md), `reduce_axis` |
| 7 | Tiling constraints | `TILE_SIZE`, `CORE_NUM`, partitioning mode | [glossary §3](glossary.md), `partitioning` |
| 8 | Padding / tail behavior | `aligned_only | host_pad | mask | real_shape | host_dispatcher | unsupported` | [glossary §5](glossary.md), `tail_behavior` |
| 9 | Accumulator dtype | `null | float16 | float32` | [glossary §4](glossary.md), `accumulator_dtype` |
| 10 | Numerical tolerance | `atol`, `rtol`, and the reference computation | [glossary §1](glossary.md) |
| 11 | Platform | `Ascend950PR_9599` for every cell today | [glossary §2](glossary.md) |
| 12 | Build/run command | the exact one-liner the agent should produce | — |
| 13 | Expected evidence artifact | `kernel.py`, plus `design.md` / `verification.md` for skills-on | — |

Slots 6 and 9 mirror the Phase 1 cell metadata fields `reduce_axis` and
`accumulator_dtype`. Slot 7 mirrors `partitioning`. Slot 8 mirrors
`tail_behavior`. Cells that contradict their slot text against their
metadata fail [tests/tools/check_capabilities.py](../tests/tools/check_capabilities.py)
under the Phase 1 strict-metadata gate.

## Prompt-variant labeling rules

Mirrors [evaluation-methodology.md §"Prompt-variant labeling rules"](evaluation-methodology.md#prompt-variant-labeling-rules)
with the additional structural constraint that each variant uses a
documented subset of the 13 slots:

| Variant | Slots populated | Allowed content |
|---|---|---|
| `minimal` | 1, 2, 3, 4 | operator name, dtype, shape, semantic definition only |
| `guided` | all 13 | API hints, verification hints, general implementation guidance — but **no oracle content** |
| `oracle_guided` | all 13 + workaround block | anything `guided` plus exact workaround, exact tiling, known failure fix, golden-derived clue, backend bug workaround |
| `human_assisted` | (not stored per-cell) | emitted by the harness when a human intervenes mid-run |

`oracle_guided` is stored as a string (same shape as `minimal`/`guided`)
when its content is a free-form prompt, or as a mapping when the cell
also needs to override `examples_policy` for that variant. See [Stage 2.3 of the Phase 2 plan](../.cursor/plans/phase_2_prompt_methodology.plan.md) and the worked
example below.

### Why this matters for the protocol axis

The protocol matrix in [evaluation-methodology.md §"Protocol-axis CI mapping (Phase 0)"](evaluation-methodology.md#protocol-axis-ci-mapping-phase-0)
binds variants to protocols:

- `P2` runs `minimal` + skills off — slots 1-4 only; everything else is
  the agent's job.
- `P3` runs `guided` + skills off — all 13 slots, no oracle, no skills.
- `P4` runs `guided` + AGENTS.md + skills off — all 13 slots + a
  vendored upstream `AGENTS.md` baseline as `--agents-md`.
- `P6` runs `guided` + skills on — all 13 slots + the local pyasc skill
  stack.

A `guided` variant that silently carries oracle content (e.g.
"reference golden/kernels/X.py for the proven pattern") makes `P3-P2`
inflate and `P6-P4` shrink: prompt value is mis-attributed as skill
value. Phase 2 audits all four oracle-carrying cells and moves the
oracle bits into `oracle_guided` so the deltas measure what their
labels claim. See [evaluation-methodology.md §"Baseline validity"](evaluation-methodology.md#baseline-validity)
for the broader accounting.

## Worked example: `abs/float16`

The simplest cell in [capabilities.yaml](../capabilities.yaml).
`abs/float16` is tier-0 (elementwise), uses `tile_per_core`, has
`tail_behavior=aligned_only`, and no accumulator. The four variants:

### minimal

Slots 1-4 only. No tiling, no verification recipe.

```
Operator: abs (out = |x|).
Input shapes: [1, 128], [4, 2048], [32, 4096].
Output shape: same as input.
Dtype: float16.
```

### guided

All 13 slots, no oracle content. The variant the protocols
`P3 | P4 | P6` consume.

```
Operator: abs (out = |x|).
Input shapes: [1, 128], [4, 2048], [32, 4096].
Output shape: same_as_input.
Dtype: float16.
Layout: contiguous.
Axis: null.
Tiling: TILE_SIZE divides each shape evenly; partition tile_per_core
via asc2.range(asc2.block_idx(), num_tiles, asc2.block_num()).
Tail: aligned_only — no tail logic required.
Accumulator: null.
Tolerance: atol=1e-3, rtol=1e-3 against np.abs(x).
Platform: Ascend950PR_9599.
Build/run: python3.10 kernel.py -r Model -v Ascend950PR_9599.
Evidence: kernel.py; skills-on adds design.md + verification.md.
```

### oracle_guided

`abs/float16` does not have a known workaround (the asc2 builtin
`asc2.abs` works on `float16` on `Ascend950PR_9599` today), so the
cell does *not* declare an `oracle_guided` variant. The four cells
that do (`gelu/float32`, `matmul/float16`, `rms_norm/float16`,
`rms_norm/float32`) follow the same `guided` body with an additional
"Workaround" block at the end:

```
... (all 13 guided slots) ...

Workaround:
  <free-form text describing the backend bug or golden-derived clue
   that fixes the failure mode the guided variant alone cannot
   recover from>
```

### human_assisted

Not stored. If a human intervenes mid-run, the harness records
`prompt_variant: human_assisted` on the evidence file and the cell's
`generative_status` remains unchanged.

## Slot-order sensitivity

Some models perform measurably better when slots follow this exact
order vs. a permuted variant. The Phase 2 measurement (Stage 2.5)
treats slot order as fixed; per-cell slot-order deviations are
documented in a separate appendix here if Stage 2.5 surfaces any
sign-flip.

Today there is no documented per-model deviation: `dashscope/glm-5`
on cloud-default tolerates the canonical order on every cell tested.

## Cross-references

- Vocabulary: [docs/glossary.md](glossary.md).
- Per-cell metadata schema: [docs/glossary.md §6](glossary.md), enforced by
  [tests/tools/check_capabilities.py](../tests/tools/check_capabilities.py).
- Protocol matrix: [docs/evaluation-methodology.md §"Protocol-axis CI mapping (Phase 0)"](evaluation-methodology.md#protocol-axis-ci-mapping-phase-0).
- Comparisons of interest: [docs/evaluation-methodology.md §"Comparisons of interest"](evaluation-methodology.md#comparisons-of-interest).
- Examples policy: per-cell `examples_policy` field, validated by
  [tests/tools/check_capabilities.py](../tests/tools/check_capabilities.py)
  to match the protocol's `allowed_context`.
