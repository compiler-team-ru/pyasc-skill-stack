---
name: Phase 1 golden-kernel spec hygiene
overview: "Concrete sprint plan (~5 engineer-days) for Phase 1 of the quarterly roadmap. Adds per-cell metadata (shape_regime, reduce_axis, output_shape, accumulator_dtype, identity, tail_behavior, padding, partitioning, unsupported_regimes) to every cell in capabilities.yaml, ships docs/glossary.md as the term-of-art reference, decorates every golden kernel with a non-obvious-constraint header block, decides the ReLU scope question, and tightens check_capabilities.py so the new fields are enforced rather than optional."
todos:
  - id: p1-1-design-glossary-relu
    content: "Stage 1.1: Design the cell metadata schema (allowed values + back-compat policy); write docs/glossary.md covering the standard pyasc/asc2 terminology + the new enum values; decide ReLU scope (recommendation: close as covered-by-pattern via abs.representative_of)."
    status: pending
  - id: p1-2-populate-cells
    content: "Stage 1.2: Populate shape_regime/reduce_axis/output_shape/accumulator_dtype/identity/tail_behavior/padding/partitioning/unsupported_regimes for all 12 cells in capabilities.yaml. Per-cell research where the field is not obvious (rms_norm dispatcher, softmax full-row path, matmul cube alignment, reduce_sum OUT_PAD)."
    status: pending
  - id: p1-3-golden-headers
    content: "Stage 1.3: Add a non-obvious-constraint header docstring block to every golden/kernels/*.py covering alignment, UB/L1/L0 placement, padding, tail behavior, accumulator dtype, dispatcher choice, and simulator/platform assumptions (per notes 1.5)."
    status: pending
  - id: p1-4-enforce-checker
    content: "Stage 1.4: Extend tests/tools/check_capabilities.py to validate the new fields (presence + enum membership + cross-field consistency: reduce_axis null iff op is non-reducing). Start as warning, promote to hard fail once 1.2 lands."
    status: pending
  - id: p1-5-verify-noop-nightly
    content: "Stage 1.5: Re-run pr-gate locally; dispatch one nightly to confirm no regression on the 12 cells (the metadata additions must not change prompts or kernel behavior). Update docs/evaluation-methodology.md with a 'Capability cell metadata schema (Phase 1)' section."
    status: pending
isProject: false
---

# Phase 1 — Golden-kernel spec hygiene

Drills into Phase 1 of [pyasc_skill_stack_quarterly_roadmap_aed2c154.plan.md](pyasc_skill_stack_quarterly_roadmap_aed2c154.plan.md) and only that phase. Sized at ~5 engineer-days across ~1.5 weeks. All changes are additive and reporting-only: no prompt rewrites (those land in Phase 2), no kernel behavior changes, no schema_version bump.

## Outcome

After this sprint, every cell in [capabilities.yaml](../../capabilities.yaml) advertises *what it actually claims to prove* — the shape regime, the reduce axis, the accumulator dtype, the tail-handling mechanism, the partitioning strategy, and the regimes that are explicitly out of scope. [docs/glossary.md](../../docs/glossary.md) is the single source of truth for the terminology. Every golden kernel has a non-obvious-constraint header that the agent can read alongside the code. [tests/tools/check_capabilities.py](../../tests/tools/check_capabilities.py) refuses cells with missing or contradictory metadata.

## Stage 1.1 — Design schema + glossary + ReLU scope (~1 ED)

Decide and document, do not touch cells yet.

### Cell metadata schema (additive on `schema_version: "3"`)

```yaml
cells:
  - dtype: float16
    platform: Ascend950PR_9599
    # NEW Phase 1 fields (all additive, no defaults that change behavior):
    shape_regime: fixed | runtime_size_only | dynamic
    reduce_axis: <int> | null            # last-axis is -1; null for non-reducing ops
    output_shape: <list> | "same_as_input"
    accumulator_dtype: float16 | float32 | null
    identity: "0" | "1" | "-inf" | "+inf" | null
    tail_behavior: aligned_only | host_pad | mask | real_shape | host_dispatcher | unsupported
    padding: <int> | null                # element count, e.g. OUT_PAD=8 for f32
    partitioning: row_per_core | tile_per_core | block_grid | host_dispatcher
    unsupported_regimes: [split_row, multi_axis_reduction, dynamic_num_cols, ...]
    # Existing fields untouched:
    prompt: ...
    prompt_variants: ...
    shapes: ...
    golden_status: ...
    golden: ...
    generative_status: ...
    generative_evidence: ...
    semantic_check: [...]
```

Allowed values:

- `shape_regime`:
  - `fixed` — both `num_rows` and `num_cols` (or their per-tier equivalent) are compile-time constants. The current `abs`, `add`, `gelu`, `leaky_relu`, `softmax`, `matmul`, `reduce_sum` cells are all `fixed` today.
  - `runtime_size_only` — shape is a runtime `int` but the kernel does not branch on it (e.g. `rms_norm` `split_d` path streams along D in fixed 64-element tiles).
  - `dynamic` — multiple shape regimes routed by a host dispatcher (e.g. `rms_norm` overall: full_row vs split_d).
- `tail_behavior`:
  - `aligned_only` — kernel assumes shape is a multiple of `tile_size`; no tail logic.
  - `host_pad` — host wraps the result with `asc2.full([1, OUT_PAD], ...)` for 32-byte alignment (today's `reduce_sum`).
  - `mask` — kernel uses `asc2.mask` on the vector path for partial tiles. *(Not yet exercised; will be validated in Phase 5.)*
  - `real_shape` — kernel uses `asc2.load(..., real_shape=...)` for partial loads. *(Not yet exercised; Phase 5.)*
  - `host_dispatcher` — the host picks between a full-tile kernel and a tail-aware kernel before launch (today's `rms_norm`).
  - `unsupported` — the cell explicitly does not handle non-aligned tails; the unsupported case is in `unsupported_regimes`.
- `partitioning`:
  - `row_per_core` — `for row in asc2.range(asc2.block_idx(), num_rows, asc2.block_num())`.
  - `tile_per_core` — each block processes one or more tiles in a flat 1D iteration.
  - `block_grid` — 2D grid (matmul `m_tile × n_tile`).
  - `host_dispatcher` — partition decided by host before launch.
- `identity`: kept as a string so YAML parsers don't coerce `0` to int and `-inf` to a float-NaN surprise.
- `unsupported_regimes`: free-form list of slugs; standard slugs documented in the glossary (`split_row`, `multi_axis_reduction`, `dynamic_num_rows`, `non_16_multiple_shapes`, `k_tiled`, ...).

### Glossary

Author [docs/glossary.md](../../docs/glossary.md). Terms from notes §2.1 plus the new enum values. Structure:

- §1 Shape vocabulary: shape, dtype, contiguous layout, axis, stride.
- §2 Memory hierarchy: GM, UB, L1, L0A, L0B.
- §3 Compute partitioning: tile, block/core, host dispatcher, full-row path, split-row path.
- §4 Reduction vocabulary: reduce axis, accumulator dtype, identity / neutral element, finalization.
- §5 Tail vocabulary: padding, tail, mask, real_shape.
- §6 Cell metadata enums: `shape_regime`, `tail_behavior`, `partitioning`, the standard `unsupported_regimes` slugs.

Each term: one sentence definition + one example from a golden kernel + a back-link to the cell where it is used.

### ReLU scope decision (notes §1.6)

Recommendation: close as covered-by-pattern. The `abs` cell at line 17 of [capabilities.yaml](../../capabilities.yaml) already declares `representative_of: [exp, log, sqrt, relu, erf, sin, cos, neg, ceil, floor, rsqrt, tanh]`. The Tier-0 dashboard chip implicitly covers ReLU; adding a standalone cell would test the same template-substitution generative pattern with no new information.

Deliverable: a one-line decision note appended to [capabilities.yaml](../../capabilities.yaml) under the `abs` operation (`coverage_note: "relu covered by representative_of from abs"`) and a paragraph in [docs/glossary.md](../../docs/glossary.md) §6.

## Stage 1.2 — Populate metadata for the 12 cells (~2 ED)

Per-cell research and fill-in. Order from cheapest (Tier 0) to most research-heavy (rms_norm).

### Tier 0 — `abs/{f16,f32}`, `add/f16`

```yaml
shape_regime: fixed
reduce_axis: null
output_shape: same_as_input
accumulator_dtype: null
identity: null
tail_behavior: aligned_only
padding: null
partitioning: tile_per_core
unsupported_regimes: []
```

Justification: all shapes in [capabilities.yaml](../../capabilities.yaml) for these cells are multiples of 64 (smallest is `[1, 128]`); the goldens use `asc2.range` with `TILE_SIZE` dividing each shape evenly.

### Tier 1 — `reduce_sum/{f32,f16}`

```yaml
shape_regime: fixed
reduce_axis: -1
output_shape: [32]            # for shape [32, 4096]
accumulator_dtype: float32    # both f32 and f16 use a float32 accumulator (see golden header)
identity: "0"
tail_behavior: host_pad
padding: 8                    # f32; for the f16 cell use padding: 16 (OUT_PAD=16)
partitioning: row_per_core
unsupported_regimes:
  - non_last_axis
  - multi_axis_reduction
  - dynamic_num_rows
```

Verify by reading [golden/kernels/reduce_sum_f32.py](../../golden/kernels/reduce_sum_f32.py) and [_f16.py](../../golden/kernels/reduce_sum_f16.py). Update `padding` per dtype.

### Tier 2 — `gelu/{f16,f32}`, `leaky_relu/f16`

```yaml
shape_regime: fixed
reduce_axis: null
output_shape: same_as_input
accumulator_dtype: null
identity: null
tail_behavior: aligned_only
padding: null
partitioning: tile_per_core
unsupported_regimes: []
```

Note: `gelu/f32` uses the tanh/Padé approximation (post the prior plan); add `note: "tanh/Pade form; do not use asc2.erf on float32"` per-cell.

### Tier 3 — `softmax/f16`

```yaml
shape_regime: fixed
reduce_axis: -1
output_shape: same_as_input
accumulator_dtype: float32    # max stage + sum stage both use f32 accumulator
identity: null                # softmax has no single identity; documented as "max-stage identity = -inf, sum-stage identity = 0"
tail_behavior: aligned_only
padding: null
partitioning: row_per_core
unsupported_regimes:
  - split_row                  # long rows that exceed UB are not handled by this cell
  - long_rows_exceeding_UB
```

Add a `regime_note: "Full-row path only: row fits in UB and asc2.softmax handles the full row in one call. Split-row is a separate capability cell, deferred to Phase 5/8."` so the regime claim is explicit (notes §1.3).

### Tier 3 — `matmul/f16`

```yaml
shape_regime: fixed
reduce_axis: -1               # K is the reduction axis
output_shape: [M, N]
accumulator_dtype: float32    # cube accumulator is always f32
identity: "0"
tail_behavior: aligned_only
padding: null
partitioning: block_grid
unsupported_regimes:
  - non_16_multiple_shapes
  - k_tiled                   # current golden does not tile along K
  - bl1_full
  - al1_full
  - abl1_full
```

The `unsupported_regimes` list anchors the Phase 4 MatMul branch: each `unsupported_regimes` entry becomes a candidate new cell.

### Tier 3 — `rms_norm/{f16,f32}`

The most research-heavy cell. Today's cell already documents a two-regime dispatcher (full_row + split_d). Translate that into the new schema:

```yaml
shape_regime: dynamic         # host dispatcher between full_row (ConstExpr num_cols) and split_d (runtime num_cols)
reduce_axis: -1
output_shape: same_as_input
accumulator_dtype: float32
identity: "0"
tail_behavior: host_dispatcher
padding: null                 # full_row requires num_cols % 8 == 0; split_d uses host zero pad to 64-multiple
partitioning: host_dispatcher
unsupported_regimes:
  - dynamic_num_cols_not_8_aligned_in_full_row
  - num_cols_below_split_d_tile_threshold
```

Add a `dispatcher_note: "rms_norm_launch(x, gamma, eps) picks full_row when num_cols * dtype_bytes <= 64KB and num_cols % 8 == 0, otherwise split_d."` so the host contract is captured in metadata, not only in the prompt.

Deliverable: [capabilities.yaml](../../capabilities.yaml) updated for all 12 cells; one local pr-gate run green; no nightly regression.

## Stage 1.3 — Golden kernel header comments (~1 ED)

Per notes §1.5, every file under [golden/kernels/](../../golden/kernels/) gets a top-of-file docstring block. Template:

```python
"""Golden kernel: <op>/<dtype>

Cell metadata (mirrors capabilities.yaml; do not drift):
  - shape_regime: fixed
  - reduce_axis: -1
  - tail_behavior: host_pad (OUT_PAD=8 for f32)
  - partitioning: row_per_core
  - unsupported_regimes: [non_last_axis, multi_axis_reduction]

Non-obvious constraints (notes 1.5):
  - Alignment: input shape must be a multiple of TILE_SIZE=<n>; if not,
    behavior is undefined and verification must reject (no silent
    truncation).
  - UB/L1/L0 placement: <where each tile lives>.
  - Padding: <element count / 32-byte alignment requirement>.
  - Tail behavior: <how partial tiles are handled, or 'unsupported'>.
  - Accumulator dtype: <f32 / cube accumulator / null>.
  - Dispatcher choice (rms_norm only): full_row when ..., otherwise split_d.
  - Simulator/platform assumptions: Ascend950PR_9599 (C310); numpy
    buffers are silently zeroed (use torch.Tensor).
"""
```

Each kernel gets exactly one of these headers, mirroring the cell metadata so the agent never sees them diverge.

Files to touch:

- [golden/kernels/abs_f16.py](../../golden/kernels/abs_f16.py)
- [golden/kernels/gelu_f16.py](../../golden/kernels/gelu_f16.py)
- [golden/kernels/gelu_f32.py](../../golden/kernels/gelu_f32.py)
- [golden/kernels/leaky_relu_f16.py](../../golden/kernels/leaky_relu_f16.py)
- [golden/kernels/matmul_f16.py](../../golden/kernels/matmul_f16.py)
- [golden/kernels/reduce_sum_f16.py](../../golden/kernels/reduce_sum_f16.py)
- [golden/kernels/reduce_sum_f32.py](../../golden/kernels/reduce_sum_f32.py)
- [golden/kernels/rms_norm_f16.py](../../golden/kernels/rms_norm_f16.py)
- [golden/kernels/rms_norm_f32.py](../../golden/kernels/rms_norm_f32.py)
- [golden/kernels/softmax_f16.py](../../golden/kernels/softmax_f16.py)

Deliverable: 10 kernels updated; one local `bash tests/run-tests.sh --fast` run green; no golden simulator regression (merge-gate still passes).

## Stage 1.4 — Enforce in `check_capabilities.py` (~0.5 ED)

Concrete edits in [tests/tools/check_capabilities.py](../../tests/tools/check_capabilities.py):

- Add `_check_cell_metadata(cell, op, tier)` validating:
  - All new required fields are present.
  - Each enum field's value is in the allowed set documented in Stage 1.1.
  - Cross-field consistency:
    - `reduce_axis` is `null` iff `tier in {elementwise}`.
    - `accumulator_dtype` is `null` iff op is purely elementwise (no asc2.reduce_sum and no cube path).
    - `tail_behavior == host_dispatcher` implies `partitioning == host_dispatcher`.
    - `unsupported_regimes` slugs are members of a known set (typo guard).
- Two-phase rollout:
  - **Phase A (during 1.2):** missing/invalid fields emit a `WARN:` line; gate exits 0.
  - **Phase B (after 1.2 lands):** missing/invalid fields hard-fail; gate exits 1.
- New CLI flag `--strict-metadata` toggles Phase A vs Phase B. Default to Phase A for one PR cycle so partial pushes are not blocked.
- Wire pr-gate ([tests/unit/tools/test-capabilities.sh](../../tests/unit/tools/test-capabilities.sh)) to pass `--strict-metadata` once Stage 1.2 has landed for all 12 cells.

Deliverable: enforcement in place; one CI run shows the gate enforcing the new contract.

## Stage 1.5 — Verify + nightly no-op (~0.5 ED)

The metadata additions are reporting-only — no prompt rewrites, no kernel behavior changes. Verify:

- Local pr-gate: `bash tests/ci-gate.sh --tier pr` green.
- Local merge-gate proxy: run each golden through [tests/tools/run_and_verify.py](../../tests/tools/run_and_verify.py) `--mode simulator --platform Ascend950PR_9599`. All 10 goldens still pass with their original tolerances.
- Dispatch one nightly with `tier=nightly`. Expected: 12/12 cells `gen_confirmed` (same baseline as today). If any cell flips, it means a prompt rewrite slipped in — revert immediately.

Update [docs/evaluation-methodology.md](../../docs/evaluation-methodology.md) with a "Capability cell metadata schema (Phase 1)" section pointing at [docs/glossary.md](../../docs/glossary.md). Note that the new fields are reporting-only and that schema_version stays at `"3"`.

Deliverable: one nightly green; the dashboard implicitly gains the new metadata (rendered in Phase 2 / Phase 3 panels).

## Definition of done for Phase 1

- 12/12 cells in [capabilities.yaml](../../capabilities.yaml) have all 9 new metadata fields populated.
- [docs/glossary.md](../../docs/glossary.md) covers every term used in the new metadata + the standard pyasc/asc2 vocabulary from notes §2.1.
- 10/10 goldens in [golden/kernels/](../../golden/kernels/) carry a non-obvious-constraint header that mirrors the cell's metadata.
- [tests/tools/check_capabilities.py](../../tests/tools/check_capabilities.py) enforces the new contract (`--strict-metadata` on in pr-gate).
- One full nightly green: 12/12 cells `gen_confirmed`, no behavior regression.
- ReLU scope decision documented (`coverage_note` on the `abs` op + glossary paragraph).

## Risks specific to Phase 1

- **Premature `unsupported` claims.** Marking softmax `split_row` and matmul `k_tiled` as `unsupported_regimes` today is a *capability claim*, not a *property of asc2*. If Phase 5 (tail/mask) or Phase 4 (MatMul branch) discovers a cheap way to express these regimes, we will need to delete the `unsupported_regimes` entry rather than just adding new cells. Mitigation: prefix every cross-cell-impacting slug with `currently_unsupported_in_this_cell:` in the glossary, and document that removing an entry is the path forward (not adding a contradictory one).
- **Drift between cell metadata and golden kernel header.** Two sources of truth (capabilities.yaml + golden header) will diverge. Mitigation: add a [tests/unit/tools/test-golden-header.sh](../../tests/unit/tools/test-golden-header.sh) check in Phase 1.4 that greps the golden header for the cell's `shape_regime`/`tail_behavior`/`partitioning` strings and fails on mismatch. Cheap to write, hard to defeat.
- **Identity field encoding.** YAML's `-inf` is not a valid float literal in all parsers (PyYAML in safe mode rejects it). Storing identity as a string (`"-inf"`) avoids coercion bugs at the cost of one extra `eval`/`float()` call in any future consumer. The current consumer set is only `check_capabilities.py`, which does string comparison only — no risk.
- **rms_norm `unsupported_regimes` over-specification.** The dispatcher contract is precise (`num_cols * dtype_bytes <= 64KB AND num_cols % 8 == 0` for full_row). Spelling this out in `unsupported_regimes` slugs would be verbose. Mitigation: keep the slug short (`dynamic_num_cols_not_8_aligned_in_full_row`) and put the full contract in `dispatcher_note`.
- **check_capabilities.py false positives on legacy cells.** During Stage 1.2 some cells will be partially populated. Phase-A warnings (not hard fails) prevent blocking the in-flight branch.

## Deferred from Phase 1 (intentionally)

- **Prompt rewrites that read the new metadata.** Phase 2 explicitly. Phase 1 only writes the metadata.
- **`shape_regime: runtime_size_only` cells beyond rms_norm.** Adding truly-dynamic-shape coverage for other ops (abs, gelu, reduce_sum) is a separate workstream once tail/mask (Phase 5) lands.
- **New goldens for split-row softmax, ABL1Full matmul, etc.** Each becomes a new cell in Phases 4/5/8; Phase 1 only declares them as `unsupported_regimes` so the metadata is honest about today's coverage.
- **Tail/mask probes.** Phase 5. Phase 1 documents `tail_behavior: mask` / `tail_behavior: real_shape` as allowed enum values but no cell uses them yet.
- **Schema v4 promotion.** Phase 7. Phase 1's metadata stays under `schema_version: "3"`.
- **`reduce_sum_f32`-like new reduction cells** (`reduce_max`, `reduce_min`, `reduce_prod`). Today's `representative_of: [reduce_max, reduce_min, reduce_prod]` is enough until a generative failure shows otherwise.
