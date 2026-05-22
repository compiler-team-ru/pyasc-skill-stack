---
name: Phase 1 golden-kernel spec hygiene
overview: "Concrete sprint plan for Phase 1 of the quarterly roadmap. Adds per-cell metadata to every cell in capabilities.yaml, ships docs/glossary.md, decorates every golden kernel with a non-obvious-constraint header, decides ReLU scope, and tightens check_capabilities.py. As of the 2026-05-22 precision audit, all five stages are implemented uncommitted in the worktree (10/10 golden headers, 12/12 populated cells, glossary.md present at 288 lines, check_capabilities --strict-metadata default, test-golden-header.sh green). This revision demotes stages to commit + audit and adds Stage 1.0 (land in-flight code)."
todos:
  - id: p1-0-land-inflight
    content: "Stage 1.0 (NEW): Land the in-flight Phase 1 worktree as the 'Group B' PR described in phase_0_protocol_aware_harness.plan.md Stage 0.0 (10 golden kernels + capabilities.yaml metadata + check_capabilities.py + docs/glossary.md + tests/unit/tools/test-golden-header.sh). One commit per concern; CI green; --strict-metadata default verified in pr-gate."
    status: pending
  - id: p1-1-design-glossary-relu
    content: "Stage 1.1 [DONE in worktree]: docs/glossary.md (288 lines, \u00a7\u00a71\u20136) present; covers shape vocabulary, memory hierarchy, compute partitioning, reduction vocabulary, tail vocabulary, cell metadata enums. Verify during Stage 1.0 PR review. ReLU scope decision: still owed (no coverage_note on the abs op yet)."
    status: pending
  - id: p1-2-populate-cells
    content: "Stage 1.2 [DONE in worktree]: 12/12 cells in capabilities.yaml carry shape_regime, reduce_axis, output_shape, accumulator_dtype, identity, tail_behavior, padding, partitioning, unsupported_regimes. Verify the dispatcher_note on rms_norm and the regime_note on gelu/f32 + softmax/f16 are present during Stage 1.0 PR review."
    status: pending
  - id: p1-3-golden-headers
    content: "Stage 1.3 [DONE in worktree]: 10/10 golden kernels have 'Cell metadata (mirrors capabilities.yaml; do not drift)' + 'Non-obvious constraints (Phase 1.5)' header blocks. test-golden-header.sh enforces shape_regime/tail_behavior/partitioning match against capabilities.yaml."
    status: pending
  - id: p1-4-enforce-checker
    content: "Stage 1.4 [DONE in worktree]: check_capabilities.py _check_cell_metadata enforces presence + enum membership + cross-field consistency (reduce_axis<->accumulator_dtype, tail_behavior<->partitioning host_dispatcher, tier<->reduce_axis); --strict-metadata default; --no-strict-metadata escape hatch documented."
    status: pending
  - id: p1-5-verify-noop-nightly
    content: "Stage 1.5 [DONE in worktree]: docs/evaluation-methodology.md 'Capability cell metadata schema (Phase 1)' section referenced from check_capabilities.py docstring (line 207). Dispatch a nightly post-Stage-1.0 to confirm 12/12 cells stay gen_confirmed (no prompt/behavior regression)."
    status: pending
  - id: p1-6-relu-and-test-wiring
    content: "Stage 1.6 (NEW): Two leftovers from the precision audit. (a) Add the explicit ReLU scope decision as a coverage_note on the abs op + glossary paragraph (currently undocumented despite Stage 1.1 design). (b) Wire tests/unit/tools/test-golden-header.sh into pr-gate so future header/metadata drift hard-fails (today the test exists but no wrapper invokes it)."
    status: pending
isProject: false
---

# Phase 1 — Golden-kernel spec hygiene

Drills into Phase 1 of [pyasc_skill_stack_quarterly_roadmap_aed2c154.plan.md](pyasc_skill_stack_quarterly_roadmap_aed2c154.plan.md) and only that phase. As originally planned this was ~5 engineer-days; with the in-flight worktree already covering stages 1.1–1.5, the remaining work fits in ~1 ED (Stage 1.0 land + Stage 1.6 leftovers + Stage 1.5 verification nightly).

## Precision audit (2026-05-22)

A critical review against the live worktree found that stages 1.1–1.5 are implemented but uncommitted. Evidence:

- [docs/glossary.md](../../docs/glossary.md) — 288 lines, sections §1–§6, exists untracked.
- [capabilities.yaml](../../capabilities.yaml) — every one of the 12 cells has all 9 new metadata fields populated; `python3 tests/tools/check_capabilities.py` passes for all 12 with the default `--strict-metadata`.
- [golden/kernels/](../../golden/kernels/) — every one of the 10 kernels carries a "Cell metadata (mirrors capabilities.yaml; do not drift)" block and a "Non-obvious constraints (Phase 1.5)" block; [tests/unit/tools/test-golden-header.sh](../../tests/unit/tools/test-golden-header.sh) (untracked) verifies the header ↔ cell mapping and passes.
- [tests/tools/check_capabilities.py](../../tests/tools/check_capabilities.py) — lines 203–386 implement `_check_cell_metadata` exactly as Stage 1.4 described, with `_ALLOWED_SHAPE_REGIME`, `_ALLOWED_TAIL_BEHAVIOR`, `_ALLOWED_PARTITIONING`, `_KNOWN_UNSUPPORTED_REGIMES`, cross-field rules, and `--strict-metadata` / `--no-strict-metadata` flags. Line 207 already references "Capability cell metadata schema (Phase 1)" in docs/evaluation-methodology.md.

Conclusion: Phase 1's design and code work is complete; what is missing is (a) the commit + PR, (b) the ReLU coverage_note decision (Stage 1.1 noted it but no edit lands in capabilities.yaml or glossary.md), and (c) wiring the new tests into pr-gate.

## Outcome

After this sprint, every cell in [capabilities.yaml](../../capabilities.yaml) advertises *what it actually claims to prove* — the shape regime, the reduce axis, the accumulator dtype, the tail-handling mechanism, the partitioning strategy, and the regimes that are explicitly out of scope. [docs/glossary.md](../../docs/glossary.md) is the single source of truth for the terminology. Every golden kernel has a non-obvious-constraint header that the agent can read alongside the code. [tests/tools/check_capabilities.py](../../tests/tools/check_capabilities.py) refuses cells with missing or contradictory metadata.

## Stage 1.0 — Land the in-flight Phase 1 worktree (~0.5 ED) — NEW

This is the "Group B" PR referenced from [phase_0_protocol_aware_harness.plan.md](phase_0_protocol_aware_harness.plan.md) Stage 0.0. The exact file list lives there; the shape of the PR is:

1. Commit A — `golden/kernels/*.py` headers (10 files, no behavioral change). CI green via the existing simulator gate.
2. Commit B — [capabilities.yaml](../../capabilities.yaml) Phase 1 metadata additions for all 12 cells (separate hunk from the `prompt_variants.minimal` additions in Phase 0's Group A).
3. Commit C — [tests/tools/check_capabilities.py](../../tests/tools/check_capabilities.py) `_check_cell_metadata` + `_check_prompt_variants` + the `--strict-metadata` / `--no-strict-metadata` / `--soft-runtime` flags.
4. Commit D — [docs/glossary.md](../../docs/glossary.md) + [tests/unit/tools/test-golden-header.sh](../../tests/unit/tools/test-golden-header.sh).

Sequencing matters: commit A and B *must* land before commit C, otherwise `--strict-metadata` (the default) hard-fails the gate on the not-yet-populated cells. The escape hatch is `--no-strict-metadata` during the partial-rollout window — already implemented and documented in the docstring of check_capabilities.py.

PR pre-merge checklist:

```bash
python3 tests/tools/check_capabilities.py            # default strict; must PASS
bash tests/unit/tools/test-golden-header.sh           # must PASS
git diff capabilities.yaml | grep -E 'tail_behavior|partitioning|shape_regime' | head
```

Acceptance: Phase 1 PR merged; main branch carries every Phase 1 deliverable.

## Stage 1.1 — Design schema + glossary + ReLU scope (~0 ED) — VERIFY ONLY

[DONE in worktree, except ReLU coverage_note]. Verify during Stage 1.0 PR review.

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

## Stage 1.6 — Close precision leftovers (~0.25 ED) — NEW

Two items that the original Stage 1.1 + 1.4 plans called for but did not actually land in the worktree.

### 1.6.a — ReLU coverage_note + glossary paragraph

The Stage 1.1 design says: *"Deliverable: a one-line decision note appended to capabilities.yaml under the `abs` operation (`coverage_note: \"relu covered by representative_of from abs\"`) and a paragraph in docs/glossary.md §6."* Neither edit is in the worktree. Add:

- In [capabilities.yaml](../../capabilities.yaml), at the `abs` operation level (sibling of `tier:` and `representative_of:`):

  ```yaml
  - name: abs
    tier: elementwise
    representative_of: [exp, log, sqrt, relu, erf, sin, cos, neg, ceil, floor, rsqrt, tanh]
    coverage_note: |
      ReLU is intentionally covered as part of abs.representative_of rather than
      as its own cell: the generative challenge (single asc2.<op> template
      substitution) is identical, so a standalone ReLU cell would not
      surface any new evaluation signal. Revisit if a generative failure
      shows the model treats relu/abs as semantically distinct.
  ```

- In [docs/glossary.md](../../docs/glossary.md) §6 (or a new §7 "Coverage policy"), add a short paragraph anchoring the decision and pointing back to the `abs` op.

### 1.6.b — Wire test-golden-header.sh into pr-gate

[tests/unit/tools/test-golden-header.sh](../../tests/unit/tools/test-golden-header.sh) passes against the current worktree but is not invoked by any wrapper. Without that wiring, a stale header (e.g. someone edits the kernel but forgets the docstring) lands with a green pr-gate. Add the invocation to `tests/unit/tools/run-tests.sh` (or the local equivalent the pr-gate calls), alongside the new `test-protocol-derivation.sh` and `test-baseline-agents-md.sh` from Phase 0 Stage 0.8.

Acceptance: pr-gate fails when a Phase 1 metadata field is edited in capabilities.yaml without a matching docstring update; one local synthetic-drift test demonstrates the catch.

## Definition of done for Phase 1

- Stage 1.0 PR (Group B) landed on the active branch.
- 12/12 cells in [capabilities.yaml](../../capabilities.yaml) have all 9 new metadata fields populated.
- [docs/glossary.md](../../docs/glossary.md) covers every term used in the new metadata + the standard pyasc/asc2 vocabulary.
- 10/10 goldens in [golden/kernels/](../../golden/kernels/) carry a non-obvious-constraint header that mirrors the cell's metadata.
- [tests/tools/check_capabilities.py](../../tests/tools/check_capabilities.py) enforces the new contract (`--strict-metadata` on by default in pr-gate).
- [tests/unit/tools/test-golden-header.sh](../../tests/unit/tools/test-golden-header.sh) wired into pr-gate (Stage 1.6.b).
- ReLU scope decision documented as `coverage_note` + glossary paragraph (Stage 1.6.a).
- One full nightly green: 12/12 cells `gen_confirmed`, no behavior regression.

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
