---
name: Phase 2 prompt methodology
overview: "Concrete sprint plan (~4 engineer-days) for Phase 2 of the quarterly roadmap. Publishes docs/prompt-template.md as the canonical slot order for every capability prompt, rewrites all 12 cells' prompt + prompt_variants.minimal + prompt_variants.guided to read the new Phase 1 metadata fields in one consistent vocabulary, adds an examples_policy field per cell, refreshes the prompt-methodology section of pyasc-codegen-workflow, and runs one nightly to confirm no behavior regression."
todos:
  - id: p2-1-prompt-template
    content: "Stage 2.1: Author docs/prompt-template.md defining the 13 required slots (operator semantics, input shapes, output shapes, dtype, layout, axis, tiling constraints, padding/tail behavior, accumulator dtype, numerical tolerance, platform, build/run command, expected evidence artifact); reference Phase 1 docs/glossary.md vocabulary."
    status: pending
  - id: p2-2-rewrite-cells
    content: "Stage 2.2: Rewrite prompt, prompt_variants.minimal, prompt_variants.guided for all 12 cells in capabilities.yaml to follow the template; cells with workaround content (matmul torch.Tensor note, gelu/f32 Tile-on-left rule, rms_norm dispatcher contract) get re-labeled as oracle-guided per evaluation-methodology.md."
    status: pending
  - id: p2-3-examples-policy
    content: "Stage 2.3: Add examples_policy field per cell (declares whether agent may see golden_kernels, golden_docs, external_web during the run); validate in check_capabilities.py; default to {golden_kernels: false, golden_docs: false, external_web: false} for every cell."
    status: pending
  - id: p2-4-skill-update
    content: "Stage 2.4: Update skills/pyasc-codegen-workflow/SKILL.md with a 'Prompt methodology' section pointing at docs/prompt-template.md + docs/glossary.md; ensure phase-0/phase-1 workflow steps reference the canonical vocabulary."
    status: pending
  - id: p2-5-verify-nightly
    content: "Stage 2.5: Re-run pr-gate locally; dispatch one nightly to confirm 12/12 cells stay gen_confirmed under the rewritten prompts; if any cell flips, classify the regression as F1/F2/F4/F5 per evaluation-methodology.md failure taxonomy and fix before scaling Phase 3."
    status: pending
isProject: false
---

# Phase 2 — Prompt methodology

Drills into Phase 2 of [pyasc_skill_stack_quarterly_roadmap_aed2c154.plan.md](pyasc_skill_stack_quarterly_roadmap_aed2c154.plan.md) and only that phase. Sized at ~4 engineer-days across ~1 week. Depends on Phase 1's [docs/glossary.md](../../docs/glossary.md) (vocabulary) and per-cell metadata (the prompts read it).

## Outcome

After this sprint, every prompt in [capabilities.yaml](../../capabilities.yaml) reads from one canonical template, uses one canonical vocabulary, and explicitly declares its `examples_policy`. The agent always sees the same slot order across cells; the dashboard can render the prompt diff between protocols meaningfully because every prompt is structurally comparable. Phase 3 can then quantify `P3−P2` and `P6−P4` without the "is this an oracle-guided prompt?" ambiguity blocking the analysis.

## Stage 2.1 — Author `docs/prompt-template.md` (~0.5 ED)

Define the canonical slot order. The template is *positional* (slots in this order, every time) and *labeled* (each slot has a short header). Slots are the union of notes §2.2 with one practical addition (`evidence artifact`):

1. Operator semantics — one-sentence definition, with formula where applicable.
2. Input shapes — concrete list, or `runtime_size_only`.
3. Output shape — concrete shape, or `same_as_input`.
4. Dtype — `float16 | float32`; output dtype if different.
5. Layout — `contiguous` unless otherwise.
6. Axis — for reductions; `null` otherwise.
7. Tiling constraints — `TILE_SIZE` value, `CORE_NUM`, partitioning mode (`row_per_core | tile_per_core | block_grid | host_dispatcher`).
8. Padding / tail behavior — `aligned_only | host_pad | mask | real_shape | host_dispatcher | unsupported`.
9. Accumulator dtype — `null | float16 | float32`.
10. Numerical tolerance — `atol` and `rtol`, plus the reference computation.
11. Platform — `Ascend950PR_9599` for every cell today.
12. Build/run command — the exact one-liner the agent should produce.
13. Expected evidence artifact — `kernel.py`, plus `design.md` / `verification.md` for skills-on runs.

Per-variant rules (from [docs/evaluation-methodology.md](../../docs/evaluation-methodology.md) §"Prompt-variant labeling rules"):

- `minimal`: slots 1, 2, 3, 4 only. No API hints, no workaround, no example.
- `guided`: all 13 slots, but no oracle content.
- `oracle-guided`: anything `guided` plus workarounds for known backend bugs, exact tiling, golden-derived clues. The current `matmul/f16` guided prompt (`Reference golden/kernels/matmul_f16.py for the proven pattern`) and the `gelu/f32` Tile-on-left rule are oracle-guided today; Phase 2 re-labels them honestly rather than removing the content.
- `human-assisted`: any manual prompt edit. Not stored per-cell; emitted by the harness when a human intervenes mid-run.

Deliverable: [docs/prompt-template.md](../../docs/prompt-template.md) (~80 lines) with one worked example for `abs/f16` covering all four variants. Cross-link to [docs/glossary.md](../../docs/glossary.md).

## Stage 2.2 — Rewrite all 12 cells' prompts (~2 ED)

Per-cell rewrite, following the template. Order: simplest first.

### Tier 0 — `abs/{f16,f32}`, `add/f16`

The current `prompt_variants.guided` for `abs/f16` is already close. Rewrite for slot completeness:

```yaml
prompt_variants:
  minimal: |
    Implement abs/float16 for shapes [1,128], [4,2048], [32,4096].
    Output shape: same as input. Semantics: out = |x|.
  guided: |
    Operator: abs (out = |x|). Input shapes: [1,128], [4,2048], [32,4096].
    Output shape: same_as_input. Dtype: float16. Layout: contiguous.
    Tiling: TILE_SIZE divides each shape evenly; partition tile_per_core
    via asc2.range(asc2.block_idx(), num_tiles, asc2.block_num()).
    Tail: aligned_only — no tail logic required. Accumulator: null.
    Tolerance: atol=1e-3, rtol=1e-3 against np.abs(x).
    Platform: Ascend950PR_9599. Build/run: python kernel.py -r Model.
    Evidence: kernel.py + verification.md (skills-on: + design.md).
```

Same template for `abs/f32` and `add/f16` with the appropriate slot values.

### Tier 1 — `reduce_sum/{f32,f16}`

Slot 7 (tiling) explicitly captures `row_per_core` and the OUT_PAD wrapping; slot 8 captures `host_pad`; slot 9 captures `float32` accumulator. The current "use atol=2.0, rtol=5e-2" boilerplate becomes slot 10.

### Tier 2 — `gelu/{f16,f32}`, `leaky_relu/f16`

`gelu/f32`'s current `prompt_variants.guided` contains the Tile-on-left workaround (`x_cubed * 0.044715, NOT 0.044715 * x_cubed`). Per the labeling rules this is **oracle-guided content**. Two changes:

- Add `prompt_variants.oracle_guided` as a new variant key carrying the workaround.
- The `guided` variant for `gelu/f32` keeps the tanh/Padé formula (a correctness specification, not a workaround) but drops the `__rmul__` AttributeError commentary.

### Tier 3 — `softmax/f16`

Slot 7 captures `row_per_core` with the `block_size = ceildiv(num_rows, CORE_NUM)` partitioning. Slot 8 captures `aligned_only` and `unsupported_regimes: [split_row]`.

### Tier 3 — `matmul/f16`

The current `guided` says "Reference golden/kernels/matmul_f16.py for the proven pattern" — that is **oracle-guided** (golden-derived clue). Move it to a new `prompt_variants.oracle_guided`. The `guided` variant retains the cube-unit, L0A/L0B, torch.Tensor requirement (correctness facts) but drops the "reference the golden" pointer.

### Tier 3 — `rms_norm/{f16,f32}`

The current `guided` prompt explicitly names KernelRmsNormRegBase / KernelRmsNormRegBaseSplitD (CANN reference). That is **oracle-guided** (reference-derived clue). Same split: move CANN-reference naming to `oracle_guided`; keep the dispatcher contract (`full_row when num_cols * dtype_bytes <= 64KB and num_cols % 8 == 0, otherwise split_d`) in `guided` because it is the operator's correctness specification at this cell.

Deliverable: all 12 cells' `prompt`, `prompt_variants.minimal`, `prompt_variants.guided`, and (where applicable) `prompt_variants.oracle_guided` rewritten in [capabilities.yaml](../../capabilities.yaml).

## Stage 2.3 — `examples_policy` field per cell (~0.5 ED)

Add an additive field per cell:

```yaml
examples_policy:
  task_prompt: true             # always true
  glossary: true                # docs/glossary.md is always allowed
  golden_kernels: false         # default: forbidden in P2/P3/P4/P6
  golden_docs: false            # default: forbidden
  external_web: false           # default: forbidden
  human_hints: false            # default: forbidden
```

The aggregator already records `protocol.allowed_context` per evidence file (Phase 0); `examples_policy` is the *declared* policy per cell, while `protocol.allowed_context` is the *actual* allowed set per run. The two must match — [tests/tools/check_capabilities.py](../../tests/tools/check_capabilities.py) cross-checks them after the nightly.

For oracle-guided variants the policy is documented as a separate row:

```yaml
prompt_variants:
  oracle_guided:
    prompt: "..."
    examples_policy:
      golden_kernels: true     # explicitly allowed for this variant
```

Deliverable: `examples_policy` populated for all 12 cells; checker validates presence; default values applied to existing cells.

## Stage 2.4 — Update `pyasc-codegen-workflow` SKILL (~0.5 ED)

In [skills/pyasc-codegen-workflow/SKILL.md](../../skills/pyasc-codegen-workflow/SKILL.md), add a new section "Prompt methodology" between the workflow overview and Phase 0:

- Point at [docs/prompt-template.md](../../docs/prompt-template.md) as the canonical structure for every prompt the agent reads or writes.
- Point at [docs/glossary.md](../../docs/glossary.md) for the vocabulary.
- Spell out the labeling rule (`minimal | guided | oracle-guided | human-assisted`) so the agent does not silently introduce oracle content into a `guided` prompt during re-prompting.
- Cross-link to the `examples_policy` field so the agent knows what it may *not* read.

The other skills in [skills/](../../skills/) need no edits in this sprint — they reference operator APIs, not prompt structure.

Deliverable: one SKILL.md edit; one local L1 test ([tests/run-tests.sh](../../tests/run-tests.sh) `--fast`) green.

## Stage 2.5 — Verify + nightly no-op (~0.5 ED)

Run the rewritten prompts through one full nightly. Expected: 12/12 cells stay `gen_confirmed` — the rewrite is structural, not semantic. The risk is that re-ordering slots silently triggers a different model behavior (slot order matters for some models). The acceptance gate is:

- Local pr-gate green.
- Local L1 + L2 tests green.
- One dispatched nightly with `tier=nightly`: 12/12 cells pass on P6 (current `cloud-default + on`).
- Any cell that flips is classified per [docs/evaluation-methodology.md](../../docs/evaluation-methodology.md) §"Failure taxonomy" (F1 if prompt misunderstood, F4 if tiling mismatch, F5 if dtype/precision drift). If the failure is F1, revert the slot order for that cell and document the model's slot-order sensitivity in [docs/prompt-template.md](../../docs/prompt-template.md).

Deliverable: one nightly green; one short "slot-order sensitivity" addendum to [docs/prompt-template.md](../../docs/prompt-template.md) if any cell required slot re-ordering.

## Definition of done for Phase 2

- [docs/prompt-template.md](../../docs/prompt-template.md) published, cross-linked to [docs/glossary.md](../../docs/glossary.md).
- 12/12 cells in [capabilities.yaml](../../capabilities.yaml) have `prompt`, `prompt_variants.minimal`, `prompt_variants.guided` rewritten in the canonical 13-slot order.
- All workaround content moved to `prompt_variants.oracle_guided` for the 4 cells that had it (`gelu/f32`, `matmul/f16`, `rms_norm/f16`, `rms_norm/f32`).
- 12/12 cells declare `examples_policy`; checker enforces.
- [skills/pyasc-codegen-workflow/SKILL.md](../../skills/pyasc-codegen-workflow/SKILL.md) has the "Prompt methodology" section.
- One full nightly green on P6; no F1/F2/F4 regressions.

## Risks specific to Phase 2

- **Slot-order sensitivity.** Some models behave differently when slots reorder, even with identical content. Mitigation: pick one slot order that mirrors how the existing high-success prompts (`abs/f16`, `add/f16`, `gelu/f16`) flow today, and document any per-model deviation in [docs/prompt-template.md](../../docs/prompt-template.md).
- **Oracle-content extraction may shrink `guided` quality.** Moving the "reference golden/kernels/matmul_f16.py" hint out of `guided` into `oracle_guided` will likely reduce `matmul/f16` pass-rate at P3/P6 in the next nightly. That is the intended honest measurement, not a regression — Phase 3 will quantify the gap. Mitigation: communicate this expectation in the commit message so a nightly drop is not mis-classified as a bug.
- **`examples_policy` mismatch with the harness.** Stage 2.3's cross-check assumes the harness already respects `protocol.allowed_context`; that contract was wired in Phase 0. Verify the wiring before scaling.
- **Phase 1 dependency.** Phase 2 reads Phase 1's metadata to populate slots 6, 7, 8, 9 of the template. If Phase 1 is not fully merged, Phase 2 cannot start. Hard prerequisite.

## Deferred from Phase 2 (intentionally)

- **Per-model prompt variants.** Stage 2.5 may discover slot-order sensitivity; if so, model-specific prompt variants are a Phase 6 concern.
- **Auto-generation of prompts from metadata.** The natural next step (Phase 1 metadata → Phase 2 prompt slots is fully mechanical) is an attractive automation but defers to Phase 7 once schema v4 lands.
- **Localized prompts.** All prompts stay English-only this sprint.
- **Prompt fuzzing / sensitivity studies.** Methodology question (notes §3.4 "skills on/off token comparison") that lives in Phase 3 and Phase 6.
