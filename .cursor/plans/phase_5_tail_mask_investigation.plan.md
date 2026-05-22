---
name: Phase 5 tail and mask investigation
overview: "Concrete sprint plan (~6 engineer-days) for Phase 5 of the quarterly roadmap. Replaces today's untested assumptions about asc2 tail/mask behavior with six tightly-scoped micro-probes (asc2.mask vector op, asc2.load(real_shape) partial load, partial asc2.store) on both Ascend910B1 and Ascend950PR_9599, publishes docs/tail-handling.md as the verified mechanism reference, updates skills/pyasc-api-patterns/SKILL.md with the verified patterns, revisits the rms_norm split_d path, and tightens shape_regime claims in capabilities.yaml where the spike showed unsupported behavior."
todos:
  - id: p5-1-probe-design
    content: "Stage 5.1: Probe design — write 6 single-purpose micro-kernels under tests/probes/tail/ for asc2.mask (aligned + tail-only + full-tile), asc2.load(real_shape) (aligned + partial), partial asc2.store (aligned + tail-only). Each probe has a numpy oracle, a runtime asserter, and a one-line verdict written to a verdict.txt."
    status: pending
  - id: p5-2-run-910b1
    content: "Stage 5.2: Run all 6 probes on Ascend910B1 inside the existing simulator container; capture verdict + tail-element value(s) + agent.timeout/exception status per probe; commit probe artifacts to evidence/probes/tail/Ascend910B1/."
    status: pending
  - id: p5-3-run-950pr9599
    content: "Stage 5.3: Run all 6 probes on Ascend950PR_9599 (the only CI platform); capture the same artifacts under evidence/probes/tail/Ascend950PR_9599/; note any platform-conditional behavior."
    status: pending
  - id: p5-4-analyze
    content: "Stage 5.4: Analyze the 12 verdicts; categorize each mechanism per platform as {works_as_expected, works_with_caveat, silently_wrong, raises_at_compile, raises_at_runtime, unsupported}; build a 6x2 mechanism-by-platform truth matrix as the source of truth for docs/tail-handling.md."
    status: pending
  - id: p5-5-write-doc
    content: "Stage 5.5: Write docs/tail-handling.md with: per-mechanism truth matrix, when-to-use guidance (when to mask, when to pad, when to reject), code snippets per platform, and an explicit 'do not use' list for mechanisms that fail silently."
    status: pending
  - id: p5-6-skill-and-cells
    content: "Stage 5.6: Update skills/pyasc-api-patterns/SKILL.md with the verified pattern; revisit golden/kernels/rms_norm_{f16,f32}.py split_d path against the new truth matrix; tighten capabilities.yaml shape_regime / tail_behavior claims where Phase 1 over-claimed; remove tail_behavior enum values that no platform supports (e.g., demote mask + real_shape if the spike showed they don't work today)."
    status: pending
isProject: false
---

# Phase 5 — Tail and mask investigation

Drills into Phase 5 of [pyasc_skill_stack_quarterly_roadmap_aed2c154.plan.md](pyasc_skill_stack_quarterly_roadmap_aed2c154.plan.md) and only that phase. Sized at ~6 engineer-days across ~1.5 weeks. Replaces conjectured tail/mask behavior in Phase 1 metadata with verified, platform-conditional truth.

## Outcome

After this sprint, the question "what is the right asc2 mechanism for partial tiles on this platform?" has a one-paragraph answer for each of three candidate mechanisms (`asc2.mask`, `asc2.load(real_shape)`, partial `asc2.store`) on each of two platforms (`Ascend910B1`, `Ascend950PR_9599`). The Phase 1 metadata schema's `tail_behavior` enum (`aligned_only | host_pad | mask | real_shape | host_dispatcher | unsupported`) is trimmed to remove values that no platform supports, and the cells that previously claimed `mask` or `real_shape` are corrected.

## Stage 5.1 — Probe design (~0.5 ED)

Six probes; each is a single-purpose kernel + a single-purpose host driver + a `verdict.txt`. Naming: `tests/probes/tail/<mechanism>_<case>.py`.

| Probe | Mechanism | Case | Oracle |
|---|---|---|---|
| `mask_aligned.py` | `asc2.mask` | shape divisible by TILE_SIZE | `np.where(mask, x*2, 0)` |
| `mask_tail_only.py` | `asc2.mask` | last tile is partial; only those elements masked off | same |
| `mask_full_tile.py` | `asc2.mask` | full tile, no tail; assert mask is the identity | same |
| `load_real_shape_aligned.py` | `asc2.load(real_shape=...)` | shape divisible by TILE_SIZE | `x.copy()` |
| `load_real_shape_partial.py` | `asc2.load(real_shape=...)` | last tile is shorter than `TILE_SIZE` | first `real_shape` elements copied; remainder undefined |
| `store_partial.py` | partial `asc2.store` | store fewer elements than the tile width | first N elements stored; remainder of GM unchanged |

Each probe's `verdict.txt` is one of `works_as_expected`, `works_with_caveat`, `silently_wrong`, `raises_at_compile`, `raises_at_runtime`, `unsupported`. The harness writes the verdict by comparing the kernel output to the numpy oracle:

- `works_as_expected`: kernel output matches oracle elementwise.
- `works_with_caveat`: matches oracle within tolerance but with non-trivial side conditions (e.g., requires TILE_SIZE multiple of 64). The caveat is recorded as a free-form string in `verdict.txt`.
- `silently_wrong`: kernel runs to completion, output differs from oracle without raising. This is the dangerous case.
- `raises_at_compile`: `@asc2.jit` compilation raises (with the exception text recorded).
- `raises_at_runtime`: simulator/runtime raises.
- `unsupported`: the platform does not expose the API at all.

Each probe takes 30–60 lines including the oracle. Order: write `mask_aligned.py` first as a sanity check; the others mirror its structure.

Deliverable: 6 probe files under [tests/probes/tail/](../../tests/probes/) plus a [tests/probes/tail/README.md](../../tests/probes/) describing the verdict vocabulary.

## Stage 5.2 — Run probes on `Ascend910B1` (~1 ED)

The existing `pyasc-sim:py3.11` Docker image contains both simulators. Run each probe:

```bash
for probe in tests/probes/tail/*.py; do
  python3.10 tests/tools/run_and_verify.py "$probe" \
    --mode simulator --platform Ascend910B1 --timeout 300 \
    --json > "evidence/probes/tail/Ascend910B1/$(basename $probe .py).json"
done
```

Capture per probe:

- Verdict string.
- Output sample around the tail boundary (first 8 elements before tail, all tail elements, first 8 after tail).
- Compile-time exception (if any).
- Runtime exception (if any).
- Wall time.

Commit the 6 evidence files to [evidence/probes/tail/Ascend910B1/](../../evidence/probes/).

Deliverable: 6 evidence files for `Ascend910B1`; a one-line summary in the Stage 5.4 spreadsheet.

## Stage 5.3 — Run probes on `Ascend950PR_9599` (~1 ED)

Same loop, swap the platform flag. This is the CI-blessed platform; verdicts here drive the dashboard reality.

Capture the same artifacts under [evidence/probes/tail/Ascend950PR_9599/](../../evidence/probes/).

If any probe behaves differently between platforms, both verdicts are kept — the doc renders them side-by-side.

Deliverable: 6 evidence files for `Ascend950PR_9599`.

## Stage 5.4 — Truth matrix (~1 ED)

Aggregate the 12 verdicts into a 6×2 matrix:

```
                       Ascend910B1        Ascend950PR_9599
mask_aligned           ?                  ?
mask_tail_only         ?                  ?
mask_full_tile         ?                  ?
load_real_shape_alig   ?                  ?
load_real_shape_part   ?                  ?
store_partial          ?                  ?
```

Per row, the resolution rule:

- Both cells `works_as_expected` → mechanism is portable; documentation says "use".
- One cell `works_as_expected`, the other `silently_wrong` → mechanism is platform-conditional and *dangerous* on the failing platform; documentation says "do not use on <platform>".
- Both cells `silently_wrong` → mechanism is unusable today; documentation says "avoid; use <fallback>".
- Any `raises_at_compile` → mechanism is feature-gated; documentation says "blocked until pyasc adds <feature>".
- All other combinations are documented verbatim with the caveat string.

The matrix is the canonical input to [docs/tail-handling.md](../../docs/tail-handling.md). Commit it as a YAML file at [evidence/probes/tail/matrix.yaml](../../evidence/probes/tail/matrix.yaml) so future probe sprints can diff against it.

Deliverable: 6×2 verdict matrix; one short writeup of any "silently_wrong" finding (this is the highest-priority signal for the dashboard).

## Stage 5.5 — `docs/tail-handling.md` (~1 ED)

Single page; readable in 5 minutes. Sections:

- §1 Vocabulary recap — point at [docs/glossary.md](../../docs/glossary.md) for `tail`, `mask`, `padding`, `real_shape`, `host_pad`, `host_dispatcher`.
- §2 Truth matrix — the Stage 5.4 6×2 matrix, rendered.
- §3 Per-mechanism guidance:
  - `asc2.mask`: when to use, when to avoid, code snippet, caveats.
  - `asc2.load(real_shape=…)`: same.
  - Partial `asc2.store`: same.
- §4 Fallback decision tree — given a shape that has a tail, which mechanism does the agent reach for, in what order? Default: aligned_only (host pad) → host_dispatcher (rms_norm pattern) → mask (if §2 says supported) → real_shape (if §2 says supported) → reject and surface to user.
- §5 Do-not-use list — explicit list of platform/mechanism combinations the spike found `silently_wrong`. This is the most important section: silent-wrong is the failure mode that destroys evidence trust.

The doc is reporting-only — it does not gate CI. But it is the input that Stage 5.6 reads.

Deliverable: [docs/tail-handling.md](../../docs/tail-handling.md) published; reviewed for accuracy against the Stage 5.4 matrix.

## Stage 5.6 — Skill + cell + golden tightening (~1.5 ED)

Now that the truth is known, fix everything that drifted from it.

### 5.6.1 Update `skills/pyasc-api-patterns/SKILL.md`

Add a "Tail handling" section that mirrors §3 of [docs/tail-handling.md](../../docs/tail-handling.md). The agent reads SKILL.md, not the docs/; the canonical examples need to live in both. Cross-link.

Remove any pre-existing tail-handling guidance from the skill that contradicts the spike. Specifically: the existing skill text under "Common Mistakes" that mentions `asc2.mask` must be reconciled with §5's do-not-use list.

### 5.6.2 Revisit `rms_norm` `split_d`

Read [golden/kernels/rms_norm_f16.py](../../golden/kernels/rms_norm_f16.py) and [_f32.py](../../golden/kernels/rms_norm_f32.py). The current implementation streams along D in fixed 64-element tiles with host-side zero padding. Cross-check:

- Is the host-side padding actually necessary on `Ascend950PR_9599`, given Stage 5.4's verdict for `asc2.load(real_shape=...)`?
- If the verdict was `works_as_expected`, simplify the kernel (drop the host pad, add `real_shape` to the last-tile load). Update the cell's `tail_behavior` from `host_dispatcher` to `host_dispatcher_plus_real_shape` or, if the new code is cleaner, to `real_shape`.
- If the verdict was `silently_wrong` or `raises_at_runtime`, leave the kernel alone and document why in the golden header (Phase 1 stage 1.3 already added a header; append a paragraph).

### 5.6.3 Tighten `capabilities.yaml` claims

Re-examine every cell whose Phase 1 `tail_behavior` was set optimistically:

- `softmax/f16` claimed `aligned_only` with `unsupported_regimes: [split_row]`. Cross-check whether the Stage 5.4 verdict for `mask` would change that claim. If `mask` works on `Ascend950PR_9599`, leave the cell unchanged but add a `future_tail_behavior: mask` note pointing at the Phase 8 backlog for adding a long-row split-row variant.
- For any cell with `tail_behavior: mask` or `tail_behavior: real_shape`, demote to a supported value if the spike showed the mechanism doesn't work. Right now no cell uses these values, but Phase 1 documented them as allowed enum entries; if Stage 5.4 invalidates them, update the schema in [docs/glossary.md](../../docs/glossary.md) and remove from the enum.

### 5.6.4 Enum cleanup

If any value in the `tail_behavior` enum is universally `silently_wrong` per Stage 5.4, remove it from the allowed set in [tests/tools/check_capabilities.py](../../tests/tools/check_capabilities.py). The probe artifacts under [evidence/probes/tail/](../../evidence/probes/) document the rationale.

Deliverable: skill updated; rms_norm goldens reviewed (possibly simplified); cell `tail_behavior` claims tightened; enum cleanup if applicable; one nightly green to confirm no regression.

## Definition of done for Phase 5

- 6 probes committed under [tests/probes/tail/](../../tests/probes/) with a README.
- 12 verdict evidence files committed under [evidence/probes/tail/](../../evidence/probes/).
- 6×2 matrix committed at [evidence/probes/tail/matrix.yaml](../../evidence/probes/).
- [docs/tail-handling.md](../../docs/tail-handling.md) published with the truth matrix, per-mechanism guidance, fallback decision tree, and do-not-use list.
- [skills/pyasc-api-patterns/SKILL.md](../../skills/pyasc-api-patterns/SKILL.md) "Tail handling" section in place.
- `rms_norm/{f16,f32}` goldens reviewed; either simplified or with a paragraph explaining why simplification is unsafe.
- Every cell's `tail_behavior` in [capabilities.yaml](../../capabilities.yaml) is consistent with the truth matrix.
- One full nightly green; no F4/F8 regressions traceable to a tail-handling change.

## Risks specific to Phase 5

- **Silent wrong is the dangerous finding.** If Stage 5.4 surfaces a `silently_wrong` row for a mechanism a current cell uses, the cell's existing `gen_confirmed` status is suspect. Mitigation: surface this as a P0 issue and re-run that cell's nightly under the new fallback before publishing the truth matrix.
- **Probe variance.** A single probe run may pass on noise. Mitigation: each probe runs 3 iterations; verdict is the majority outcome (or "noisy" if the iterations disagree).
- **Platform availability.** `Ascend910B1` simulator is no longer the CI target ([.github/workflows/ci.yml](../../.github/workflows/ci.yml) line 73). The probes still run on it because the Docker image contains both simulators, but CI does not enforce 910B1 verdicts going forward. Mitigation: document that `Ascend910B1` verdicts are reference-only; the cell `platform` field stays `Ascend950PR_9599`.
- **rms_norm regression.** Simplifying the `split_d` golden risks reintroducing the very C310 zeroing bug fixed in [.cursor/plans/rms_norm_platform_fix_and_gelu_tanh_9c16b5ec.plan.md](rms_norm_platform_fix_and_gelu_tanh_9c16b5ec.plan.md). Mitigation: any simplification must run through one full nightly before landing on main; if the rms_norm cell flips, revert.
- **`Phase 1`-`Phase 5` ordering inversion.** Phase 1 documented `mask` and `real_shape` as allowed enum values; Phase 5 may invalidate them. The enum cleanup in Stage 5.6.4 closes the loop, but a careless reader of [docs/glossary.md](../../docs/glossary.md) between Phase 1 and Phase 5 lands could trust an invalidated enum value. Mitigation: Phase 5 lands a single "this enum value was invalidated by Stage 5.4 — see evidence/probes/tail/" note in the glossary entry instead of silently removing the value.

## Deferred from Phase 5 (intentionally)

- **New cells for split-row softmax or other long-row variants.** Phase 5 documents whether the mechanism works; Phase 8 decides whether to add a new cell.
- **Performance comparison across mechanisms.** Once the simulator emits runtime numbers (Phase 7+), revisit which mechanism is cheapest.
- **Cross-dtype probes.** This sprint runs f16 only; f32 probes are a follow-up if any verdict differs.
- **Mask masking-condition variants (per-element vs per-tile).** Phase 5 tests the simplest mask pattern; more elaborate mask conditions stay deferred.
- **PR for upstream pyasc to add missing constructs.** If a mechanism's verdict is `raises_at_compile` because asc2 lacks a feature, file an upstream issue but do not block this sprint on its resolution.
