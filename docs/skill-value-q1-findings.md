# Skill stack value — Q1 findings (Phase 3 Stage 3.6)

**Headline:** the pyasc skill stack does NOT add measurable
static-verify pass-rate over an AGENTS.md baseline at the current
12-cell scope, while costing ~2.2× more tokens. The protocol decomposition
isolates this clearly:

| Comparison | Δ pass-rate | 95% CI (Newcombe) | Δ tokens | Interpretation |
|---|---:|---:|---:|---|
| `P3 − P2` (value of guided prompting) | **+58.4 pp** | [+19.7, +79.0] | +31.4 % | **significant** |
| `P4 − P3` (value of vendored AGENTS.md) | **+33.3 pp** | [+2.2, +60.9] | −13.5 % | **significant**, cheaper |
| `P6 − P4` (value of pyasc skill stack on top of AGENTS.md) | **0.0 pp** | [−24.3, +24.3] | +223.9 % | **not significant**, expensive |

CIs are 95% Newcombe (Method 10) intervals on the proportion difference,
built from Wilson single-proportion intervals — robust at 0/n and n/n
edges where Wald collapses. CI on `P6 − P4` includes zero: at n=12
per arm, even a +24 pp true effect is plausible — but so is a −24 pp
regression. The data does not yet distinguish them.

## Setup

- **Date:** 2026-05-26, single-night snapshot (Stage 3.4 three-night
  median sweep is the follow-up that turns this snapshot into a
  finding with temporal CIs).
- **Profile:** `cloud-default` (Alibaba DashScope / `dashscope/glm-5`).
- **Harness:** `opencode 1.15.10` via
  [`tests/tools/collect_generative_evidence.py`](../tests/tools/collect_generative_evidence.py)
  on the local machine (no GitHub Actions runner). Workspace
  permissions per [`docker/opencode-profiles/cloud-default.json`](../docker/opencode-profiles/cloud-default.json).
- **Verification level:** *static only* (CANN simulator unavailable
  locally). Pass = `static_verify=="pass"` AND `score.accepted` AND
  `semantic_check.passed` AND kernel produced. Runtime/numerical
  agreement on the C310 simulator is NOT in this dataset and is left
  to the future merge-gate run.
- **Sample:** 12 cells × 4 protocols = 48 evidence files;
  `--max-attempts 1` (one attempt per leg, no retry); 300 s timeout.
- **Aggregation:** [`tests/tools/compare_skills_value.py`](../tests/tools/compare_skills_value.py)
  → `by_profile.cloud-default.{by_protocol, deltas_pp, deltas_pp_history}`.

## Per-cell breakdown

✓ = static-verify-pass with kernel emitted; ✗ = no kernel OR
static/semantic/score failure.

| Cell | P2 (minimal) | P3 (guided) | P4 (+AGENTS.md) | P6 (skills on) |
|---|---|---|---|---|
| `abs/float16` | ✗ (4,503t) | ✗ (18,021t) | ✓ (34,116t) | ✓ (107,308t) |
| `abs/float32` | ✗ (8,244t) | ✗ (4,276t) | ✓ (23,833t) | ✓ (79,492t) |
| `add/float16` | ✗ (5,571t) | ✗ (6,591t) | ✓ (19,451t) | ✓ (116,178t) |
| `reduce_sum/float32` | ✗ (80,610t) | ✓ (20,310t) | ✓ (51,603t) | ✓ (64,299t) |
| `reduce_sum/float16` | ✗ (6,331t) | ✓ (5,923t) | ✓ (15,833t) | ✓ (77,662t) |
| `gelu/float16` | ✗ (84,585t) | ✓ (77,313t) | ✓ (41,386t) | ✓ (102,263t) |
| `gelu/float32` | ✗ (4,527t) | ✗ (4,093t) | ✓ (28,607t) | ✓ (73,936t) |
| `leaky_relu/float16` | ✗ (14,949t) | ✓ (71,607t) | ✓ (25,994t) | ✓ (84,416t) |
| `softmax/float16` | ✓ (56,314t) | ✓ (65,437t) | ✓ (15,882t) | ✓ (95,534t) |
| `matmul/float16` | ✗ (16,339t) | ✓ (60,103t) | ✓ (17,543t) | ✓ (143,755t) |
| `rms_norm/float16` | ✗ (19,198t) | ✓ (45,598t) | ✓ (65,784t) | ✓ (123,772t) |
| `rms_norm/float32` | ✗ (14,975t) | ✓ (36,173t) | ✓ (19,395t) | ✓ (95,451t) |

Token totals per protocol:

| Protocol | Σ tokens | mean tokens / cell |
|---|---:|---:|
| P2 | 316 K | 26 346 |
| P3 | 415 K | 34 620 |
| P4 | 359 K | 29 952 |
| P6 | 1 164 K | 97 005 |

## Failure-mode mix per protocol

The 11 failing P2 cells split roughly into three patterns:

- **F10 — no kernel emitted.** The minimal 4-slot prompt did not give
  the agent enough structure to commit to a single API. Affected: most
  elementwise cells (`abs/f{16,32}`, `add/f16`, `gelu/{f16,f32}`,
  `leaky_relu/f16`, `rms_norm/{f16,f32}`).
- **F0 — wrong host harness.** A kernel was emitted but the host
  driver did not match the cell's documented shape regime
  (`reduce_sum`, `matmul/f16` — small subset).
- **F7 / F8 — incomplete API surface.** The agent guessed at `asc2`
  imports rather than reading the local skills. Tail behavior, axis,
  and accumulator dtype were unset — slots 6–9 from the canonical
  template are required for these cells.

P3 (guided) recovers 7 cells; the 4 remaining failures (all tier-0
elementwise: `abs/{f16,f32}`, `add/f16`, `gelu/f32`) share a single
root cause: without an AGENTS.md to anchor the `--platform
Ascend950PR_9599 -r Model` invocation, the agent generates kernels
that target the wrong platform or skip the `@asc2.jit` decorator.
P4 (vendored AGENTS.md) closes all four. **AGENTS.md value is
concentrated in the tier-0 cells, not the complex ones.**

P6 (skills on) does not flip any cell. The skill stack produces extra
artifacts — `design.md`, `self_review.md`, `acceptance_review.md` —
but the kernel that lands is structurally equivalent to the P4
output. The +0 pp delta IS the finding.

## Skill stack value, isolated per-cell

| Cell | P4 status | P6 status | Skill stack effect |
|---|---|---|---|
| `abs/float16` | ✓ | ✓ | no flip |
| `abs/float32` | ✓ | ✓ | no flip |
| `add/float16` | ✓ | ✓ | no flip |
| `reduce_sum/float32` | ✓ | ✓ | no flip |
| `reduce_sum/float16` | ✓ | ✓ | no flip |
| `gelu/float16` | ✓ | ✓ | no flip |
| `gelu/float32` | ✓ | ✓ | no flip |
| `leaky_relu/float16` | ✓ | ✓ | no flip |
| `softmax/float16` | ✓ | ✓ | no flip |
| `matmul/float16` | ✓ | ✓ | no flip |
| `rms_norm/float16` | ✓ | ✓ | no flip |
| `rms_norm/float32` | ✓ | ✓ | no flip |

**0 of 12 cells flip on the skill intervention at this verification
fidelity.** Any value the skill stack provides is currently invisible
to static verification.

## Budget realized vs estimated

Stage 3.2 ([`docs/perf-methodology/phase-3-budget.md`](perf-methodology/phase-3-budget.md))
projected the 4-leg matrix at **1.31× the 2-leg legacy**, using
`abs/float16` as the basis cell.

| Metric | Projected (abs/f16-basis) | Realized (12-cell matrix) |
|---|---:|---:|
| Σ tokens P2+P3+P4+P6 / cell | 163 948 | 187 826 (avg over 12) |
| Σ tokens P3+P6 / cell | 125 329 | 132 626 (avg over 12) |
| Matrix multiplier (4-leg / 2-leg) | 1.31× | 1.42× |
| Σ tokens / nightly (12 cells) | 1.97 M | 2.25 M |

Realized multiplier 1.42× is +8 % over projection; still well below
the 2.5× weekly-split guardrail. The largest contributor to the gap
is the tier-2/3 cells (`matmul`, `rms_norm`, `softmax`) with P6
spends 2-3× larger than the abs/f16 basis. **Decision holds: 4 legs
nightly, no schedule split.**

## Anomalies flagged for Phase 5 / 6

1. **`softmax/float16` passes at P2.** It is the only cell where a
   4-slot minimal prompt produces a static-passing kernel; even the
   harness's "no AGENTS.md, no skills, no API hints" leg succeeds. The
   cell's `prompt_variants.minimal` is overspecified — either the
   minimal slot population is too generous, or `softmax` is genuinely
   trivial for `glm-5` on cloud-default. Re-audit Stage 2.2's minimal
   variant on this cell before treating the P2 row as representative.
   Owner: pyasc-skill-stack-author.
2. **`reduce_sum/float32` P2 spend is 80 K tokens for no kernel.** The
   minimal prompt sent the agent on a long tool-use spiral. The other
   P2 spends are ≤ 20 K. Investigate whether `reduce_sum` triggers a
   loop in the `glm-5` toolchain or whether the spec needs more
   constraint at the minimal level. Owner: harness-runtime.
3. **`gelu/float16` P3 spend is 77 K, then drops to 41 K at P4.**
   Adding the AGENTS.md baseline made the kernel CHEAPER, not more
   expensive. This is the only cell where `P4 < P3`. Likely the
   AGENTS.md gives the agent the right tanh-vs-erf branch directly,
   short-circuiting an exploration the bare-guided prompt forces.
   Owner: docs/evaluation-methodology.md author.
4. **Static-verify-only blindness on `oracle_guided` cells.** The
   four cells with documented workarounds (`gelu/f32`, `matmul/f16`,
   `rms_norm/{f16,f32}`) all pass at P4 and P6. The skill stack might
   make a difference at *runtime* verification but the static AST
   check does not exercise the workaround. Schedule a CANN
   simulator-enabled run on these four cells before publishing a
   finding on oracle-guided behaviour. Owner: docs/baseline / Phase 5.

## Caveats

- **Single-night snapshot.** All 12 × 4 cells were run sequentially
  over ~41 minutes once tonight. Day-over-day model temperature is
  unsampled; the Newcombe CIs treat the 12 cells as 12 independent
  Bernoulli draws, which they are not (cells share an underlying
  agent quality signal). A three-night sweep (Stage 3.4) is needed
  to produce a CI on the night-over-night noise floor.
- **Static-only verification.** No CANN simulator was invoked. Cells
  reported as ✓ are not guaranteed to produce numerically-correct
  output on hardware. P4 = 100 % under static-only does NOT mean
  P4 = 100 % under simulator-enabled. The merge-gate's
  `simulator-verify` job (when re-enabled) will produce the matched
  runtime-verified version of this table.
- **`--max-attempts 1`.** One attempt per leg, no retry. A P2 leg
  that times out at 300 s is recorded as a single fail; it could
  succeed on a re-run. Stage 3.4 may bump `--max-attempts` to 2 or 3
  if the noise floor proves bound by attempt variance.
- **Local provider quotas.** dashscope/glm-5 has a per-key QPS limit;
  the matrix used 4-parallel local launches and no rate-limit was
  exceeded, but a CI runner with the same key and a different
  parallelism could see different timings.

## What this changes for the sprint

- **No-go for adding skill complexity until runtime verification is
  back online.** P6's +0 pp at static-only is the strong signal: the
  skill stack's measurable value, if any, lives in the runtime / golden-
  matching layer. Adding more skill content (Phase 5, Phase 6) without
  a runtime check that exercises it would compound the blind-spot.
- **AGENTS.md baseline is the new bar.** Future nightly comparisons
  should foreground `P6 − P4` (skill value) over `P6 − P3` (any
  doc-or-skill value) so the dashboard's headline matches the
  experimental design.
- **The Newcombe CI is the publication unit.** A bare `+0 pp` is
  meaningless without `[−24.3, +24.3]`. The Stage 3.5 panel renders
  it; consumers of this report should always cite both.

## Cross-references

- Methodology: [`docs/evaluation-methodology.md` §"Comparisons of interest"](evaluation-methodology.md#comparisons-of-interest).
- Protocol matrix: [`docs/evaluation-methodology.md` §"Protocol-axis CI mapping (Phase 0)"](evaluation-methodology.md#protocol-axis-ci-mapping-phase-0).
- Prompt template: [`docs/prompt-template.md`](prompt-template.md).
- Budget projection: [`docs/perf-methodology/phase-3-budget.md`](perf-methodology/phase-3-budget.md).
- Aggregator: [`tests/tools/compare_skills_value.py`](../tests/tools/compare_skills_value.py).
- Dashboard panel: [`tests/tools/generate_dashboard.py`](../tests/tools/generate_dashboard.py) function `renderProtocolDecomp`.
- Evidence: `evidence/*-generative-cloud-default-{p2,p3,p4,p6}.json` (48 files).
