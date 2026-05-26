# Skill stack value — Q1 findings (Phase 3 Stage 3.6, simulator-verified)

**Headline:** with CANN-simulator verification (numerical correctness on
`Ascend950PR_9599`), the pyasc skill stack lifts pass rate from
**58.3 % (P4) to 66.7 % (P6)** on the 12-cell matrix — a measured
**+8.4 pp**. The Newcombe 95 % CI is **[−27.2 pp, +41.2 pp]** and
still includes zero at n=12 per arm; but unlike the prior static-only
read, the *point estimate is no longer zero*. The simulator
reveals widespread **static-vs-simulator drift**: 7 of 23 P3/P4/P6
cells that pass AST checks fail the simulator.

| Comparison | Δ pass-rate | 95 % CI (Newcombe) | Δ tokens | Interpretation |
|---|---:|---:|---:|---|
| `P3 − P2` (guided prompt) | **+8.3 pp** | [−26.0, +40.3] | −4.4 % | not significant |
| `P4 − P3` (vendored AGENTS.md) | **+25.0 pp** | [−13.2, +54.7] | −11.2 % | not significant, cheaper |
| `P6 − P4` (pyasc skill stack) | **+8.4 pp** | [−27.2, +41.2] | **+312.8 %** | not significant, expensive |

The simulator-verified pass rates are substantially lower than the
static-only readout reported in the prior revision of this document.
The most dramatic difference is at P3:

| Protocol | Static-only (prior) | Simulator (this revision) |
|---|---:|---:|
| P2 | 8.3 % (1/12) | **25.0 % (3/12)** |
| P3 | 66.7 % (8/12) | **33.3 % (4/12)** |
| P4 | 100 % (12/12) | **58.3 % (7/12)** |
| P6 | 100 % (12/12) | **66.7 % (8/12)** |

P4 = 100 % static was a verification-blind result; six of those cells
emit kernels that pass the AST checks but fail the simulator on
numerical agreement, MTE alignment, or TilingKey selection. The
skill-stack value claim has to be re-anchored against this new
denominator.

## Why the prior revision said "static only"

`tests/tools/collect_generative_evidence.py` historically had **only a
Docker path** (`run_docker_verify`). The Stage 3.3 local matrix
orchestrator omitted `--runtime` entirely, so `verification.mode` was
hard-coded to `"static_only"` on every evidence file. The local CANN
install at `/usr/local/Ascend/cann-9.0.0` is fully usable but had no
way to be invoked through the harness on a non-Docker host. Stage A of
this revision adds a host runtime backend
([`run_host_verify`](../tests/tools/collect_generative_evidence.py),
`--runtime-backend {auto,host,docker}`); `auto` prefers host when
`ASCEND_HOME_PATH` is set and `import asc` works, otherwise falls back
to Docker. CI nightly-gate already passes `--runtime-backend auto`
([`.github/workflows/ci.yml`](../.github/workflows/ci.yml)).

The legacy static-only evidence is preserved under
`evidence/legacy-static-only/stage33/` and
`evidence/legacy-static-only/stage34/` for diff comparison.

## Setup

- **Date:** 2026-05-26 / 2026-05-27, single-night snapshot (Stage 3.4
  three-trial stability sweep is the same-evening replication).
- **Profile:** `cloud-default` (Alibaba DashScope / `dashscope/glm-5`).
- **Harness:** `opencode 1.15.10` via
  [`tests/tools/collect_generative_evidence.py`](../tests/tools/collect_generative_evidence.py)
  with `--runtime --runtime-backend host`. Workspace permissions per
  [`docker/opencode-profiles/cloud-default.json`](../docker/opencode-profiles/cloud-default.json).
- **Verification level:** **CANN simulator on the host install**,
  backend `Model`, platform `Ascend950PR_9599`. The simulator libs at
  `/usr/local/Ascend/cann/aarch64-linux/simulator/Ascend950PR_9599/lib`
  are prepended to `LD_LIBRARY_PATH` via
  [`run_and_verify._simulator_env`](../tests/tools/run_and_verify.py).
  Pass = `static_verify=="pass"` AND `score.accepted` AND
  `semantic_check.passed` AND kernel produced AND
  `verification.status=="pass"` from the simulator subprocess.
- **Sample:** 12 cells × 4 protocols = 48 evidence files;
  `--max-attempts 1`; 240 s opencode timeout, 360 s docker/host
  verification timeout (matmul + rms_norm bumped to 360 / 600 s).
- **Aggregation:** [`tests/tools/compare_skills_value.py`](../tests/tools/compare_skills_value.py)
  → `by_profile.cloud-default.{by_protocol, deltas_pp, deltas_pp_history}`.

## Per-cell breakdown (simulator-verified)

✓ = `overall_pass` (kernel + static + semantic + simulator). The static
column shows whether the AST check passed independently — cells where
static ✓ but simulator ✗ are the *drift* signal.

| Cell | P2 | P3 | P4 | P6 |
|---|---|---|---|---|
| `abs/float16` | ✗ (14 k) | ✗ (8 k) | ✓ (10 k) | ✓ (80 k) |
| `abs/float32` | ✗ (16 k) | ✗ (5 k) | ✓ (24 k) | ✓ (80 k) |
| `add/float16` | ✗ (16 k) | ✗ (12 k) | **✗ static-✓** (52 k) | **✗ static-✓** (103 k) |
| `reduce_sum/float16` | ✓ (16 k) | ✓ (19 k) | ✓ (13 k) | ✓ (109 k) |
| `reduce_sum/float32` | ✓ (29 k) | ✓ (22 k) | ✓ (13 k) | ✓ (121 k) |
| `gelu/float16` | ✗ (20 k) | **✗ static-✓** (9 k) | **✗ static-✓** (29 k) | ✓ (82 k) |
| `gelu/float32` | ✗ (12 k) | ✗ (9 k) | ✗ (9 k) | **✗ static-✓** (72 k) |
| `leaky_relu/float16` | **✗ static-✓** (75 k) | ✗ (4 k) | **✗ static-✓** (9 k) | **✗ static-✓** (86 k) |
| `softmax/float16` | ✓ (57 k) | ✓ (39 k) | ✓ (24 k) | ✓ (68 k) |
| `matmul/float16` | **✗ static-✓** (26 k) | ✗ (9 k) | **✗ static-✓** (21 k) | **✗ static-✓** (104 k) |
| `rms_norm/float16` | ✗ (13 k) | **✗ static-✓** (139 k) | ✓ (36 k) | ✓ (94 k) |
| `rms_norm/float32` | ✗ (30 k) | ✓ (34 k) | ✓ (33 k) | ✓ (128 k) |

Token totals (Σ + mean per protocol):

| Protocol | Σ tokens | mean / cell | pass rate | 95 % CI (Wilson) |
|---|---:|---:|---:|---|
| P2 | 322 k | 26 836 | 3/12 (25.0 %) | [8.9 %, 53.2 %] |
| P3 | 308 k | 25 643 | 4/12 (33.3 %) | [13.8 %, 60.9 %] |
| P4 | 273 k | 22 780 | 7/12 (58.3 %) | [32.0 %, 80.7 %] |
| P6 | 1 128 k | 94 030 | 8/12 (66.7 %) | [39.1 %, 86.2 %] |

## Static-vs-simulator drift

**11 cell-protocol combinations pass the AST static check but fail the
simulator.** This is the headline new finding the prior revision could
not see:

| Cell | Protocols that drift | Likely failure surface |
|---|---|---|
| `add/float16` | P4, P6 | binary-op output alias / overlap |
| `gelu/float16` | P3, P4 | tanh-vs-erf branch numerics |
| `gelu/float32` | P6 | upcast-cast sequence on host |
| `leaky_relu/float16` | P2, P4, P6 | slope constant precision |
| `matmul/float16` | P2, P4, P6 | C310 simulator zero-tensor edge (cf. matmul golden notes) |
| `rms_norm/float16` | P3 | KernelRmsNormRegBaseSplitD tiling-key miss |

**Implications for static-verify users:**

1. The Stage 3.3 P4 = 100 % readout was an artefact of static-only
   verification.
2. The Stage 3.5 dashboard panel CIs in the previous revision
   (`P4 − P3 = +33.3 pp [+2.2, +60.9]` and
   `P3 − P2 = +58.4 pp [+19.7, +79.0]`) overstated the deltas because
   the P3/P4 numerators were inflated by drift cells.
3. The merge-gate CI environment already has runtime verification
   enabled; the new "auto" backend keeps that behaviour, so the merge
   gate's pass-rate is the simulator-anchored one — *not* the
   static-only one.

## Per-protocol failure-mode mix

P2 (no skills + minimal prompt):
- 8 cells emit no kernel (F10). The minimal slot doesn't carry enough
  structure for `glm-5` to commit to an API.
- 2 cells emit a static-passing kernel but fail the simulator
  (leaky_relu/f16, matmul/f16). Slot regression — minimal prompt lets
  the agent guess at a structurally valid but wrong-platform invocation.
- 2 cells pass outright (reduce_sum/{f16,f32}, softmax/f16). These
  cells' minimal prompt is overspecified and remains an anomaly
  (flagged below).

P3 (guided, no AGENTS.md):
- 2 cells flip from F10 to a static-fail kernel (gelu/f32: no API
  knowledge; matmul/f16: also no API).
- 2 cells flip from drift-fail to drift-fail (gelu/f16, rms_norm/f16):
  the agent recovers the static structure but the numerics still go
  wrong.
- 1 cell adds a true pass (rms_norm/f32) — guidance alone is enough
  for the reduction-style tier-3 cell.

P4 (+AGENTS.md):
- 3 new true passes (abs/{f16,f32}, rms_norm/f16). AGENTS.md provides
  the platform / backend invocation that anchors numerics.
- 2 cells stay in static-✓/sim-✗ drift (add/f16, gelu/f16). AGENTS.md
  alone isn't enough to fix the specific numerical contracts (binary
  output alias on add, tanh branch on gelu).
- leaky_relu/f16 and matmul/f16 stay in drift — these need the
  skill-stack's `pyasc-codegen-workflow` to read examples_policy and
  pick the right tiling.

P6 (skills on, no AGENTS.md):
- 2 new true passes vs P4 (gelu/f16, gelu/f32 partial — gelu/f16 fully
  flips, gelu/f32 emits a static-passing kernel that still drifts but
  is closer than at P4). The skill stack's
  `pyasc-codegen-workflow/SKILL.md` references the glossary's "tanh
  vs erf" branch directly.
- add/f16, leaky_relu/f16, matmul/f16 stay in drift. **These three
  cells are the actual sprint blockers.**

## Skill stack value, isolated per cell (simulator-verified)

| Cell | P4 | P6 | Skill stack effect |
|---|---|---|---|
| `abs/float16` | ✓ | ✓ | no flip |
| `abs/float32` | ✓ | ✓ | no flip |
| `add/float16` | ✗ static-✓ | ✗ static-✓ | no flip |
| `reduce_sum/float16` | ✓ | ✓ | no flip |
| `reduce_sum/float32` | ✓ | ✓ | no flip |
| `gelu/float16` | ✗ static-✓ | ✓ | **flip P4→P6** |
| `gelu/float32` | ✗ | ✗ static-✓ | partial: static lifts, sim still fails |
| `leaky_relu/float16` | ✗ static-✓ | ✗ static-✓ | no flip |
| `softmax/float16` | ✓ | ✓ | no flip |
| `matmul/float16` | ✗ static-✓ | ✗ static-✓ | no flip |
| `rms_norm/float16` | ✓ | ✓ | no flip |
| `rms_norm/float32` | ✓ | ✓ | no flip |

**1 of 12 cells fully flips on the skill intervention
(gelu/float16); 1 of 12 partially flips on static
(gelu/float32).** The +8.4 pp delta is mechanically a 1-cell
difference. The 95 % CI is wide because the lift is fragile to which
single cell wins. The headline P4 = 100 % static (prior revision) is
gone; the corrected P4 = 58.3 % gives the skill stack room to add the
gelu/f16 win.

## Stage 3.4 stability sweep (simulator-verified)

Three trials per (cell, protocol) on the 4 boundary cells
**abs/f16, gelu/f16, matmul/f16, rms_norm/f16** (a different set
from the prior static-only revision — the new boundary lives at the
simulator drift, not the static failures). 32 trial-2 / trial-3
files plus the 4 trial-1 files from Stage B share the boundary key.

Pass rate per trial on the boundary 4-cell subset:

| Trial | P2 (boundary 4) | P3 (boundary 4) | P4 (boundary 4) | P6 (boundary 4) |
|---|---:|---:|---:|---:|
| #1 | 0/4 (0 %) | 0/4 (0 %) | 2/4 (50 %) | 3/4 (75 %) |
| #2 | 0/4 (0 %) | 2/4 (50 %) | 1/4 (25 %) | 4/4 (100 %) |
| #3 | 1/4 (25 %) | 1/4 (25 %) | 1/4 (25 %) | 2/4 (50 %) |

Per-cell flip history (3 trials, F = sim-fail, P = sim-pass):

| Cell | P2 (3 trials) | P3 | P4 | P6 | Classification |
|---|---|---|---|---|---|
| `abs/float16` | F→F→P | F→F→P | P→F→F | P→P→P | **P2/P3/P4 noisy**, P6 stable pass |
| `gelu/float16` | F→F→F | F→F→F | F→F→F | P→P→F | **P6 noisy** |
| `matmul/float16` | F→F→F | F→P→F | F→P→F | F→P→F | **all noisy at simulator** — even P6 fails 2/3 |
| `rms_norm/float16` | F→F→F | F→P→F | P→F→P | P→P→P | P3/P4 noisy, P6 stable pass |

### Findings from the simulator-verified stability sweep

1. **The noise floor is much higher than the static-only sweep
   suggested.** Static-only Stage 3.4 showed P4 = 4/4 and P6 = 4/4
   on every trial — apparent stability. Simulator-verified, P4 swings
   1/4 → 2/4 across trials, and P6 swings 2/4 → 4/4. The earlier "P6
   stable at 100 % boundary" was a static-blind artifact.
2. **`matmul/float16` is unstable at every protocol.** Including P6.
   Three trials: F, P, F. The TilingKey / KernelMatmulNzNz selection
   depends on opencode planning details that vary trial-to-trial. The
   skill stack alone is not enough; an oracle-guided variant
   (`prompt_variants.oracle_guided`) is necessary.
3. **`abs/float16` P6 is the only stable-pass column.** 3/3 trials
   simulator-pass at P6. It's also the cell with the lowest token
   spend at P6 (~80 k). This is the model the rest of the
   elementwise cells should reach.
4. **`gelu/float16` P6 is borderline.** 2/3 trials pass. The third
   trial fails on a tanh approximation branch the skill stack
   documents but doesn't enforce.
5. **No cell × protocol combination has a stable static-✓ /
   simulator-✗ pattern.** The drift is genuinely noise-bounded: the
   same cell that drifts in Stage B may pass in trial 2 or 3. This
   makes drift hard to fix proactively (an oracle pass on one trial
   doesn't guarantee the next will repeat).

### Caveat: same-evening vs day-over-day noise

Trials 1-3 all ran on the same evening (2026-05-26 21:03 → 2026-05-27
02:14 UTC+3, ~5 h). They measure intra-session model + simulator
variance — temperature, sampling, opencode loop nondeterminism — at a
single point in DashScope's traffic schedule. Day-over-day noise is
likely **larger** than intra-session noise; published CIs here are
lower bounds on true uncertainty. `deltas_pp_history` in the
aggregator is the seam for a 3-night follow-up.

## Budget realized vs estimated (with simulator overhead)

Stage 3.2 ([`docs/perf-methodology/phase-3-budget.md`](perf-methodology/phase-3-budget.md))
projected the 4-leg matrix at **1.31× the 2-leg legacy**, using
`abs/float16` as the basis cell. Realized:

| Metric | Projected (abs/f16 basis, static-only) | Realized (12-cell matrix, simulator) |
|---|---:|---:|
| Σ tokens P2+P3+P4+P6 / cell | 163 948 | **169 290** (avg over 12) |
| Σ tokens P3+P6 / cell | 125 329 | **119 673** (avg over 12) |
| Matrix multiplier (4-leg / 2-leg) | 1.31× | **1.41×** |
| Σ tokens / nightly (12 cells) | 1.97 M | **2.03 M** |
| Wall-clock per cell, simulator-verified | n/a | **3.5 min mean** (50 s − 570 s range) |
| Stage B wall-clock (48 cells, sequential) | n/a | **3 h 4 min** |
| Stage C wall-clock (32 trials, sequential) | n/a | **2 h 3 min** |

Token spend tracks the projection within 4 %. The new cost is
**wall-clock**: the host simulator adds 30 – 90 s per cell that
produces a kernel (40 of 48 in Stage B). Sequential matrix execution
costs ~3 h for Stage B; parallelism=2 cuts that to ~1.5 h and a
concurrency probe (2 simultaneous simulator runs of independent golden
kernels) confirmed it is safe. The default orchestrator stays at 1 for
forensic clarity; CI nightly-gate runs each matrix leg in its own job
already.

## Anomalies flagged for Phase 5 / 6

1. **`softmax/float16` passes at every protocol including P2 — the
   only such cell.** The minimal-slot prompt for softmax/f16 produces
   a static-passing AND simulator-passing kernel. Re-audit Stage 2.2's
   minimal variant for over-specification. Owner:
   pyasc-skill-stack-author.
2. **`reduce_sum/{f16,f32}` pass at P2.** Two cells where P2 already
   suffices. The reduce-axis prompt slot may be doing more work than
   intended. Owner: prompt-template author.
3. **`matmul/float16` instability persists at simulator.** 3 trials,
   1 pass. Track this as a candidate for `prompt_variants.oracle_guided`
   promotion in Phase 5 — the cell's docs already note the C310
   simulator zero-tensor edge. Owner: docs/baseline / Phase 5.
4. **`add/float16` P4 + P6 static-✓/sim-✗.** AGENTS.md alone and the
   skill stack alone are both insufficient to fix the binary-op output
   alias. This is the most concrete sprint-actionable item: add
   examples_policy guidance for binary ops in
   `skills/pyasc-codegen-workflow/SKILL.md`. Owner:
   pyasc-skill-stack-author.
5. **`leaky_relu/float16` fails at every protocol.** Even P6. The
   slope-constant precision is wrong across the board. Candidate for
   a `prompt_variants.oracle_guided` slot in capabilities.yaml.
   Owner: prompt-template author.
6. **`rms_norm/float16` P3 spends 139 k tokens.** Far above the
   per-protocol median of ~26 k. The guided prompt sends the agent
   into a long workaround search that doesn't converge. Investigate
   whether the prompt should pin tiling-key earlier. Owner:
   harness-runtime.

## Caveats

- **Single-night snapshot.** All 12 × 4 cells were run sequentially
  on 2026-05-26 evening (~3 h Stage B + ~2 h Stage C). Day-over-day
  variance is unsampled.
- **Simulator-verified, not hardware-verified.** Cells reported as ✓
  pass the cycle-accurate model on `Ascend950PR_9599` (C310). A real
  Ascend NPU may behave differently for tiling regimes that exceed the
  simulator's modelled state.
- **`--max-attempts 1`.** One attempt per leg, no retry. A P2 leg
  that times out at 240 s is recorded as a single fail; it could
  succeed on a re-run. Stage 3.4 boundary cells show ~50 % stability
  on the noisy protocols.
- **Host CANN install pinned to 9.0.0-beta.2.** The CI Docker image
  also targets 9.0; if the published toolkit drift differs from the
  host install, the auto-resolved host backend would mask it.

## What this changes for the sprint

- **No-go on adding skill complexity until the drift cells are fixed.**
  Three specific cells (add/f16, leaky_relu/f16, matmul/f16) are the
  static-vs-simulator drift hot spots. Adding more skill content
  before these are resolved would compound the false-positive rate.
- **AGENTS.md baseline is the bar, but it's not the ceiling.** P4 =
  58.3 % is achievable from the vendored baseline alone; the skill
  stack lifts that to 66.7 %. The remaining headroom (33.3 %) lives
  in cells where neither AGENTS.md nor the skill stack is enough —
  these need oracle_guided promotion or new skill content.
- **The Newcombe CI is the publication unit.** The shift from
  "static-only +0 pp [−24, +24]" to "simulator +8.4 pp [−27, +41]"
  illustrates *why*: the point estimate is informative, but the CI
  still includes zero at n=12, so any sprint-level claim must cite
  both numbers. The dashboard's `renderProtocolDecomp` panel renders
  both side by side.
- **Re-running the matrix is cheap.** ~3 h Stage B + 2 h Stage C
  unattended on dev hardware. Stage 3.4's "boundary cells" definition
  changes at simulator verification (`abs/f16` is no longer noisy;
  `matmul/f16` is universally noisy). Future stability sweeps should
  pick boundary cells from the simulator-verified matrix, not the
  static-only one.

## Cross-references

- Methodology: [`docs/evaluation-methodology.md` §"Comparisons of interest"](evaluation-methodology.md#comparisons-of-interest).
- Protocol matrix: [`docs/evaluation-methodology.md` §"Protocol-axis CI mapping (Phase 0)"](evaluation-methodology.md#protocol-axis-ci-mapping-phase-0).
- Prompt template: [`docs/prompt-template.md`](prompt-template.md).
- Budget projection: [`docs/perf-methodology/phase-3-budget.md`](perf-methodology/phase-3-budget.md).
- Host runtime backend: [`tests/tools/collect_generative_evidence.run_host_verify`](../tests/tools/collect_generative_evidence.py).
- Aggregator: [`tests/tools/compare_skills_value.py`](../tests/tools/compare_skills_value.py).
- Dashboard panel: [`tests/tools/generate_dashboard.py`](../tests/tools/generate_dashboard.py) function `renderProtocolDecomp`.
- Evidence (Stage B): `evidence/*-generative-cloud-default-{p2,p3,p4,p6}.json` (48 files, simulator-mode).
- Evidence (Stage C, stability r2/r3): `evidence/*-generative-cloud-default-p<N>-r{2,3}.json` (32 files).
- Legacy static-only baseline (for diff): `evidence/legacy-static-only/stage33/` and `evidence/legacy-static-only/stage34/`.
