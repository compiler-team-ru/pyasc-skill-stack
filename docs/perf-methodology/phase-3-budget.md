# Phase 3 — Token-budget rehearsal (Stage 3.2)

Decision: **proceed with all 4 legs nightly**. Measured multiplier is
1.31× the legacy 2-leg matrix, well under the plan's 2.5× guardrail.
No `workflow_dispatch.tier: protocol-full` split is needed.

## Measurement basis

One full end-to-end run of `abs/float16` on `cloud-default` against
`dashscope/glm-5`, dispatched via
[`tests/tools/collect_generative_evidence.py`](../../tests/tools/collect_generative_evidence.py)
with `--protocol-id` ∈ {P2, P3, P4, P6}. Evidence files:

- [`evidence/abs-f16-generative-cloud-default-p2.json`](../../evidence/abs-f16-generative-cloud-default-p2.json)
- [`evidence/abs-f16-generative-cloud-default-p3.json`](../../evidence/abs-f16-generative-cloud-default-p3.json)
- [`evidence/abs-f16-generative-cloud-default-p4.json`](../../evidence/abs-f16-generative-cloud-default-p4.json)
- [`evidence/abs-f16-generative-cloud-default-p6.json`](../../evidence/abs-f16-generative-cloud-default-p6.json)

Captured 2026-05-26, single trial each, local cloud-default profile.

## Per-leg numbers (abs/float16, single trial)

| Leg | Variant | skills | AGENTS.md | tokens.total | tokens.in | tokens.out | cache_read | wall-clock |
|---|---|---|---|---:|---:|---:|---:|---:|
| P2 | minimal | off | no | 4 503 | 3 526 | 977 | 39 113 | 45.1 s |
| P3 | guided  | off | no | 18 021 | 15 982 | 2 039 | 87 688 | 128.6 s |
| P4 | guided  | off | yes | 34 116 | 32 349 | 1 767 | 298 968 | 300.0 s (timeout) |
| P6 | guided  | on  | no | 107 308 | 101 769 | 5 539 | 407 283 | 300.0 s (timeout) |

Notes:
- `tokens.total = input + output` (cache reads are not billed).
- P4 and P6 both hit the 300 s harness timeout. They produced kernels
  but the agent loop did not voluntarily exit. The token count is the
  cumulative spend at the timeout cutoff, not the model's "natural"
  conclusion. Stage 3.4 will tighten or relax the timeout based on
  three-night observation.
- P6's spend (107 308 tokens) is 2.4× P4's (34 116) and 24× P2's
  (4 503). On `abs/f16` the skill stack is invoked deeply: design.md,
  self_review.md, acceptance_review.md, code-review skill load, etc.
  Tier-2/3 cells should scale similarly.

## Projection formula

Let `t(P)` = mean tokens per cell per leg. Total nightly spend:

```
tokens_nightly_4leg = sum_{P in {P2, P3, P4, P6}} sum_{cell} t(P, cell)
                    ≈ 12 cells × sum_{P in {P2, P3, P4, P6}} mean_t(P)
                    (under the assumption that abs/f16 is
                     representative of mean per-cell cost; tier 2/3
                     cells likely spend more, this projection is the
                     conservative-floor)
```

Today's 2-leg matrix (the legacy `skills_mode: [off, on]` matrix
mapped to `{P3, P6}`):

```
tokens_nightly_2leg = sum_{P in {P3, P6}} sum_{cell} t(P, cell)
                    ≈ 12 cells × (t(P3) + t(P6))
```

Multiplier:

```
multiplier = tokens_nightly_4leg / tokens_nightly_2leg
           = (t(P2) + t(P3) + t(P4) + t(P6)) / (t(P3) + t(P6))
           = 163 948 / 125 329
           = 1.31
```

## 12-cell projection

| Metric | 2-leg legacy | 4-leg Phase 3 | Δ |
|---|---:|---:|---:|
| Tokens / nightly | 1 503 948 | 1 967 376 | +463 428 |
| Multiplier | 1.0× | 1.31× | +31 % |

The +463 K tokens / nightly delta is the price of the protocol-axis
decomposition. The 4-leg matrix adds two new measurement legs (P2 +
P4) for ~31 % more spend.

## Realized (Q1 re-run on `pyasc-v2-eval @ 7b85554a`)

After Stage A added the host CANN runtime backend and the Q1 plan
re-pinned the harness from the stale `cann/pyasc#wip-matmul-sync-and-reduce-fuse`
clone onto the canonical `compiler-team/pyasc#v2 @ 7b85554a` baseline,
the full 12-cell × 4-protocol matrix and 4-cell × 4-protocol × 2-trial
stability sweep were re-collected on 2026-05-27 (`--runtime
--runtime-backend host`, `--parallel 2`, `--max-attempts 3`).

| Metric | Projected (abs/f16, Stage 3.2) | Realized — wip tree | Realized — `v2-eval @ 7b85554a` | Δ (v2 − wip) |
|---|---:|---:|---:|---:|
| Σ tokens P2+P3+P4+P6 / cell (mean over 12) | 163 948 | 169 290 | **407 469** | +140.7 % |
| Σ tokens P3+P6 / cell (mean over 12) | 125 329 | 119 673 | **255 152** | +113.2 % |
| Σ tokens / nightly (12 cells, 4 legs) | 1.97 M | 2.03 M | **4.90 M** | +2.87 M |
| Multiplier (4-leg / 2-leg) | 1.31× | 1.41× | **1.60×** | +0.19× |
| Wall-clock, simulator-pass cell (mean) | n/a | 200–450 s | 240–820 s | longer plan/review loops |
| Wall-clock, no-kernel cell (P2 fail-fast) | n/a | 25–140 s | 142–365 s | — |
| Stage B wall-clock (48 cells, parallelism=2) | n/a | n/a | **3 h 0 min** | — |
| Stage C wall-clock (32 stability trials, parallelism=2) | n/a | n/a | **2 h 0 min** + 50 min resume | — |
| Total run (Stage B + C + resume) | n/a | n/a | **7 h 26 min** | — |

**Decision still holds: 4 legs nightly, no schedule split.** Multiplier
on v2 is 1.60× vs the 2.5× guardrail; the new cost is concentrated in
P6 where the skill stack burns ~177 k tokens / cell (vs 94 k on wip).
The driver is `gelu/float16` P6 (drift cell that costs 371 k tokens
to fail) and `abs/{f16,f32}` P6 (drift cells that cost 271 k / 338 k
to fail). The Q1 findings file flags these for skill-stack rewrite +
verify-tolerance audit (see "Anomalies" section there).

The mid-run editable-install diversion to `pyasc-fork` lost 18
Stage 3.4 trials at 11:30 UTC; the orchestrator was extended with a
`--resume-from` flag and a `PYTHONPATH` pin (see
[`tests/tools/run_matrix_v2_eval.py`](../../tests/tools/run_matrix_v2_eval.py))
and the resume completed cleanly with all 18 trials passing or
failing on a recorded SHA. Total wall (including resume) was
**7 h 26 min**.

## Decision tree (per the Phase 3 plan)

| Projected multiplier | Decision |
|---|---|
| ≤ 2.5× | proceed with all 4 legs nightly (this is where we are) |
| > 2.5× | introduce `workflow_dispatch.tier: protocol-full` running P2 + P4 weekly; keep P3 + P6 nightly |

**Decision: 4 legs nightly.** No CI schedule changes required.

## What the projection does NOT capture

- **Per-tier variance.** abs/f16 is tier-0 (elementwise). Tier-2/3
  cells (gelu/f32, matmul/f16, rms_norm/{f16,f32}) likely have higher
  per-cell `t(P6)` because the skill stack produces more design /
  review artifacts on complex kernels. The full Stage 3.3 matrix will
  re-compute the multiplier across all 12 cells.
- **Per-run variance.** Single-trial measurement. Stage 3.4's
  stability sweep produces three trials per leg, after which we can
  report a confidence interval on the multiplier.
- **Cache-read costs.** Some providers bill cache reads at a discount;
  `dashscope` documentation does not detail this. The `cache_read`
  column in the table is shown for completeness but is excluded from
  the multiplier above.
- **Provider rate-limit head-room.** dashscope has per-key QPS limits;
  4 cells × 4 protocols launched in parallel would hit them. Stage 3.3
  serializes intra-protocol (parallelism within one protocol leg) and
  parallelizes across protocols.

## Risks the plan called out

- **Budget overrun.** Currently 1.31× vs the 2.5× guardrail — wide
  head-room. If the full-matrix multiplier turns out > 2× on tier 2/3
  cells (plausible given P6's 24× elementwise advantage), revisit
  before committing to a long-run nightly schedule.
- **Per-leg attempt budget.** P2 currently exhausts `--max-attempts 3`
  (default) without producing a kernel. The plan's mitigation —
  dropping `--max-attempts` to 2 for P2 — would cut P2's spend by ~33 %
  without losing any signal (F10 / no-artifact is fully diagnostic at
  2 attempts).
- **Timeout bleeding into spend.** P4 and P6 both hit the 300 s harness
  timeout. The agent kept calling tools and spending tokens up to the
  cutoff. If the typical "natural" exit is far short of 300 s, the
  timeout overshoot is wasted spend; if it isn't, the protocol-axis
  decomposition is incomplete. Stage 3.4 monitors this.

## Orchestrator timeouts and effective sim budget (realized on v2-eval)

The Phase 3 budget projection only fixed the per-leg LLM timeout. The
v2-eval matrix exposed two additional timeout layers that materially
shape spend and wall:

| Layer | Where | Default | Effect |
|---|---|---:|---|
| Agent / LLM timeout | `collect_generative_evidence.py --timeout` | 240 s | Stops the agent loop. Tokens billed up to this cutoff. |
| Sim verify timeout | `collect_generative_evidence.py --docker-timeout` | 180 s | Wall budget for one `python kernel.py -r Model -v Ascend950PR_9599` invocation. |
| Effective sim budget | [`tests/tools/run_and_verify.py`](../../tests/tools/run_and_verify.py) | `docker-timeout − 30 s` = **150 s** | Buffer for orchestration overhead; the kernel itself sees a 150 s wall. |

The `run_matrix_v2_eval.py` orchestrator now defaults to these values.
The Stage 3.3 / 3.4 runs reported above were collected with
`--timeout 240 --docker-timeout 180 --parallel 2`.

**Risk — composed-math timeouts.** `gelu/float16` P4 and P6 retries hit
the 150 s effective sim budget repeatedly during Stage 3.3, contributing
to the 0/12 stability sweep result. Several other Tier-2 cells
(softmax, leaky_relu) ran 110–140 s — uncomfortably close. The Phase 9
plan does **not** raise `--docker-timeout` (every cell's wall budget
would change), but flags it for a follow-up sprint. The drift cells
might still be tight after the rank-consistency teaching lands; revisit
after the Phase 9 targeted re-run.

**Risk — mid-run editable-install diversion (one-time).** Stage 3.3
lost 18 trials on 2026-05-27 11:30 UTC to an unintended
`pip install -e ../pyasc-fork` triggered by the P4 baseline
AGENTS.md (see
[`docs/skill-value-q1-findings.md` Appendix B](../skill-value-q1-findings.md#appendix-b--postmortem-2026-05-27-1130-utc-pip-install-diversion)).
The orchestrator absorbed the failure with a `--resume-from` flag and
the resume completed in **50 min**. This is one-time cost recorded
here so the budget memo reflects realized wall (7 h 26 min) faithfully;
the mitigation (PYTHONPATH pin + hard ERROR) is already in place.

## Cross-references

- Stage 3.2 spec in [`.cursor/plans/phase_3_agents_md_baseline.plan.md`](../../.cursor/plans/phase_3_agents_md_baseline.plan.md).
- Protocol matrix in [`docs/evaluation-methodology.md`](../evaluation-methodology.md#protocol-axis-ci-mapping-phase-0).
- Prompt-template (slot order that drove the per-leg prompt size) in
  [`docs/prompt-template.md`](../prompt-template.md).
