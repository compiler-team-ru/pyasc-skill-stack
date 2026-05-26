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

## Cross-references

- Stage 3.2 spec in [`.cursor/plans/phase_3_agents_md_baseline.plan.md`](../../.cursor/plans/phase_3_agents_md_baseline.plan.md).
- Protocol matrix in [`docs/evaluation-methodology.md`](../evaluation-methodology.md#protocol-axis-ci-mapping-phase-0).
- Prompt-template (slot order that drove the per-leg prompt size) in
  [`docs/prompt-template.md`](../prompt-template.md).
