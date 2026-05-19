# Evaluation methodology

This document describes what the `pyasc-skill-stack` matrix is, what it
measures, and how to read its numbers without overclaiming.

## What is being compared

The pyasc-skill-stack matrix is an evidence-backed **capability dashboard**
and an **OpenCode skills-intervention experiment**. It compares OpenCode
runs with pyasc skills enabled against OpenCode runs with pyasc skills
disabled.

The score belongs to the full generation protocol — task cell, prompt
variant, OpenCode harness, model/profile, allowed context, skill mode,
budget, and evaluator. Guided prompts, golden references, oracle
workarounds, and human hints are separate interventions and must be
labeled separately.

The intended comparison unit is:

> the full candidate-generation protocol: OpenCode harness, same
> model/profile where applicable, same task cell, same budget, same
> evaluator, with skills enabled or disabled.

It is **not** "model vs skills". The model is one held-constant
variable inside the protocol, not one of the legs of the comparison.

## Data flow

```
task cell (capabilities.yaml)
   -> OpenCode harness  (skills-on or skills-off)
   -> generated artifacts (kernel.py, design.md, ...)
   -> evaluator (static -> JIT -> simulator -> [performance, planned])
   -> evidence/<op>-<dtype>-generative*.json
   -> compare_skills_value.py -> evidence/skills-value-summary.json
   -> generate_dashboard.py   -> Pages dashboard
```

A task cell is the unit of work: `(operation, dtype, platform, prompt,
shapes, prompt_variants, golden, ...)` as defined in
[capabilities.yaml](../capabilities.yaml). One task cell yields one or
more evidence files; the file name suffix encodes the
`(model_profile, skills_mode)` slice it was produced under, for example
`evidence/abs-f16-generative-cloud-default-off.json`.

## Intended design vs. measured comparability

The CI matrix in [`.github/workflows/ci.yml`](../.github/workflows/ci.yml)
uses `skills_mode: ["on", "off"]` and is **designed** as a paired
skills-on/off intervention with all other variables held constant: same
OpenCode harness, same resolved model, same prompt, same timeout, same
max attempts, same evaluator, same allowed filesystem except for the
skill modules.

Whether each evidence *pair* (on, off) actually achieved that designed
comparability is a separate question that
[`tests/tools/compare_skills_value.py`](../tests/tools/compare_skills_value.py)
answers per-pair via a validity classifier. Pairs that fail the check
are not a clean OpenCode-without-skills baseline; they are reported
separately and excluded from the headline delta. See
[Baseline validity](#baseline-validity) below.

## Three matrix views

The dashboard renders the first two today. The third is documented as
the next deliverable.

### A. Capability coverage matrix

Purpose: show which pyasc kernel patterns are covered.

Fields (already in [capabilities.yaml](../capabilities.yaml) and
`evidence/*.json`):

| Field | Source |
|---|---|
| op, dtype, platform | cell |
| tier | operation |
| shape family / `shapes` | cell |
| `golden_status`, `generative_status` | cell |
| `static_verify`, `verification.mode`, `verification.status`, `score` | evidence |
| evidence link | cell |

This is the existing top table on the dashboard.

### B. OpenCode skills intervention matrix

Purpose: measure OpenCode-with-skills vs OpenCode-without-skills on the
same model and protocol.

Fields (in `evidence/skills-value-summary.json` and the dashboard
banner):

| Field | Notes |
|---|---|
| op, dtype, tier, model_profile, model | identifies the cell + slice |
| skills_off pass / score / tokens / elapsed / attempts | raw off-leg metrics |
| skills_on pass / score / tokens / elapsed / attempts | raw on-leg metrics |
| `delta_pass`, `delta_score`, `delta_tokens`, `delta_elapsed`, `delta_attempts` | per-cell deltas (on minus off) |
| `pass_rate_off`, `pass_rate_off_clean` (nullable) | raw vs. validity-filtered off pass-rate |
| `viability_unlocked_clean` | strict unlock count (see below) |
| `unresolved_due_to_off_infra` | pairs where on passes and off failed comparability |

Never call this view "model vs skills". It is "OpenCode skills
intervention" / "OpenCode skills-on vs skills-off".

### C. Model/profile capability matrix (planned)

Purpose: show which model/profile configurations are strong enough to
benefit from the workflow.

Proposed fields (not yet rendered):

- `model_profile`, `model`, `protocol`
- `pass@1`, `pass@k` (if `--max-attempts > 1`)
- `artifact_found_rate`, `static_verify_rate`, `jit_simulator_pass_rate`
- `clean_completion_rate`, `timeout_after_success_rate`
- `failure_category_distribution`

Some of these require schema v4 (see
[Evidence schema v4 (proposed)](#evidence-schema-v4-proposed)).

## Generation protocol taxonomy

The matrix should be readable per *protocol*, not per "raw model".

| Id | Definition | Status in this repo |
|---|---|---|
| P0 | one-shot minimal prompt, no OpenCode agent loop | not implemented |
| P1 | one-shot guided prompt, no OpenCode agent loop | not implemented |
| P2 | OpenCode agent, minimal prompt, skills off | future: `--skills-mode off --prompt-variant minimal` |
| P3 | OpenCode agent, guided prompt, skills off | nearest current: today's skills-off CI leg uses the legacy single `prompt`; closer to P3 than P2 |
| P4 | OpenCode agent + allowed docs/examples, skills off | future |
| P5 | OpenCode agent, minimal prompt, skills on | future |
| P6 | OpenCode agent + allowed docs/examples, skills on | nearest current: today's skills-on CI leg |
| P7 | oracle/reference-guided diagnostic mode | partial — `golden/` is used as an oracle by humans, not by the agent |
| P8 | human-assisted development mode | manual workflow; **not** an autonomous leaderboard score |

**Comparisons of interest.** Only these comparisons quantify the skills
intervention itself:

- `P5 − P2` — marginal value of pyasc skills in OpenCode under minimal
  prompting.
- `P6 − P4` — marginal value of pyasc skills beyond docs/examples.

These are prompting / context interventions, not skill effects:

- `P3 − P2` — value of guided prompting.
- `P4 − P3` — value of docs/examples/reference context.

These are diagnostic upper bounds, not autonomous leaderboard scores:

- `P7`, `P8`.

Today's `skills-value-summary.json` only contains data on a slice that
is *near* `P6 − P3`. Wiring the variants to first-class protocol ids is
listed in [Out of scope (planned)](#out-of-scope-planned).

## Prompt-variant labeling rules

| Variant | Allowed content |
|---|---|
| minimal | operator name, dtype, shape, semantic definition only |
| guided | API hints, verification hints, general implementation guidance |
| oracle-guided | exact workaround, exact tiling, known failure fix, golden-derived clue, backend bug workaround |
| human-assisted | any human-edited prompt, code, debugging step, or manual intervention |

The strings in `capabilities.yaml -> cells[*].prompt_variants.guided`
are **guidance-grade**, not oracle-grade. Do not silently count
oracle-guided information as pure skill-stack value: if a prompt
includes a workaround for a known backend bug, it belongs in
`oracle-guided`, and the resulting score belongs to `P7`.

## Baseline validity

A rigorous skills-off run should have:

- OpenCode active
- same resolved model/profile as the skills-on run
- same task cell
- same prompt variant
- same timeout
- same max attempts
- same evaluator
- same allowed filesystem/context except for disabled skills

`compare_skills_value.py` classifies each evidence file with a
`validity` value using the signals already in schema v3:

- **ok** — the harness appears to have executed and evidence is
  complete enough for comparison. Concretely: at least one of model
  resolved, non-zero tokens, non-empty `agent.artifacts_found`, or
  non-empty `kernel_path`, **and** no contradictory mix of fields.
- **infra_fail** — strong instrumentation/configuration-failure
  signature: `model in (null, "")` **and** `tokens.total == 0`
  **and** `agent.artifacts_found == []` **and** `kernel_path == ""`.
  This is not a comparable OpenCode-without-skills baseline; the
  classifier excludes the cell from the headline pass-rate.
- **incomplete** — partial or contradictory evidence (some indicators
  present, others missing) that is not strong enough to call
  `infra_fail` but is also not safe to count as a clean baseline.
  Reported separately; not folded silently into `pass_rate_off`.

### Derived clean aggregates

The classifier feeds three derived aggregates used by the dashboard:

- `pass_rate_off_clean` — `pass_off_clean / cells_off_ok`. Computed
  only when `cells_off_ok > 0`. When the denominator is zero,
  `pass_rate_off_clean` is **null** and the report renders
  `"Clean skills-off baseline unavailable: N off-runs classified as
  infra/config failures"` instead of a fabricated `0/0` ratio.
- `viability_unlocked_clean` — count of pairs where
  `on.pass AND off.validity == "ok" AND not off.pass`. This is the
  strict, defensible "unlock" count.
- `unresolved_due_to_off_infra` — count of pairs where
  `on.pass AND off.validity == "infra_fail"`. These pairs are
  **not** claimed as unlocked because the no-skills counterfactual was
  never actually measured.

The raw `pass_rate_off`, `viability_unlocked_count`, and the per-cell
delta fields stay in the summary so consumers that only render raw
deltas keep working; the validity-aware fields are added alongside.

### Headline rendering rule

The dashboard verdict renders one of two forms depending on whether a
clean off-baseline exists for a profile:

- **At least one clean off-run:** `P_off_clean -> P_on_clean (Δ pp)`,
  with a footnote tooltip "Excluding N off-runs classified as
  infra/config failures; raw off pass-rate = X%".
- **No clean off-runs:** `Skills-on pass rate: X%. Clean skills-off
  baseline unavailable: N off-runs classified as infra/config
  failures.` The Δ-pp line is suppressed. Today's data falls into this
  branch.

## Local-model interpretation

Local small models failing under both skills-on and skills-off
indicates the model/harness capability boundary for that protocol on
that hardware, not that skills are useless. The rigorous interpretation
of "skills did not help" requires `off.validity == "ok"`, i.e. the
skills-off leg actually produced measurable evidence to compare
against.

## Evidence schema v4 (proposed)

Current evidence is on `schema_version: "3"`. Schema v4 is an additive
extension; all v3 readers (`compare_skills_value.py`,
`generate_dashboard.py`, `sync_capabilities.py`) continue to work
because the new fields are optional and read defensively.

```yaml
schema_version: "4"

protocol:
  id: P6                       # P0..P8
  name: opencode-skills-on-guided
  prompt_variant: minimal | guided | oracle-guided | human-assisted
  skills_enabled: true
  skill_set_hash: <sha256 of mounted skill set>
  allowed_context:
    task_prompt: true
    capabilities_yaml: true
    golden_kernels: false
    golden_docs: false
    external_web: false
    human_hints: false

model:
  provider: openrouter
  model_id: dashscope/glm-5
  model_version_or_date: 2026-05-12
  temperature: 0
  seed: null

agent:
  harness: opencode
  harness_version: 1.4.1
  tools_allowed: [bash, edit, read, ...]
  max_attempts: 4
  timeout_s: 420

result:
  artifact_found: true         # a kernel.py exists on disk
  artifact_accepted: true      # static+semantic+verify checks pass
  agent_clean_completion: true # opencode exited cleanly
  timeout_after_success: false # accepted artifact, then harness timed out
  static_verify: pass
  jit_verify: pass
  simulator_verify: pass
  performance_verified: null   # see Performance layer (planned)
  score:
    value: 16
    threshold: 12
    accepted: true
  failure_category: null       # F1..F13 or null on success
```

### v3 -> v4 mapping

| v3 path | v4 path |
|---|---|
| `skills_mode` | `protocol.skills_enabled` |
| `model_profile` | (kept as top-level for grouping) |
| `model` | `model.model_id` |
| `tokens.{input,output,cache_read,total}` | unchanged |
| `agent.platform` | `agent.harness` |
| `agent.timeout_s` | `agent.timeout_s` |
| `agent.artifacts_found` (non-empty) | `result.artifact_found = true` |
| `kernel_path` + `score.accepted` + `verification.status == "pass"` | `result.artifact_accepted = true` |
| `agent.completed` | `result.agent_clean_completion` |
| `static_verify` | `result.static_verify` |
| `verification.status` (mode `simulator`) | `result.simulator_verify` |
| `score` | `result.score` |
| (none) | `result.failure_category`, `protocol.*`, `model.provider/version/temperature/seed`, `agent.harness_version`, `agent.tools_allowed`, `result.timeout_after_success`, `result.performance_verified` |

Crucially, `result.artifact_found`, `result.artifact_accepted`, and
`result.agent_clean_completion` are separate fields. A run can produce
an accepted kernel and still time out afterward
(`timeout_after_success: true`); a run can also produce no artifact at
all (`artifact_found: false`).

## Failure taxonomy

`result.failure_category` records exactly one category per evidence
file — the earliest in the F1 → F13 order that applies.

| Id | Definition |
|---|---|
| F1 | prompt/spec misunderstood |
| F2 | wrong pyasc/asc2 API usage |
| F3 | invalid `@asc2.jit` syntax |
| F4 | tiling or shape mismatch |
| F5 | dtype/precision mismatch |
| F6 | static verification failure |
| F7 | JIT/codegen failure |
| F8 | simulator/runtime failure |
| F9 | correct but slow |
| F10 | no artifact generated |
| F11 | timeout before usable artifact |
| F12 | timeout after usable artifact |
| F13 | infrastructure/configuration failure |

`F13` is the schema-v4 promotion of today's `validity == "infra_fail"`
case; a v4 collector should write `failure_category: F13` for those
runs.

## Performance layer (planned)

For pyasc kernels, correctness is not enough. The proposed performance
fields (none populated today; the C310 simulator does not yet provide
them reliably; **do not invent numbers**) are:

- `result.runtime_ms`
- `result.baseline_runtime_ms`
- `result.speedup_vs_reference`
- `result.speedup_vs_golden`
- `result.correct_but_slow` (boolean)
- `result.performance_regression` (boolean)

These graduate to first-class fields once the evaluator can measure
them on a representative platform.

## Out of scope (planned)

The following items are the next deliverables; this document is the
contract they will conform to.

- Bump evidence `schema_version` to `"4"` and emit the new fields from
  [`tests/tools/collect_generative_evidence.py`](../tests/tools/collect_generative_evidence.py).
- Wire `--prompt-variant {minimal,guided,oracle-guided,human-assisted}`
  through to the collector so P2/P3/P5/P6 can be exercised in CI.
- Implement the failure-taxonomy classifier in code (today,
  `failure_category` is documented but not auto-filled).
- Render matrix view C (model/profile capability) on the dashboard.
- Performance-layer fields and runtime measurements.
