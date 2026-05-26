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

### Failure taxonomy (in-flight, derived from evidence v3)

Schema v4's `result.failure_category` is not auto-filled yet (no v4
collector exists). To unblock the dashboard's "why didn't this cell
pass" view today,
[`tests/tools/compare_skills_value.py`](../tests/tools/compare_skills_value.py)
provides a `_classify_failure_mode(ev)` helper that derives the
failure mode heuristically from existing v3 fields. It returns
`None` when the cell passed (`_is_overall_pass`), otherwise exactly
one of the codes below.

The classifier checks F13 and F12 first because they exclude the cell
from comparability (and would otherwise be misread as a normal
failure shape). After that it follows the natural pipeline order:
artifact -> static -> semantic -> runtime.

| Code | Rule (first match wins) | Schema v4 mapping |
|---|---|---|
| `F13_infra_fail` | `_classify_validity == "infra_fail"` | `failure_category: F13` |
| `F12_incomplete` | `_classify_validity == "incomplete"` | (no direct v4 code; deferred) |
| `F10_no_artifact` | `agent.artifacts_found` contains no `kernel.py` | `failure_category: F10` |
| `F7_static` | kernel exists, `static_verify == "fail"` | `failure_category: F6` (static verification) |
| `F9_semantic` | kernel exists, static passes, `semantic_check.passed is False` | (no direct v4 code; closest is F1/F2 misuse) |
| `F8_correctness` | kernel exists, static + semantic pass, `verification.status == "fail"` and `verification.mode in {"simulator", "runtime"}` | `failure_category: F8` |
| `F0_unknown` | did not pass but doesn't fit any rule above | (no direct v4 code; surfaced so we notice new failure shapes) |

The dashboard renders these as small chips beneath the pass-rate
bars (`failure_mode_counts_on`, `failure_mode_counts_off` in the
per-profile aggregate). Skills-on counts every paired cell whose
on-leg didn't pass — including F13/F12 — so a broken on-leg surfaces
visibly. Skills-off counts only cells whose off-leg validity is `ok`,
so the off chips describe failure modes that meaningfully contribute
to the comparable baseline; F13/F12 are reported separately via the
existing off-validity row.

Today's data exhibits two dominant codes: `F8_correctness` (the
cloud-default `leaky_relu_f16` cell — kernel passes static and
semantic, simulator output is numerically wrong) and `F10_no_artifact`
(7B/8B local profiles where the agent never writes a kernel file
under either skills-on or skills-off, indicating the model/harness
capability boundary for those profiles).

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

## Summary-level additive fields (schema v2)

The top-level [`evidence/skills-value-summary.json`](../evidence/skills-value-summary.json)
document carries a few additive freshness/run-status fields that the
dashboard reads to surface what the most recent nightly actually
measured. They are emitted by
[`tests/tools/compare_skills_value.py`](../tests/tools/compare_skills_value.py)
and are explicitly part of the `schema_version: "2"` contract — older
v1 summary readers continue to work because all fields default to
`None` or `false`.

```yaml
schema_version: "2"
generated_at: "2026-05-20T08:00:00Z"

# True when the most recent nightly was partial — at least one matrix
# leg of `nightly-gate` or `local-stability-gate` was cancelled or
# failed before writing fresh evidence. The skills-value-report job
# preserves the previously-committed per-cell evidence in this case
# (and skips capabilities.yaml sync) to avoid mixing fresh and stale
# rows; only the summary itself is rewritten.
partial_run: false

# Verbatim copy of artifacts/legs-status.json from the CI run that
# produced this summary. `null` when the aggregator was invoked
# outside CI without `--legs-status-file`.
legs_status:
  needs:
    nightly-gate: success
    local-stability-gate: success
  partial_run: false
  legs:
    - {name: "nightly-gate (on)", conclusion: success}
    - {name: "nightly-gate (off)", conclusion: success}

cells:
  - op: abs
    dtype: float16
    profile: cloud-default
    on_date: "2026-05-19T12:00:00Z"   # ISO ts of the on-leg evidence file
    off_date: "2026-05-19T13:00:00Z"  # ISO ts of the off-leg evidence file
    # ... existing v2 fields ...

by_profile:
  cloud-default:
    # ... existing v2 fields ...
    on_last_run_at: "2026-05-19T12:00:00Z"
    off_last_run_at: "2026-05-19T13:00:00Z"
    # Age in whole days of the OLDEST off-leg evidence currently
    # classified as a clean baseline (validity == "ok"). Used by the
    # dashboard to flag profiles whose headline delta is built on
    # data from a previous nightly. `null` when no clean off-baseline
    # exists yet.
    off_max_staleness_days: 0

    # Explainability + intervention-efficiency additions.
    # All additive; older readers ignore them safely.

    # Per-leg breakdown of why cells didn't pass, as one entry per
    # F-code derived from existing v3 fields (see
    # "Failure taxonomy (in-flight, derived from evidence v3)").
    # `failure_mode_counts_on` counts every paired cell whose on-leg
    # did not pass (including F13/F12 to surface a broken on-leg).
    # `failure_mode_counts_off` counts only cells whose off-leg
    # validity is `ok`, so the off breakdown describes failure modes
    # that meaningfully contribute to the comparable baseline.
    # Empty dict (not `null`) when nothing is tallied.
    failure_mode_counts_on:  {F8_correctness: 1}
    failure_mode_counts_off: {}

    # 1-based mean / median of `attempts_to_pass` (index of the first
    # attempt that produced a usable kernel, see
    # `_attempts_to_pass`). Skills-on is averaged over every cell
    # whose on-leg passed; skills-off is averaged only over cells
    # whose off-leg validity is `ok` AND off-leg passed. `null` when
    # the leg never passed a cell on this profile. `*_n` records the
    # sample size so the dashboard can show how many cells fed the
    # average.
    attempts_to_pass_on_mean: 1.18
    attempts_to_pass_on_median: 1
    attempts_to_pass_on_n: 11
    attempts_to_pass_off_clean_mean: null
    attempts_to_pass_off_clean_median: null
    attempts_to_pass_off_clean_n: 0

    # Wilson 95% confidence interval on the displayed pass-rates,
    # rendered as `[low, high]` rounded to 3 decimals. `null` when
    # the denominator is zero (e.g. no clean off-baseline yet). The
    # dashboard appends a "(CI lo-hi%)" hint to the verdict so small
    # samples like 11/12 don't read as more precise than they are.
    pass_rate_on_ci: [0.646, 0.985]
    pass_rate_off_clean_ci: null
```

The corresponding workflow contract is:

- [`tests/tools/merge_evidence_artifacts.sh`](../tests/tools/merge_evidence_artifacts.sh)
  folds each matrix leg's fresh per-leg files into `evidence/` using a
  leg-specific glob, so cross-leg stale copies cannot win an overwrite
  race.
- [`tests/tools/detect_partial_run.py`](../tests/tools/detect_partial_run.py)
  classifies the current CI run as `partial_run: true|false` based on
  the per-job conclusions of `nightly-gate` and `local-stability-gate`
  reported by the GitHub Actions API.

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

## Protocol-axis CI mapping (Phase 0)

Phase 0 of the quarterly roadmap promotes the protocol axis to a
first-class CI concept. The collector
([`tests/tools/collect_generative_evidence.py`](../tests/tools/collect_generative_evidence.py))
takes a single `--protocol-id` flag that resolves the existing
`skills_mode` / `prompt_variant` / `agents_md` knobs in one shot.

### Mapping table

The mapping is the single source of truth for what each protocol id
means in CI. The collector enforces it; user-supplied `--skills-mode`
or `--prompt-variant` that contradict the derived values exit 1.

| Protocol id | name                              | `skills_mode` | `prompt_variant` | `agents_md` |
|-------------|-----------------------------------|---------------|------------------|-------------|
| `P2`        | `opencode-skills-off-minimal`     | `off`         | `minimal`        | `false`     |
| `P3`        | `opencode-skills-off-guided`      | `off`         | `guided`         | `false`     |
| `P4`        | `opencode-skills-off-agents-md`   | `off`         | `guided`         | `true`      |
| `P6`        | `opencode-skills-on-guided`       | `on`          | `guided`         | `false`     |

`agents_md=true` means the baseline AGENTS.md vendored at
[`docs/baseline/pyasc-fork-AGENTS.md`](baseline/pyasc-fork-AGENTS.md)
is copied into the test project as `AGENTS.md`. This is the **baseline**
AGENTS.md from the upstream pyasc-fork checkout, *not* the skill-stack
AGENTS.md at [`teams/pyasc-kernel-dev-team/AGENTS.md`](../teams/pyasc-kernel-dev-team/AGENTS.md)
(which is mounted only under `skills_mode=on`). Mixing the two would
conflate "skill value" with "baseline-AGENTS.md value", so the
collector refuses `--protocol-id P4 --skills-mode on` and
`--protocol-id P6 --agents-md ...`.

P5 (skills on + minimal prompt) and P7/P8 (oracle / human-assisted)
are documented above but not yet wired into the CI matrix — the
aggregator emits `P5-P2: null` until a future sprint adds the leg.

### Evidence filename scheme

Today's legacy `evidence/<op>-<dtype>-generative.json` is preserved as
a back-compat alias for the (cloud-default + P6) leg. The aggregator
continues to read it.

New per-protocol evidence files are written to:

```
evidence/<op>-<dtype>-generative-<profile>-<protocol_id_lower>.json
```

For example: `evidence/abs-f16-generative-cloud-default-p2.json`.

The existing `<op>-<dtype>-generative-<profile>-{on,off}.json` files
for local profiles are preserved; the aggregator treats `on` as P6
and `off` as P3 when no protocol-id file exists for that cell.

### Additive evidence-document fields

Schema version stays `"3"` — the `protocol` object is additive and
optional, so older readers ignore it safely. The collector populates
it from the derivation table:

```json
{
  "protocol": {
    "id": "P6",
    "name": "opencode-skills-on-guided",
    "prompt_variant": "guided",
    "skills_enabled": true,
    "allowed_context": {
      "task_prompt": true,
      "agents_md": false,
      "skills": true,
      "golden_kernels": false
    }
  }
}
```

The `agents_md` flag in `allowed_context` reflects whether the
baseline AGENTS.md was mounted into the project; `skills` reflects
the skill-stack mount. The existing top-level `skills_mode`,
`model_profile`, `prompt`, and `model` fields continue to be written
for back-compat with v3 readers that don't yet know about `protocol`.

### Aggregator summary fields

[`tests/tools/compare_skills_value.py`](../tests/tools/compare_skills_value.py)
groups by `(profile, protocol_id)` when `protocol.id` is present in
the evidence, and falls back to `(profile, skills_mode)` for legacy
files (`on` → P6, `off` → P3). It emits two additive blocks on the
existing `schema_version: "2"` summary:

```yaml
by_profile:
  cloud-default:
    # ... existing v2 fields ...
    by_protocol:
      P2: { pass_rate: 0.0,  attempts_to_pass_mean: null, tokens_mean: 12000, n_cells: 12, n_clean: 12 }
      P3: { pass_rate: 0.4,  attempts_to_pass_mean: 1.3,  tokens_mean: 25000, n_cells: 12, n_clean: 11 }
      P4: { pass_rate: 0.6,  attempts_to_pass_mean: 1.2,  tokens_mean: 26000, n_cells: 12, n_clean: 12 }
      P6: { pass_rate: 0.92, attempts_to_pass_mean: 1.1,  tokens_mean: 32000, n_cells: 12, n_clean: 12 }
    deltas_pp:
      "P3-P2": { pass_pp:  40, tokens_pct: 108, attempts_delta:  0.3 }
      "P4-P3": { pass_pp:  20, tokens_pct:   4, attempts_delta: -0.1 }
      "P6-P4": { pass_pp:  32, tokens_pct:  23, attempts_delta: -0.1 }
      "P5-P2": null  # P5 not yet run
```

[`tests/tools/sync_capabilities.py`](../tests/tools/sync_capabilities.py)
treats `P6` evidence (or the legacy alias) as authoritative for
`generative_status` updates; P2/P3/P4 are reporting-only.

## Capability cell metadata schema (Phase 1)

Phase 1 of the quarterly roadmap promotes each capability cell from
"prompt + shapes + status" to "prompt + shapes + status + a structured
self-description of what the cell claims to prove". The fields are
additive on `schema_version: "3"` — no schema bump, no prompt rewrite,
no kernel behavior change. Today's consumers
([check_capabilities.py](../tests/tools/check_capabilities.py),
[capabilities.yaml](../capabilities.yaml)) read them; downstream
dashboard / aggregator panels will land in Phase 2/3.

[docs/glossary.md](glossary.md) is the single source of truth for the
terminology used by these fields and for the standard pyasc / asc2 /
CANN vocabulary. The same file documents the ReLU scope decision
(`relu` is covered by pattern under the `abs` op's
`representative_of` list; see [docs/glossary.md §7](glossary.md#7-operator-coverage-notes)).

### Required fields per cell

| Field                 | Type / domain                                          | Notes                                                                  |
|-----------------------|--------------------------------------------------------|------------------------------------------------------------------------|
| `shape_regime`        | `fixed` / `runtime_size_only` / `dynamic`              | How the kernel handles input-shape variability.                        |
| `reduce_axis`         | `int` or `null`                                        | `-1` for last-axis reductions; `null` for non-reducing ops.            |
| `output_shape`        | `same_as_input` or `list`                              | Output shape descriptor; `[M, N]` placeholders allowed for matmul.     |
| `accumulator_dtype`   | `float16` / `float32` / `null`                         | Accumulator dtype; `null` iff the op does no accumulation.             |
| `identity`            | `"0"` / `"1"` / `"-inf"` / `"+inf"` / `null` (string!) | Stored as a string so YAML does not coerce `0` or `-inf`.              |
| `tail_behavior`       | `aligned_only` / `host_pad` / `mask` / `real_shape` / `host_dispatcher` / `unsupported` | How partial tiles are handled. |
| `padding`             | `int` (element count) or `null`                        | E.g. `OUT_PAD=8` for `reduce_sum/f32`.                                 |
| `partitioning`        | `row_per_core` / `tile_per_core` / `block_grid` / `host_dispatcher` | How work is distributed across cores.                       |
| `unsupported_regimes` | `list[str]`                                            | Free-form slugs; typo-guarded against the set in `check_capabilities.py`. |

Allowed values for the enum fields and the canonical
`unsupported_regimes` slug list are documented in
[docs/glossary.md §6](glossary.md#6-cell-metadata-enums).

### Cross-field consistency

[`check_capabilities.py`](../tests/tools/check_capabilities.py)
enforces the following invariants (hard-fail by default, demote to
warnings with `--no-strict-metadata`):

- `reduce_axis` is `null` iff `accumulator_dtype` is `null` — a
  reducing op has both, a non-reducing op has neither.
- Tier `elementwise` ⇒ `reduce_axis` must be `null`.
- Tier `reduction` ⇒ `reduce_axis` must be set.
- `tail_behavior == host_dispatcher` ⇔ `partitioning == host_dispatcher`.
- Every `unsupported_regimes` entry must be in the canonical set
  documented in [docs/glossary.md §6](glossary.md#6-cell-metadata-enums).

### Golden-header ↔ capabilities.yaml mirroring

Every golden under [golden/kernels/](../golden/kernels/) carries a
"Cell metadata (mirrors capabilities.yaml; do not drift)" block in
its top-of-file docstring. The
[tests/unit/tools/test-golden-header.sh](../tests/unit/tools/test-golden-header.sh)
PR-gate test greps each golden's docstring for the cell's
`shape_regime`, `tail_behavior`, and `partitioning` strings and fails
if either source has drifted from the other. Reporting and prompt
work that consumes these fields lands in Phase 2; until then the
metadata is read by `check_capabilities.py` only.
