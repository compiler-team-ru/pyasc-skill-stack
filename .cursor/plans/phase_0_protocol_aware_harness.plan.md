---
name: Phase 0 protocol-aware harness
overview: "Concrete sprint plan for Phase 0 of the quarterly roadmap. Adds a first-class protocol axis (P2/P3/P4/P6) to collect_generative_evidence.py, the AGENTS.md-only project layout, the 4-leg nightly matrix, and per-protocol aggregator/dashboard views. As of the 2026-05-22 precision audit, the bulk of stages 0.1–0.6 is implemented uncommitted in the worktree (28 modified files, 5 untracked paths) with both unit tests green and check_capabilities strict-mode passing; this revision demotes those stages to commit + audit and adds Stage 0.0 (land in-flight code) and Stage 0.8 (precision-gap closure)."
todos:
  - id: p0-0-land-inflight
    content: "Stage 0.0 (NEW): Land the in-flight Phase 0 worktree (separate the Phase 0 + Phase 1 + unrelated groups, three focused commits, one PR per group). Confirms the protocol-axis code path is reviewable, lint-clean, and tested before any further work."
    status: pending
  - id: p0-1-design
    content: "Stage 0.1 [DONE in worktree]: docs/evaluation-methodology.md \u00a7'Protocol-axis CI mapping (Phase 0)' at line 528 lays out the P2/P3/P4/P6 mapping, the additive evidence schema, and the filename convention. Audit during Stage 0.0 review."
    status: pending
  - id: p0-2-minimal-variants
    content: "Stage 0.2 [DONE in worktree]: All 12 cells now have prompt_variants.{minimal,guided}; check_capabilities.py._check_prompt_variants hard-fails on missing. Audit prompts against docs/prompt-template.md once Phase 2 ships."
    status: pending
  - id: p0-3-agents-md-layout
    content: "Stage 0.3 [DONE in worktree]: docs/baseline/pyasc-fork-AGENTS.md vendored with SHA pin (sha256 1544c058df24050b5edce6c7a69d0b00809e914b787fb1783e31885883d36cda); collect_generative_evidence has --agents-md-source/--no-agents-md and three-way layout. Add a re-vendoring guard in Stage 0.8."
    status: pending
  - id: p0-4-protocol-id-flag
    content: "Stage 0.4 [DONE in worktree]: --protocol-id wired, PROTOCOL_TABLE + derive_protocol() implemented, tests/unit/tools/test-protocol-derivation.sh passes, filename convention applied. Land + wire test into pr-gate during Stage 0.0."
    status: pending
  - id: p0-5-ci-matrix
    content: "Stage 0.5 [DONE in worktree]: .github/workflows/ci.yml nightly-gate matrix expanded to protocol_id: [P2,P3,P4,P6]; minimum-pass threshold gated to P6; per-protocol artifact upload; merge_evidence_artifacts.sh updated. Validate end-to-end via Stage 0.7."
    status: pending
  - id: p0-6-aggregator-dashboard
    content: "Stage 0.6 [DONE in worktree]: compare_skills_value.py parses *-p2/p3/p4/p6 filenames + legacy fallback (_LEGACY_MODE_TO_PROTOCOL), generate_dashboard.py + sync_capabilities.py + skills_value_smoke.py modified. Confirm by_protocol/deltas_pp emit correctly on a real run during Stage 0.7."
    status: pending
  - id: p0-7-min-experiment
    content: "Stage 0.7: Run the 4-leg matrix on abs/f16 locally with the now-landed code; check 4 evidence files exist with valid protocol.id and no infra_fail; verify aggregator emits by_protocol + deltas_pp; push branch and workflow_dispatch nightly to scale to 12 cells. PRECONDITION: Stage 0.0 must have landed first."
    status: pending
  - id: p0-8-precision-gaps
    content: "Stage 0.8 (NEW): Close known precision compromises uncovered during the audit. (a) Wire tests/unit/tools/test-protocol-derivation.sh into pr-gate; (b) add a re-vendoring guard for docs/baseline/pyasc-fork-AGENTS.md (re-compute the SHA on every pr-gate run); (c) tighten OP_SEMANTIC_MARKERS['gelu'] so the marker set encodes the dtype-conditioned form requirement (f16=erf-form, f32=tanh-form) instead of accepting either; (d) backfill evidence/* legacy files with an explicit protocol.id by re-running them so the aggregator does not need to fall back to _LEGACY_MODE_TO_PROTOCOL."
    status: pending
isProject: false
---

# Phase 0 — Protocol-aware harness

Drills into Phase 0 of [pyasc_skill_stack_quarterly_roadmap_aed2c154.plan.md](pyasc_skill_stack_quarterly_roadmap_aed2c154.plan.md) and only that phase. As originally planned this was ~5 engineer-days; with the in-flight worktree already covering stages 0.1–0.6, the remaining work fits in ~1.5 ED (Stage 0.0 land + Stage 0.7 first experiment + Stage 0.8 precision closure).

## Precision audit (2026-05-22)

A critical precision review against the live worktree found that stages 0.1–0.6 are already implemented but uncommitted. Evidence:

- [tests/tools/collect_generative_evidence.py](../../tests/tools/collect_generative_evidence.py) lines 62–96 contain `PROTOCOL_TABLE` and `derive_protocol()`; line 924 declares `--protocol-id {P2,P3,P4,P6}`; lines 942–988 enforce all the mismatch validations; line 1139 emits the additive `protocol` block in the evidence.
- [.github/workflows/ci.yml](../../.github/workflows/ci.yml) line 132 holds `protocol_id: ["P2", "P3", "P4", "P6"]` and lines 161–281 wire the per-protocol leg, artifact upload, and lowercase id.
- [tests/tools/compare_skills_value.py](../../tests/tools/compare_skills_value.py) lines 65–96 parse the protocol-id filenames and apply `_LEGACY_MODE_TO_PROTOCOL` to legacy outputs.
- [docs/baseline/pyasc-fork-AGENTS.md](../../docs/baseline/pyasc-fork-AGENTS.md) is vendored (195 lines, SHA pinned in the header).
- [docs/evaluation-methodology.md](../../docs/evaluation-methodology.md) line 528 ff. is the methodology section the design called for.
- All 12 cells in [capabilities.yaml](../../capabilities.yaml) have both `prompt_variants.minimal` and `prompt_variants.guided`.
- [tests/unit/tools/test-protocol-derivation.sh](../../tests/unit/tools/test-protocol-derivation.sh) and [tests/unit/tools/test-golden-header.sh](../../tests/unit/tools/test-golden-header.sh) both pass against the worktree.
- `python3 tests/tools/check_capabilities.py` passes for all 12 cells with the strict-metadata default.

Conclusion: the design + code path is solid. What is missing is (a) the commit + PR landing, (b) a real 4-leg run, (c) closure of small precision compromises. Stages have been demoted accordingly.

## Outcome

After this sprint, [tests/tools/collect_generative_evidence.py](../../tests/tools/collect_generative_evidence.py) can drive a generative run for any of the four protocols documented in [docs/evaluation-methodology.md](../../docs/evaluation-methodology.md) §"Generation protocol taxonomy" — P2 (minimal, skills off), P3 (guided, skills off), P4 (guided, AGENTS.md mounted, skills off), P6 (guided, skills on) — and CI exercises all four on `cloud-default` for every capability cell.

## Stage 0.0 — Land the in-flight Phase 0 worktree (~0.5 ED) — NEW

Discovered during the 2026-05-22 precision audit: 28 modified files + 5 untracked paths in the worktree implement most of stages 0.1–0.6 without any commit. The risk of carrying this forward (rebase pain, partial state, no PR review) is higher than the cost of landing it now.

Partition the worktree into three commit groups; land each as its own PR.

**Group A — Phase 0 protocol axis (this sprint):**

- [tests/tools/collect_generative_evidence.py](../../tests/tools/collect_generative_evidence.py)
- [tests/tools/compare_skills_value.py](../../tests/tools/compare_skills_value.py)
- [tests/tools/generate_dashboard.py](../../tests/tools/generate_dashboard.py)
- [tests/tools/sync_capabilities.py](../../tests/tools/sync_capabilities.py)
- [tests/tools/skills_value_smoke.py](../../tests/tools/skills_value_smoke.py)
- [tests/tools/merge_evidence_artifacts.sh](../../tests/tools/merge_evidence_artifacts.sh)
- [.github/workflows/ci.yml](../../.github/workflows/ci.yml)
- [docs/evaluation-methodology.md](../../docs/evaluation-methodology.md) (the §"Protocol-axis CI mapping (Phase 0)" section at line 528)
- [docs/baseline/pyasc-fork-AGENTS.md](../../docs/baseline/pyasc-fork-AGENTS.md) (untracked, vendor it)
- [tests/unit/tools/test-protocol-derivation.sh](../../tests/unit/tools/test-protocol-derivation.sh) (untracked, add and wire into pr-gate)
- `capabilities.yaml` rows for `prompt_variants.minimal` only (split this hunk from Group B)

**Group B — Phase 1 spec hygiene (deferred to Phase 1 plan; see [phase_1_spec_hygiene.plan.md](phase_1_spec_hygiene.plan.md) Stage 1.0):**

- All 10 `golden/kernels/*.py` header blocks
- [capabilities.yaml](../../capabilities.yaml) Phase 1 metadata fields (`shape_regime`, `reduce_axis`, `output_shape`, `accumulator_dtype`, `identity`, `tail_behavior`, `padding`, `partitioning`, `unsupported_regimes`)
- [tests/tools/check_capabilities.py](../../tests/tools/check_capabilities.py) `_check_cell_metadata` + `_check_prompt_variants` + `--strict-metadata`/`--no-strict-metadata`/`--soft-runtime`
- [docs/glossary.md](../../docs/glossary.md) (untracked, add)
- [tests/unit/tools/test-golden-header.sh](../../tests/unit/tools/test-golden-header.sh) (untracked, add)

**Group C — Unrelated platform / setup work (out of Phase 0/1 scope; ship separately or via Phase 8):**

- `.gitignore`, `README.md`, `docker/Dockerfile`, `docker/Dockerfile.overlay`, `docs/cann-setup.md`, `skills/pyasc-env-check/SKILL.md`
- Deleted files: `docker/pyasc-overlay/asc_changed/language/core/ir_value.py`, `docker/pyasc-overlay/asc_changed/language/core/range.py`, `golden/docs/python-api/language/core.md`
- Untracked: `docker/pyasc-overlay/asc_C/`, `scripts/install-host-deps.sh`, `docs/manual-review-order.md`, `.cursor/plans/rms_norm_platform_fix_and_gelu_tanh_9c16b5ec.plan.md` (the historical plan record)

Pre-PR gate sequence per group:

```bash
git add -p <group A files>
git commit -m "feat(phase-0): protocol-aware harness for P2/P3/P4/P6"
python3 tests/tools/check_capabilities.py
bash tests/unit/tools/test-protocol-derivation.sh
bash tests/unit/tools/run-tests.sh  # if a wrapper exists, otherwise run individuals
```

Acceptance: each PR is reviewable in isolation, pr-gate green, no straggling worktree state remains for Phase 0/1 files. Group C is *enumerated* here only so it doesn't get accidentally swept into Group A or B; its triage happens in Phase 8 Stage 8.4.

## Stage 0.1 — Design: additive schema + filename scheme (~0 ED) — VERIFY ONLY

This stage was completed in the worktree. Verify during Stage 0.0 PR review.

- Filename scheme:
  - Legacy `evidence/<op>-<dtype>-generative.json` is preserved as a back-compat alias for `cloud-default + P6` (today's primary `on` leg). The aggregator continues to read it.
  - New per-protocol files: `evidence/<op>-<dtype>-generative-<profile>-<protocol_id_lower>.json`, e.g. `evidence/abs-f16-generative-cloud-default-p2.json`.
  - Existing `<op>-<dtype>-generative-<profile>-{on,off}.json` files for local profiles are preserved; the aggregator treats `on` as P6 and `off` as P3 when no protocol_id file exists for that cell.
- Schema additions (still `schema_version: "3"`, additive optional):

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

- Mapping table from `--protocol-id` to existing flags. This is the single source of truth that 0.3 and 0.5 read:
  - P2 → `skills_mode=off`, `prompt_variant=minimal`, `agents_md=false`.
  - P3 → `skills_mode=off`, `prompt_variant=guided`, `agents_md=false`.
  - P4 → `skills_mode=off`, `prompt_variant=guided`, `agents_md=true`.
  - P6 → `skills_mode=on`, `prompt_variant=guided`, `agents_md=false`.
- Deliverable: ~30-line section appended to [docs/evaluation-methodology.md](../../docs/evaluation-methodology.md) titled "Protocol-axis CI mapping (Phase 0)".

## Stage 0.2 — Complete `prompt_variants.minimal` for every cell (~0.5 ED)

Today only `abs/{f16,f32}` and `gelu/f16` have a `prompt_variants.minimal`. Without minimal prompts for the remaining 9 cells, P2 is unmeasurable. Add minimal prompts following the labeling rule in [docs/evaluation-methodology.md](../../docs/evaluation-methodology.md) §"Prompt-variant labeling rules": *operator name, dtype, shape, semantic definition only*.

Cells needing `prompt_variants.minimal` in [capabilities.yaml](../../capabilities.yaml):

- `add/f16`
- `reduce_sum/f32`
- `reduce_sum/f16`
- `gelu/f32`
- `leaky_relu/f16`
- `softmax/f16`
- `matmul/f16`
- `rms_norm/f16`
- `rms_norm/f32`

Update [tests/tools/check_capabilities.py](../../tests/tools/check_capabilities.py) `_check_cell` to require both `prompt_variants.minimal` and `prompt_variants.guided` for every cell. Add a one-line warning (not failure) when a cell is missing either, then promote to hard-fail once the cells are filled in.

Deliverable: 9 new minimal prompts in [capabilities.yaml](../../capabilities.yaml); updated check_capabilities; one local pr-gate run green.

## Stage 0.3 — AGENTS.md-only project layout (~1 ED)

Today [tests/tools/collect_generative_evidence.py](../../tests/tools/collect_generative_evidence.py) `create_test_project` has two layouts (`on` and `off`). Neither matches P4 ("OpenCode + AGENTS.md, no skills"). Add a third layout.

Concrete edits in [tests/tools/collect_generative_evidence.py](../../tests/tools/collect_generative_evidence.py):

- Vendor the baseline AGENTS.md once into the repo: copy `/home/aloschilov/workspace/pyasc-fork/AGENTS.md` to `docs/baseline/pyasc-fork-AGENTS.md` and pin its SHA in a header comment. (CI must not depend on a sibling checkout.)
- Add `--agents-md-source` argparse flag, defaulting to `docs/baseline/pyasc-fork-AGENTS.md` relative to `REPO_ROOT`. Add `--no-agents-md` to suppress mounting.
- Extend `create_test_project(skills_mode, profile, ..., agents_md_path: Path | None)`:
  - If `agents_md_path` is given, copy it to `<project>/AGENTS.md` *after* the other layout work, regardless of `skills_mode`.
  - Document the precedence: if `skills_mode=on` *and* `agents_md_path` is given, the user gets both AGENTS.md *and* the skills team's AGENTS.md. The current `on` leg already vendors `teams/pyasc-kernel-dev-team/AGENTS.md`; that is the skill-stack AGENTS.md, not the pyasc-fork baseline. They are intentionally distinct files (see [teams/pyasc-kernel-dev-team/AGENTS.md](../../teams/pyasc-kernel-dev-team/AGENTS.md) vs the new baseline copy). For P4 we want the baseline only.
- Refuse contradiction: `--protocol-id P4` with `--skills-mode on` exits 1 with a clear message.

Deliverable: new layout, new flag, the AGENTS.md baseline vendored, one local probe run for `abs/f16 + P4` produces an evidence file with `protocol.id == "P4"` and the kernel project directory contains exactly `golden/`, `AGENTS.md`, `opencode.json`.

## Stage 0.4 — `--protocol-id` flag + derivation logic (~1 ED)

Concrete edits in [tests/tools/collect_generative_evidence.py](../../tests/tools/collect_generative_evidence.py):

- Add `--protocol-id {P2,P3,P4,P6}` to argparse.
- When supplied, derive `skills_mode`, `prompt_variant`, `agents_md` per the 0.1 mapping table. If user also passes `--skills-mode` or `--prompt-variant`, validate match; on mismatch exit 1 with a diff.
- Filename selection in `main`:
  - If `--protocol-id` is given: emit `<op>-<dtype>-generative-<profile>-<protocol_id_lower>.json`. The legacy `cloud-default + on` short name is no longer used in this code path.
  - Otherwise: today's behavior unchanged (legacy short name for `cloud-default + on`, `*-<profile>-<mode>.json` for everything else).
- Evidence document additions:
  - Add `protocol` object (per 0.1 schema). Populate `name` from a small static map (`P6 -> opencode-skills-on-guided`, etc).
  - Continue to write the existing `skills_mode`, `model_profile`, `prompt` fields for back-compat.
- Add a one-line `print` summary of the resolved protocol at the start of each run.

Mermaid summarizing the resolved flow:

```mermaid
flowchart LR
    User["CI / dev"] --> Flag["--protocol-id P2|P3|P4|P6"]
    Flag --> Derive["derive skills_mode + prompt_variant + agents_md"]
    Derive --> Project["create_test_project (skills_mode, agents_md_path)"]
    Project --> Run["opencode run --dir project"]
    Run --> Evidence["evidence/<op>-<dtype>-generative-<profile>-<protocol_id>.json with protocol.id"]
```

Deliverable: working `--protocol-id` flag; unit test in [tests/unit/tools/](../../tests/unit/tools/) asserts the derivation table; one dry-run per protocol on `abs/f16` writes the expected filename.

## Stage 0.5 — CI matrix expansion to 4 legs (~1 ED)

Concrete edits in [.github/workflows/ci.yml](../../.github/workflows/ci.yml):

- `nightly-gate.strategy.matrix`: replace `skills_mode: ["on", "off"]` with `protocol_id: ["P2", "P3", "P4", "P6"]`.
- In the "Run OpenCode skills-intervention leg" step, replace `--skills-mode "$SKILLS_MODE"` with `--protocol-id "$PROTOCOL_ID"`. The body of the loop is otherwise unchanged.
- Adjust the per-leg minimum-pass threshold gate (today only on `SKILLS_MODE == on`) to apply only when `PROTOCOL_ID == "P6"`. P2/P3/P4 are explicitly allowed to underperform.
- Upload-artifact step `path:` expression: switch from the ternary on `skills_mode` to a glob keyed on `protocol_id`: `evidence/*-generative-cloud-default-${{ matrix.protocol_id_lower }}.json`. (Compute `protocol_id_lower` via `env` with a small `bash`-side `tr '[:upper:]' '[:lower:]'` to avoid GitHub Actions expression limitations.)
- Artifact name: `evidence-cloud-${{ matrix.protocol_id }}` instead of `evidence-cloud-${{ matrix.skills_mode }}`.
- `local-stability-gate`: leave untouched in this sprint (still uses `skills_mode: ["on", "off"]`). Doubling local-model cost is a Phase 6 decision.
- Update [tests/tools/merge_evidence_artifacts.sh](../../tests/tools/merge_evidence_artifacts.sh) so the new `evidence-cloud-P*` artifact directories are folded into `evidence/` with their existing per-leg-specific glob discipline.

Deliverable: one workflow_dispatch dry-run on a feature branch with `tier=nightly` produces 12 cells × 4 protocols = 48 evidence files in `cloud-default` (modulo failures), no overlap with the legacy filenames, and `skills-value-report` runs the aggregator without crashing on the new files.

## Stage 0.6 — Aggregator + dashboard per-protocol view (~1 ED)

Concrete edits.

- [tests/tools/compare_skills_value.py](../../tests/tools/compare_skills_value.py):
  - Extend the evidence scan to discover `<op>-<dtype>-generative-<profile>-<protocol_id>.json` in addition to today's legacy and `<profile>-{on,off}` patterns.
  - Group by `(profile, protocol_id)` when `protocol.id` is present; fall back to `(profile, skills_mode)` for legacy files.
  - Emit new top-level fields in `evidence/skills-value-summary.json` (additive on `schema_version: "2"`):

```yaml
schema_version: "2"
by_profile:
  cloud-default:
    by_protocol:
      P2: { pass_rate: 0.0, attempts_to_pass_mean: null, tokens_mean: 12000, n_cells: 12, n_clean: 12 }
      P3: { pass_rate: 0.4, attempts_to_pass_mean: 1.3, tokens_mean: 25000, n_cells: 12, n_clean: 11 }
      P4: { pass_rate: 0.6, attempts_to_pass_mean: 1.2, tokens_mean: 26000, n_cells: 12, n_clean: 12 }
      P6: { pass_rate: 0.92, attempts_to_pass_mean: 1.1, tokens_mean: 32000, n_cells: 12, n_clean: 12 }
    deltas_pp:
      "P3-P2": { pass_pp: 40, tokens_pct: 108, attempts_delta: 0.3 }
      "P4-P3": { pass_pp: 20, tokens_pct: 4, attempts_delta: -0.1 }
      "P6-P4": { pass_pp: 32, tokens_pct: 23, attempts_delta: -0.1 }
      "P5-P2": null  # P5 not yet run
```

  - Continue to emit the existing `pass_rate_off`, `pass_rate_off_clean`, `viability_unlocked_clean`, `delta_*` per-cell fields so the v1 dashboard keeps rendering.
- [tests/tools/sync_capabilities.py](../../tests/tools/sync_capabilities.py):
  - Treat the `P6` evidence (or its legacy alias) as the authoritative input for `generative_status` updates. Other protocols are reporting-only.
- [tests/tools/generate_dashboard.py](../../tests/tools/generate_dashboard.py):
  - Add a "Skill stack value decomposition" panel rendering the 4 deltas for `cloud-default`.
  - When a protocol leg has zero clean cells, render `"protocol Pn unavailable: ..."` per the existing headline-rendering convention.
- Update the smoke test [tests/tools/skills_value_smoke.py](../../tests/tools/skills_value_smoke.py) to include at least one P2 + P3 + P4 + P6 fixture file; assert the aggregator emits `by_protocol` and `deltas_pp` correctly.

Deliverable: aggregator handles both legacy and protocol-id evidence files in one pass; smoke test green; one local dashboard regen renders the new panel.

## Stage 0.7 — Minimum first experiment: abs/f16 × 4 protocols (~0.5 ED)

End-to-end validation of stages 0.1–0.6 on a single cell before scaling to all 12. PRECONDITION: Stage 0.0 must have landed (Group A merged into the active branch) so the harness changes are unambiguously in scope.

Commands (run locally on the host with CANN sourced):

```bash
for p in P2 P3 P4 P6; do
  python3.10 tests/tools/collect_generative_evidence.py \
    --op abs --dtype float16 \
    --runtime --timeout 420 --docker-timeout 1500 \
    --model-profile cloud-default \
    --protocol-id "$p" \
    --max-attempts 3 \
    --archive-dir generative-archive \
    --notes "Phase 0 minimum first experiment ($p)"
done

python3.10 tests/tools/compare_skills_value.py \
  --output evidence/skills-value-summary.json \
  --markdown skills-value-report.md
```

Acceptance:

- 4 evidence files exist at `evidence/abs-f16-generative-cloud-default-p{2,3,4,6}.json`.
- Each has `protocol.id` set correctly and populated `tokens`, `kernel_path`, `agent.artifacts_found`, `verification.status`.
- None are classified as `validity=infra_fail` by the aggregator (this is the smoke check that the AGENTS.md path and the minimal-prompt path both produce comparable runs).
- `evidence/skills-value-summary.json` `by_profile.cloud-default.by_protocol` has all 4 keys populated.
- `evidence/skills-value-summary.json` `by_profile.cloud-default.deltas_pp` has all 3 deltas populated (P5-P2 stays null).

If green: push the feature branch and dispatch the nightly with `tier=nightly` to scale to all 12 cells. Expected end state: 48 fresh evidence files plus `skills-value-summary.json` showing the 4-protocol decomposition.

If any leg fails the validity check, do not scale — fix the harness for that leg first.

## Stage 0.8 — Close precision compromises (~0.5 ED) — NEW

Four small precision gaps emerged in the audit. They are not blockers for the 4-leg experiment but they should land in the same sprint so Phase 0 is genuinely "done", not "done-ish".

### 0.8.a — Wire the protocol-derivation test into pr-gate

[tests/unit/tools/test-protocol-derivation.sh](../../tests/unit/tools/test-protocol-derivation.sh) exists and passes but is not invoked by any CI job. Without that wiring, future edits to `PROTOCOL_TABLE` can drift silently. Add the script to `tests/unit/tools/run-tests.sh` (or whichever wrapper pr-gate calls). One-line change; deliverable is the wrapper edit + green pr-gate.

### 0.8.b — Re-vendoring guard on docs/baseline/pyasc-fork-AGENTS.md

The vendored baseline has a SHA in its header (`1544c058df24050b5edce6c7a69d0b00809e914b787fb1783e31885883d36cda`). Today nothing verifies that SHA against the file body, so a silent hand-edit would not be caught. Add a small check:

```bash
# tests/unit/tools/test-baseline-agents-md.sh (new)
expected=$(grep -E '^\s+sha256:\s+' docs/baseline/pyasc-fork-AGENTS.md | awk '{print $2}')
got=$(sed -n '/^-->$/,$p' docs/baseline/pyasc-fork-AGENTS.md | tail -n +2 | sha256sum | awk '{print $1}')
[ "$expected" = "$got" ] || { echo "FAIL: baseline AGENTS.md SHA drift ($expected vs $got)"; exit 1; }
```

Wire into pr-gate (same wrapper as 0.8.a).

### 0.8.c — Tighten gelu semantic markers (form discriminator)

Today [tests/tools/semantic_markers.py](../../tests/tools/semantic_markers.py) line 21 says:

```python
"gelu": ["asc2.tanh", "0.044715", "asc2.erf", "gelu"],
```

With the per-marker check being "any match", a gelu/f32 generation that uses the erf form would pass markers — directly contradicting the gelu/f32 guided prompt which says "do NOT call asc2.erf". The marker check should be dtype-aware so f16 accepts erf form and f32 requires tanh form. Concrete fix:

```python
OP_SEMANTIC_MARKERS: dict[str, list[str] | dict[str, list[str]]] = {
    ...
    "gelu": {
        "float16": ["asc2.erf", "gelu"],
        "float32": ["asc2.tanh", "0.044715"],
    },
    ...
}
```

…with [tests/tools/score_kernel.py](../../tests/tools/score_kernel.py) and [tests/tools/collect_generative_evidence.py](../../tests/tools/collect_generative_evidence.py) consuming `OP_SEMANTIC_MARKERS[op]` via a small helper that resolves `dict | list[str]` against the `dtype`. Add a unit test `tests/unit/tools/test-semantic-markers-gelu-dtype.sh` to lock the discriminator. This also unblocks Phase 2 (it lets the prompt template encode "the markers enforce the form for me").

### 0.8.d — Backfill legacy `evidence/*` files with explicit `protocol.id`

Today 12 legacy `evidence/<op>-<dtype>-generative.json` files exist (no profile / no protocol suffix). The aggregator's `_LEGACY_MODE_TO_PROTOCOL = {"on": "P6", "off": "P3"}` falls back to "treat legacy on as P6" — correct semantics, but a *fallback* still on the read path. Re-run those 12 cells under `--protocol-id P6` once the harness lands, then delete the legacy short-name files. After this, `compare_skills_value._LEGACY_MODE_TO_PROTOCOL` becomes a defensive fallback for archived runs only; the live nightly is fully protocol-tagged.

Acceptance: `evidence/` has no `*-generative.json` files without a profile/protocol suffix; aggregator behavior is unchanged; smoke fixture green.

### 0.8 acceptance

- `tests/unit/tools/run-tests.sh` (or equivalent) invokes `test-protocol-derivation.sh` and `test-baseline-agents-md.sh`; both pass in pr-gate.
- `OP_SEMANTIC_MARKERS["gelu"]` is dtype-keyed; a kernel using erf-form for f32 now fails `semantic_check`; gelu/f16 + gelu/f32 evidence regenerated.
- `evidence/*-generative.json` (legacy short name) is empty; every active evidence file carries `protocol.id`.

## Definition of done for Phase 0

- Stage 0.0 PRs landed: Group A (Phase 0 axis) merged into the active branch; Group B held for the Phase 1 plan; Group C explicitly enumerated for Phase 8 triage.
- `--protocol-id` flag in [tests/tools/collect_generative_evidence.py](../../tests/tools/collect_generative_evidence.py) with the derivation table tested in pr-gate (Stage 0.8.a).
- All 12 cells in [capabilities.yaml](../../capabilities.yaml) have both `prompt_variants.minimal` and `prompt_variants.guided`; [tests/tools/check_capabilities.py](../../tests/tools/check_capabilities.py) enforces.
- `docs/baseline/pyasc-fork-AGENTS.md` vendored with a SHA pin *and* the SHA verified by a pr-gate test (Stage 0.8.b).
- CI `nightly-gate` runs 4 legs (P2/P3/P4/P6) for `cloud-default` per dispatch.
- `evidence/skills-value-summary.json` exposes `by_protocol` and `deltas_pp` for `cloud-default`.
- Dashboard renders the "Skill stack value decomposition" panel.
- One full nightly green: 48 evidence files written, no infra_fail, `P6 ≥ 9/12` cells passing (today's `on` baseline).
- `OP_SEMANTIC_MARKERS['gelu']` is dtype-keyed and gelu/f32 evidence regenerated (Stage 0.8.c).
- Legacy `evidence/*-generative.json` short names cleared in favour of protocol-tagged files (Stage 0.8.d).

## Risks specific to Phase 0

- **AGENTS.md duplicate role.** The skills-on layout already mounts [teams/pyasc-kernel-dev-team/AGENTS.md](../../teams/pyasc-kernel-dev-team/AGENTS.md), which is the *skill-stack* AGENTS.md. The vendored `docs/baseline/pyasc-fork-AGENTS.md` is the *baseline* AGENTS.md. Mixing them under P6 would conflate "skill value" with "baseline AGENTS.md value". Mitigation: P6 keeps the skill-stack AGENTS.md only; P4 keeps the baseline AGENTS.md only. The derivation logic in Stage 0.4 enforces this — `--protocol-id P6` does *not* mount the baseline.
- **Token cost.** Doubling the matrix from 2 legs to 4 legs on `cloud-default` is +~2× nightly OpenCode spend. If the budget binds, route P2 and P4 to a less expensive schedule (e.g. weekly) by adding a second `workflow_dispatch.tier` value `protocol-full` once Phase 3 wants the full 4 legs nightly.
- **Schema confusion.** Adding optional `protocol` to `schema_version: "3"` documents could surprise v3-strict readers. The repo doesn't have any (every reader is permissive), but call this out in the [docs/evaluation-methodology.md](../../docs/evaluation-methodology.md) update so the eventual v4 bump in Phase 7 stays clean.
- **`prompt_variants.minimal` quality.** A too-terse minimal prompt may produce an `infra_fail`-shaped failure that is actually expected at P2 (model gave up). The aggregator's `_classify_validity` flags `infra_fail` only when `model in {null, ""} AND tokens.total == 0 AND agent.artifacts_found == [] AND kernel_path == ""`; a P2 run that produced *some* tokens and no kernel will be `validity=ok, F10_no_artifact`, which is the desired classification. Verify on a Stage 0.7 P2 run before scaling.
- **Back-compat for `local-stability-gate`.** This sprint leaves `local-stability-gate` on the legacy `skills_mode: [on, off]` matrix. The aggregator must therefore handle a *mixture* of legacy filenames (local profiles) and new protocol-id filenames (cloud-default) in one pass. Tested in Stage 0.6 smoke fixture.

## Deferred from Phase 0 (intentionally)

- P5 (minimal prompt + skills on). Documented in [docs/evaluation-methodology.md](../../docs/evaluation-methodology.md) §"Comparisons of interest" but not in the CI matrix yet. The aggregator emits `P5-P2: null` until a future sprint adds the leg.
- P7 (oracle-guided, golden-kernels mounted) and P8 (human-assisted). Diagnostic upper bounds only, per the methodology doc.
- Migrating `local-stability-gate` to the 4-protocol matrix. Doubles local-model wall-clock budget; revisit in Phase 6.
- Schema v4 + `failure_category` auto-fill. Phase 7.
- The dashboard "view C" (model/profile capability matrix). Phase 7+.
