---
name: Phase 7 schema v4 and failure taxonomy
overview: "Concrete sprint plan (~7 engineer-days) for Phase 7 of the quarterly roadmap. Bumps evidence schema_version from 3 to 4 in collect_generative_evidence.py with full protocol/model/agent/result population, moves the heuristic failure-mode classifier from compare_skills_value.py into the collector and emits result.failure_category (F1..F13) directly, migrates compare_skills_value.py + sync_capabilities.py + generate_dashboard.py to read v4 fields preferentially with v3 fallback, and adds tests/integration/test_process_as_tests.py enforcing the process-as-tests checklist from notes section 5."
todos:
  - id: p7-1-schema-design
    content: "Stage 7.1: Schema v4 field design + back-compat policy — extend docs/evaluation-methodology.md schema-v4 section into the actual emission contract, including failure_category derivation rules at collection time (not at aggregation time), and an additive-only migration matrix that pins v3 reader compatibility."
    status: pending
  - id: p7-2-collector-emission
    content: "Stage 7.2: Implement v4 emission in tests/tools/collect_generative_evidence.py — populate protocol.{id,name,prompt_variant,skills_enabled,skill_set_hash,allowed_context}, model.{provider,model_id,model_version_or_date,temperature,seed}, agent.{harness,harness_version,tools_allowed,max_attempts,timeout_s}, result.{artifact_found,artifact_accepted,agent_clean_completion,timeout_after_success,static_verify,jit_verify,simulator_verify,score,failure_category}; SCHEMA_VERSION = 4; preserve every v3 field for one quarter of back-compat."
    status: pending
  - id: p7-3-failure-classifier
    content: "Stage 7.3: Move the heuristic _classify_failure_mode helper from compare_skills_value.py into collect_generative_evidence.py; expand from 8 derived codes to the full F1..F13 set; add F1/F2/F5 classifiers that read kernel source + verification stderr (today these collapse to F0_unknown)."
    status: pending
  - id: p7-4-readers-v4
    content: "Stage 7.4: Update compare_skills_value.py, sync_capabilities.py, generate_dashboard.py to read v4 fields preferentially with v3 fallback (additive contract); add a one-line schema_version dispatch at the top of each reader; remove the derived classifier from compare_skills_value.py (now redundant)."
    status: pending
  - id: p7-5-integration-test
    content: "Stage 7.5: Add tests/integration/test_process_as_tests.py asserting every nightly's evidence files produce: prompt artifact, model identity, token capture, trajectory log, compile/test command record, archived generated code, evidence JSON, classified failure mode; wire into ci-gate.sh nightly tier; document the contract in docs/evaluation-methodology.md."
    status: pending
isProject: false
---

# Phase 7 — Schema v4 and failure taxonomy

Drills into Phase 7 of [pyasc_skill_stack_quarterly_roadmap_aed2c154.plan.md](pyasc_skill_stack_quarterly_roadmap_aed2c154.plan.md) and only that phase. Sized at ~7 engineer-days across ~2 weeks. The rigor layer; lands once Phase 0–6 data shape is stable, otherwise schema churn races the experiments.

## Outcome

After this sprint, every evidence file emitted by [tests/tools/collect_generative_evidence.py](../../tests/tools/collect_generative_evidence.py) carries the full v4 contract documented in [docs/evaluation-methodology.md](../../docs/evaluation-methodology.md) §"Evidence schema v4 (proposed)" — protocol id, model identity, agent harness identity, separated `artifact_found / artifact_accepted / agent_clean_completion / timeout_after_success`, a populated `failure_category` from F1..F13. The aggregator stops needing its heuristic `_classify_failure_mode` because the collector emits the category directly. A process-as-tests integration test enforces the contract that the matrix is testable and reproducible (notes §5).

## Stage 7.1 — Schema v4 design + back-compat policy (~1 ED)

Promote the schema-v4 specification in [docs/evaluation-methodology.md](../../docs/evaluation-methodology.md) §"Evidence schema v4 (proposed)" from "proposed" to the actual emission contract. Two additions over the existing proposal:

- **`result.failure_category` is set by the collector.** The doc currently says "no v4 collector exists" and the aggregator derives the category heuristically; Phase 7 reverses this. Each pipeline failure point (artifact missing, static failure, semantic failure, runtime failure, infra failure) writes its own F-code immediately. F0_unknown stays as the catch-all but should be empty in practice.
- **`schema_version: "4"` is emitted, all v3 fields are preserved.** A v3 reader continues to read the file as a v3 document with extra unknown keys; a v4 reader prefers the v4 fields. After one quarter of dual emission, v3 fields can be retired.

Back-compat matrix (additive-only, no breaking changes this sprint):

- `skills_mode` (v3) ← derived from `protocol.skills_enabled` (v4) but still emitted verbatim.
- `model_profile` (v3) ← kept verbatim as top-level grouping key.
- `model` (v3) ← `model.model_id` (v4) but also emitted at top level.
- `tokens.{input,output,cache_read,total}` (v3) ← unchanged in v4.
- `agent.platform` (v3) ← `agent.harness` (v4) plus the top-level alias.
- `agent.artifacts_found` (v3) ← `result.artifact_found = (len(artifacts_found) > 0)` (v4) plus the original list.
- `kernel_path` + `score.accepted` + `verification.status` (v3) ← `result.artifact_accepted` (v4) — derived once at collection.
- `agent.completed` (v3) ← `result.agent_clean_completion` (v4).
- `static_verify` (v3) ← `result.static_verify` (v4).
- `verification.status` mode=simulator (v3) ← `result.simulator_verify` (v4).
- `score` (v3) ← `result.score` (v4).
- New v4: `result.failure_category`, all `protocol.*`, `model.provider/version/temperature/seed`, `agent.harness_version`, `agent.tools_allowed`, `result.timeout_after_success`, `result.performance_verified=null`.

Deliverable: schema-v4 §-section in [docs/evaluation-methodology.md](../../docs/evaluation-methodology.md) updated from "proposed" to "v4 collector emits this".

## Stage 7.2 — Collector emits v4 (~2 ED)

Concrete edits in [tests/tools/collect_generative_evidence.py](../../tests/tools/collect_generative_evidence.py).

- Bump `SCHEMA_VERSION = "4"`.
- Populate `protocol`:
  - `id`: from `--protocol-id` (Phase 0) or derived `--skills-mode` + `--prompt-variant`.
  - `name`: static map (`P2 → opencode-skills-off-minimal`, `P3 → opencode-skills-off-guided`, `P4 → opencode-skills-off-guided-agents-md`, `P6 → opencode-skills-on-guided`).
  - `prompt_variant`: from `--prompt-variant`.
  - `skills_enabled`: from `--skills-mode`.
  - `skill_set_hash`: SHA256 of the contents of every file under [skills/](../../skills/) at run time (cheap on a few KB).
  - `allowed_context`: derived from the project layout + AGENTS.md source + cell's `examples_policy` (Phase 2).
- Populate `model`:
  - `provider`, `model_id`: parsed from the resolved opencode.json (existing `resolve_configured_model`).
  - `model_version_or_date`: opencode CLI does not provide this; record `null` for cloud-default; for local models, record the Ollama digest (`ollama show <model> | grep digest`).
  - `temperature`, `seed`: read from the resolved opencode.json `provider.*.options` if present; else `null`.
- Populate `agent`:
  - `harness`: "opencode".
  - `harness_version`: from `opencode --version` (a one-shot capture at run start).
  - `tools_allowed`: from the resolved opencode.json `permission` keys (existing `_MINIMAL_OPENCODE_JSON.permission`).
  - `max_attempts`, `timeout_s`: from argparse.
- Populate `result`:
  - `artifact_found = bool(kernel_path)`.
  - `artifact_accepted = static_verify == "pass" AND semantic_check.passed AND score.accepted AND (verification.status == "pass" OR not runtime)`.
  - `agent_clean_completion = agent_completed AND exit_code in (0, 0)`.
  - `timeout_after_success = artifact_accepted AND exit_code == 124`.
  - `static_verify`, `jit_verify`, `simulator_verify`: from the existing static/JIT/simulator pipeline.
  - `score`: existing `score_data`.
  - `failure_category`: from Stage 7.3's collector classifier.
  - `performance_verified`: `null` until the perf layer ships.

All v3 fields stay; new fields are additive.

Deliverable: collector emits v4; one dry-run on `abs/f16` produces a v4-shaped JSON; one diff against a recent v3 file shows additive-only changes.

## Stage 7.3 — Failure classifier moves into the collector (~1.5 ED)

Today the F-code is derived at aggregation time from v3 fields in [tests/tools/compare_skills_value.py](../../tests/tools/compare_skills_value.py) `_classify_failure_mode`. Move it.

- Lift `_classify_failure_mode(ev)` into [tests/tools/collect_generative_evidence.py](../../tests/tools/collect_generative_evidence.py) as `_classify_failure(...)`. Same call signature (takes the in-progress evidence dict), same first-match-wins discipline.
- Expand from the existing 8 derived codes to the full F1..F13 set:
  - `F13 infra/config failure`: `model in {null, ""} AND tokens.total == 0 AND artifact_found == false AND kernel_path == ""`. Already exists as `_classify_validity == "infra_fail"`.
  - `F12 timeout after usable artifact`: `result.timeout_after_success == true`. New code path: previously F12 was the "incomplete" bucket; now it has a clean signal.
  - `F11 timeout before usable artifact`: `exit_code == 124 AND artifact_accepted == false`.
  - `F10 no artifact generated`: `artifact_found == false AND F13/F11 did not fire`.
  - `F8 simulator/runtime failure`: `artifact_found == true AND simulator_verify == "fail"`. (Was the existing `F8_correctness`.)
  - `F7 JIT/codegen failure`: `artifact_found == true AND jit_verify == "fail" AND simulator_verify != "fail"`. Currently jit_verify is rarely emitted; lift the heuristic from the existing static-but-JIT-fails path.
  - `F6 static verification failure`: `artifact_found == true AND static_verify == "fail"`. (Was the existing `F7_static`; renumber.)
  - `F5 dtype/precision mismatch`: `simulator_verify == "fail" AND verification.detail mentions "atol" or "rtol" or "dtype"`. Heuristic regex on `verification.detail`; honest about being heuristic.
  - `F4 tiling or shape mismatch`: `simulator_verify == "fail" AND verification.detail mentions "shape" or "TILE_SIZE" or "block_idx"`. Heuristic regex.
  - `F3 invalid @asc2.jit syntax`: `static_verify == "fail" AND verify_kernel.py output mentions "SyntaxError" or "@asc2.jit"`. Read [tests/tools/verify_kernel.py](../../tests/tools/verify_kernel.py) stderr.
  - `F2 wrong pyasc/asc2 API usage`: `static_verify == "pass" AND semantic_check.passed == false AND kernel source contains a misspelled asc2.* call or a v1 `asc.*` call`. Static AST check.
  - `F1 prompt/spec misunderstood`: `static_verify == "pass" AND semantic_check.passed == false AND kernel source does not match the cell's semantic_markers from [tests/tools/semantic_markers.py](../../tests/tools/semantic_markers.py)`. (Today's `F9_semantic` rebrand.)
  - `F9 correct but slow`: `simulator_verify == "pass" AND result.performance_verified == "slow"`. Pinned to null in this sprint (perf layer not shipping).
  - `F0 unknown`: fall-through. Should be empty in practice.

The F1/F2/F5 classifiers are new and may be noisy. Mark them as `heuristic: true` in a sibling field for one quarter while the rules are tuned.

Deliverable: classifier in the collector; `_classify_failure_mode` in [tests/tools/compare_skills_value.py](../../tests/tools/compare_skills_value.py) becomes a back-compat shim that reads `result.failure_category` if present, falls back to the heuristic for v3 files.

## Stage 7.4 — Reader migration (~1.5 ED)

Three consumer files migrate to v4-preferred reads.

- [tests/tools/compare_skills_value.py](../../tests/tools/compare_skills_value.py):
  - Add a one-line schema dispatch at the top of each evidence-reading function:

```python
def _is_v4(ev: dict) -> bool:
    return ev.get("schema_version") == "4"
```
  - Prefer `protocol.id` over `(skills_mode, prompt_variant)` derivation when present.
  - Prefer `result.failure_category` over the heuristic classifier.
  - Continue to read v3 evidence files unchanged (for the historical record + the existing `evidence/*-generative.json` legacy alias).
  - Remove the derived classifier function body but keep its name as a shim for back-compat:

```python
def _classify_failure_mode(ev: dict) -> str | None:
    if _is_v4(ev):
        return ev.get("result", {}).get("failure_category")
    return _classify_failure_mode_v3(ev)
```

- [tests/tools/sync_capabilities.py](../../tests/tools/sync_capabilities.py):
  - Same v4-preferred read for `result.artifact_accepted` (the authoritative pass signal).
  - Cell `generative_status` updates use the v4 signal when present.

- [tests/tools/generate_dashboard.py](../../tests/tools/generate_dashboard.py):
  - Read `protocol.id` for the per-protocol panel (Phase 3).
  - Read `result.failure_category` for the per-cell failure chip.
  - Continue to render v3 cells correctly.

Deliverable: three readers migrated; smoke test [tests/tools/skills_value_smoke.py](../../tests/tools/skills_value_smoke.py) gains a v3 + v4 mixed-fixture run.

## Stage 7.5 — `tests/integration/test_process_as_tests.py` (~1 ED)

Notes §5 says "The evaluation process itself should become testable and reproducible." Add the integration test that enforces it.

```python
# tests/integration/test_process_as_tests.py
def test_every_nightly_evidence_satisfies_process_contract():
    for ev_path in glob("evidence/*-generative-*.json"):
        ev = json.load(open(ev_path))
        if ev.get("schema_version") != "4":
            continue  # back-compat: v3 evidence is exempt
        # Prompt artifact exists.
        assert ev["prompt"], f"{ev_path}: missing prompt"
        # Protocol is recorded.
        assert ev["protocol"]["id"] in {"P2", "P3", "P4", "P6"}, ...
        # Model identity is recorded.
        assert ev["model"]["model_id"], ...
        # Token capture is present.
        assert ev["tokens"]["total"] >= 0, ...
        # Trajectory log exists on disk for the run.
        # (Verified via run_id linkage to generative-archive/<run-id>/agent-output.txt.)
        # Build/test command record exists.
        assert ev["agent"]["timeout_s"] > 0, ...
        # Archived generated code exists.
        assert ev["kernel_path"] or ev["result"]["failure_category"] in {"F10", "F11", "F13"}, ...
        # Failure mode is classified.
        if not ev["result"]["artifact_accepted"]:
            assert ev["result"]["failure_category"], ...
```

Wire into `tests/ci-gate.sh` under the `nightly` tier (run after the matrix completes, before the commit).

Deliverable: test committed; one nightly run executes it; one short subsection in [docs/evaluation-methodology.md](../../docs/evaluation-methodology.md) §"Process as tests".

## Definition of done for Phase 7

- Collector emits `schema_version: "4"` with all documented fields populated.
- `result.failure_category` is set by the collector for every evidence file produced after Phase 7 lands.
- All three readers (`compare_skills_value.py`, `sync_capabilities.py`, `generate_dashboard.py`) prefer v4 fields with v3 fallback.
- `tests/integration/test_process_as_tests.py` is green on a fresh nightly.
- [docs/evaluation-methodology.md](../../docs/evaluation-methodology.md) §"Evidence schema v4 (proposed)" is rewritten as the active contract (not "proposed").
- One quarter of dual v3 + v4 emission committed; v3 retirement is a Phase 8+ follow-up.

## Risks specific to Phase 7

- **Heuristic F-codes (F1, F2, F4, F5) noise.** Pattern-matching on `verification.detail` strings is fragile. Mitigation: emit a sibling `failure_category_confidence: "low"` field when the classification used a heuristic; the dashboard renders these with a dotted underline. After one quarter, re-evaluate.
- **`skill_set_hash` churn.** Every commit to [skills/](../../skills/) changes the hash, which means historical evidence files have different hashes from current runs. Mitigation: the hash is informational, not a comparison key. The aggregator does not require identical hashes for paired comparisons within a single nightly.
- **`timeout_after_success: true` is a new failure mode.** Today's harness conflates "timeout" with "fail". Some current evidence files were `F8_correctness` when they were actually `F12_timeout_after_success`. Phase 7 fixes the forward-going classification but does not re-classify history; the comparison aggregator should annotate v3 evidence with "may include F12 cases" in the data-freshness header.
- **`agent.tools_allowed` may leak secrets.** The opencode.json `permission` dict is benign but neighboring keys may contain API keys. Mitigation: emit only the `permission` keys, not the full opencode.json.
- **`tests/integration/test_process_as_tests.py` over-strictness.** A new assertion that fails on a single back-compat edge case (e.g., an evidence file from a partial nightly per the Phase 0 `partial_run` handling) will block the whole nightly. Mitigation: skip evidence files in `partial_run: true` summaries; document the skip in the test docstring.

## Deferred from Phase 7 (intentionally)

- **v3 retirement.** One quarter of dual emission, then a follow-up sprint removes v3 fields.
- **Performance fields.** `runtime_ms`, `speedup_vs_reference`, `correct_but_slow`. C310 simulator still does not emit reliable perf numbers; reopen when it does.
- **Tightening F1/F2/F4/F5 heuristics.** Q+1 sprint once one quarter of classified evidence is available.
- **Schema v5 / breaking changes.** Not in scope; v4 is the contract for this quarter.
- **Schema validators in pr-gate.** Phase 8 cleanup may add a JSON schema check, but not in Phase 7.
