#!/usr/bin/env python3
"""Validate capabilities.yaml (v3, tier-based) against golden kernels and evidence.

For each operation cell in capabilities.yaml, checks both golden_status and
generative_status independently:

  golden_status:
    confirmed   - golden file must exist + pass static verify + golden_evidence JSON valid
    golden_only - golden file must exist + pass static verify
    pending     - golden file should exist (warn if missing)
    claimed     - warn (no artifacts)
    untested    - info only
    blocked     - info only

  generative_status:
    confirmed   - generative_evidence JSON must exist, be valid, have kind=generative + agent section
    pending     - warn (prompt defined but no evidence yet)
    untested    - info only
    blocked     - info only

Structural validation (v3):
  - schema_version must be "3"
  - tiers block must exist with level + description
  - every operation must have a valid tier reference
  - representative_of entries are informational (no validation against asc2 API)

Exit 0 = all consistency checks pass, exit 1 = at least one confirmed cell is broken.

Usage:
    python check_capabilities.py [--json] [--verbose]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[assignment]

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
CAPABILITIES_FILE = REPO_ROOT / "capabilities.yaml"
EVIDENCE_DIR = REPO_ROOT / "evidence"
VERIFY_SCRIPT = SCRIPT_DIR / "verify_kernel.py"
PYTHON = "python3.10"


def _load_yaml(path: Path) -> dict:
    if yaml is not None:
        with open(path) as f:
            return yaml.safe_load(f)
    try:
        result = subprocess.run(
            [PYTHON, "-c", f"import yaml, json; print(json.dumps(yaml.safe_load(open('{path}'))))"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            return json.loads(result.stdout)
    except Exception:
        pass
    sys.stderr.write("ERROR: PyYAML is required. Install with: pip install pyyaml\n")
    sys.exit(2)


def _run_static_verify(kernel_path: Path) -> bool:
    try:
        result = subprocess.run(
            [PYTHON, str(VERIFY_SCRIPT), str(kernel_path), "--json"],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            data = json.loads(result.stdout)
            return data.get("passed", False)
    except Exception:
        pass
    return False


def _validate_evidence(evidence_path: Path, expected_kind: str | None = None) -> tuple[bool, str]:
    if not evidence_path.exists():
        return False, f"evidence file not found: {evidence_path}"
    try:
        with open(evidence_path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        return False, f"evidence file invalid JSON: {exc}"

    required_top = {"schema_version", "operation", "dtype", "date"}
    missing = required_top - set(data.keys())
    if missing:
        return False, f"evidence missing top-level fields: {missing}"

    if "verification" not in data:
        return False, "evidence missing 'verification' section"

    if expected_kind == "generative":
        if "score" not in data:
            return False, "evidence missing 'score' section"
        score_section = data["score"]
        if not isinstance(score_section, dict) or "value" not in score_section:
            return False, "evidence 'score' section missing 'value'"

    if expected_kind:
        actual_kind = data.get("kind", "")
        if actual_kind != expected_kind:
            return False, f"evidence kind mismatch: expected '{expected_kind}', got '{actual_kind}'"

    if expected_kind == "generative" and "agent" not in data:
        return False, "generative evidence missing 'agent' section"

    return True, "OK"


class CellResult:
    def __init__(self, op: str, dtype: str, tier: str):
        self.op = op
        self.dtype = dtype
        self.tier = tier
        self.issues: list[str] = []
        self.warnings: list[str] = []
        self.info: list[str] = []
        self.passed = True

    def fail(self, msg: str) -> None:
        self.issues.append(msg)
        self.passed = False

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    def note(self, msg: str) -> None:
        self.info.append(msg)


def _check_golden(cell: dict, result: CellResult) -> None:
    status = cell.get("golden_status", "untested")

    if status == "confirmed":
        golden = cell.get("golden")
        if not golden:
            result.fail("golden confirmed but no 'golden' path")
        else:
            golden_path = REPO_ROOT / golden
            if not golden_path.exists():
                result.fail(f"golden kernel not found: {golden}")
            elif not _run_static_verify(golden_path):
                result.fail(f"golden kernel fails static verification: {golden}")
            else:
                result.note(f"golden passes: {golden}")

        evidence_ref = cell.get("golden_evidence")
        if not evidence_ref:
            result.fail("golden confirmed but no 'golden_evidence' path")
        else:
            evidence_path = REPO_ROOT / evidence_ref
            ok, detail = _validate_evidence(evidence_path, expected_kind="golden")
            if not ok:
                result.fail(f"golden evidence: {detail}")
            else:
                result.note(f"golden evidence valid: {evidence_ref}")

    elif status == "golden_only":
        golden = cell.get("golden")
        if not golden:
            result.fail("golden_only but no 'golden' path")
        else:
            golden_path = REPO_ROOT / golden
            if not golden_path.exists():
                result.fail(f"golden kernel not found: {golden}")
            elif not _run_static_verify(golden_path):
                result.fail(f"golden kernel fails static verification: {golden}")
            else:
                result.note(f"golden passes: {golden}")

    elif status == "pending":
        golden = cell.get("golden")
        if golden:
            golden_path = REPO_ROOT / golden
            if not golden_path.exists():
                result.warn(f"golden pending but kernel not found: {golden}")
            else:
                result.note(f"golden pending, kernel exists: {golden}")
        else:
            result.warn("golden pending — no golden kernel path")

    elif status == "claimed":
        result.warn("golden claimed — no golden kernel")

    elif status == "untested":
        result.note("golden untested")

    elif status == "blocked":
        notes = cell.get("notes", "no notes")
        result.warn(f"golden blocked: {notes}")


# =============================================================================
# Phase 1: capability-cell metadata enforcement.
#
# The metadata fields are additive on schema_version "3" (see
# docs/glossary.md §6 and docs/evaluation-methodology.md "Capability cell
# metadata schema (Phase 1)"). They are reporting-only — no prompt rewrite,
# no kernel behavior change. The checker enforces:
#   (a) presence of every required field on every cell,
#   (b) enum membership for every enum field,
#   (c) cross-field consistency between (reduce_axis, accumulator_dtype),
#       (tail_behavior, partitioning), and tier ↔ reduce_axis,
#   (d) typo-guarding on unsupported_regimes slugs.
#
# Two-phase rollout via --strict-metadata. Stage 1.2 of the plan has
# already populated every cell, so the default is strict (hard-fail).
# Pass --no-strict-metadata to demote drift to warnings during partial
# pushes.
# =============================================================================

_METADATA_FIELDS: tuple[str, ...] = (
    "shape_regime",
    "reduce_axis",
    "output_shape",
    "accumulator_dtype",
    "identity",
    "tail_behavior",
    "padding",
    "partitioning",
    "unsupported_regimes",
)

_ALLOWED_SHAPE_REGIME: set[str] = {"fixed", "runtime_size_only", "dynamic"}
_ALLOWED_TAIL_BEHAVIOR: set[str] = {
    "aligned_only",
    "host_pad",
    "mask",
    "real_shape",
    "host_dispatcher",
    "unsupported",
}
_ALLOWED_PARTITIONING: set[str] = {
    "row_per_core",
    "tile_per_core",
    "block_grid",
    "host_dispatcher",
}
_ALLOWED_ACCUMULATOR_DTYPE: set[str | None] = {"float16", "float32", None}
_ALLOWED_IDENTITY: set[str | None] = {"0", "1", "-inf", "+inf", None}

# Tiers whose ops are never reducing (cells must have reduce_axis=null +
# accumulator_dtype=null). ``composed`` is excluded because it could in
# principle host a composed reduction; today its cells are non-reducing
# but the checker treats this as a per-cell invariant rather than a
# per-tier one.
_NON_REDUCING_TIERS: set[str] = {"elementwise"}
_REDUCING_TIERS: set[str] = {"reduction"}

# Standard ``unsupported_regimes`` slugs documented in docs/glossary.md §6.
# Cells that need a new slug must add it here at the same PR so the gate
# remains a typo guard.
_KNOWN_UNSUPPORTED_REGIMES: set[str] = {
    "abl1_full",
    "al1_full",
    "bl1_full",
    "dynamic_num_cols",
    "dynamic_num_cols_not_8_aligned_in_full_row",
    "dynamic_num_rows",
    "k_tiled",
    "long_rows_exceeding_UB",
    "multi_axis_reduction",
    "non_16_multiple_shapes",
    "non_last_axis",
    "num_cols_below_split_d_tile_threshold",
    "split_row",
}


def _check_cell_metadata(
    cell: dict, op_name: str, tier: str, result: CellResult,
    *, strict: bool,
) -> None:
    """Validate Phase 1 metadata fields on one capability cell.

    With ``strict=True`` (the default and the post-Stage-1.2 contract),
    every issue is a hard fail. With ``strict=False`` everything is
    demoted to a warning — useful for partial pushes during the
    Stage 1.2 rollout cycle.
    """
    fail = result.fail if strict else result.warn
    record = lambda msg: fail(f"metadata: {msg}")

    missing = [field for field in _METADATA_FIELDS if field not in cell]
    if missing:
        record(f"missing required fields {missing}")
        # The remaining checks read fields that may be missing — bail out.
        return

    shape_regime = cell.get("shape_regime")
    if shape_regime not in _ALLOWED_SHAPE_REGIME:
        record(
            f"shape_regime={shape_regime!r} not in {sorted(_ALLOWED_SHAPE_REGIME)}"
        )

    tail_behavior = cell.get("tail_behavior")
    if tail_behavior not in _ALLOWED_TAIL_BEHAVIOR:
        record(
            f"tail_behavior={tail_behavior!r} not in {sorted(_ALLOWED_TAIL_BEHAVIOR)}"
        )

    partitioning = cell.get("partitioning")
    if partitioning not in _ALLOWED_PARTITIONING:
        record(
            f"partitioning={partitioning!r} not in {sorted(_ALLOWED_PARTITIONING)}"
        )

    accumulator_dtype = cell.get("accumulator_dtype")
    if accumulator_dtype not in _ALLOWED_ACCUMULATOR_DTYPE:
        record(
            f"accumulator_dtype={accumulator_dtype!r} not in {sorted(str(x) for x in _ALLOWED_ACCUMULATOR_DTYPE)}"
        )

    identity = cell.get("identity")
    if identity not in _ALLOWED_IDENTITY:
        record(
            f"identity={identity!r} not in {sorted(str(x) for x in _ALLOWED_IDENTITY)} "
            f"(store as string, not bare float / int)"
        )

    reduce_axis = cell.get("reduce_axis")
    if reduce_axis is not None and not isinstance(reduce_axis, int):
        record(
            f"reduce_axis={reduce_axis!r} must be an int or null"
        )

    output_shape = cell.get("output_shape")
    if not (output_shape == "same_as_input" or isinstance(output_shape, list)):
        record(
            f"output_shape={output_shape!r} must be 'same_as_input' or a list"
        )

    padding = cell.get("padding")
    if padding is not None and not isinstance(padding, int):
        record(f"padding={padding!r} must be an int (element count) or null")

    unsupported_regimes = cell.get("unsupported_regimes")
    if not isinstance(unsupported_regimes, list):
        record(
            f"unsupported_regimes={unsupported_regimes!r} must be a list of slugs"
        )
        unsupported_regimes = []
    unknown_slugs = [
        slug for slug in unsupported_regimes
        if slug not in _KNOWN_UNSUPPORTED_REGIMES
    ]
    if unknown_slugs:
        record(
            f"unsupported_regimes contains unknown slug(s) {unknown_slugs}; "
            "add to docs/glossary.md §6 and _KNOWN_UNSUPPORTED_REGIMES in "
            "check_capabilities.py if intentional"
        )

    if (reduce_axis is None) != (accumulator_dtype is None):
        record(
            f"reduce_axis ({reduce_axis!r}) and accumulator_dtype "
            f"({accumulator_dtype!r}) must both be null or both be set "
            "(a reducing op has both; a non-reducing op has neither)"
        )

    if tier in _NON_REDUCING_TIERS and reduce_axis is not None:
        record(
            f"tier={tier!r} is non-reducing but reduce_axis={reduce_axis!r}; "
            f"set reduce_axis=null for {op_name}"
        )
    if tier in _REDUCING_TIERS and reduce_axis is None:
        record(
            f"tier={tier!r} is reducing but reduce_axis is null; "
            f"set reduce_axis=-1 (last axis) for {op_name}"
        )

    if (tail_behavior == "host_dispatcher") != (partitioning == "host_dispatcher"):
        record(
            "tail_behavior=host_dispatcher implies partitioning=host_dispatcher "
            f"(got tail_behavior={tail_behavior!r}, partitioning={partitioning!r})"
        )


# =============================================================================
# Phase 2 Stage 2.3: examples_policy enforcement.
#
# Every cell declares an ``examples_policy`` mapping with six boolean
# slots. The mapping is the per-cell policy the harness must respect at
# P2/P3/P4/P6 (the ``allowed_context`` block in the evidence file).
# Cells with an ``oracle_guided`` prompt variant may override individual
# slots inside ``prompt_variants.oracle_guided.examples_policy`` — only
# the oracle variant sees the override; the cell-level defaults still
# apply to ``minimal`` / ``guided``.
# =============================================================================

_EXAMPLES_POLICY_KEYS: set[str] = {
    "task_prompt",
    "glossary",
    "golden_kernels",
    "golden_docs",
    "external_web",
    "human_hints",
}

# Slots that are always true (the harness cannot opt the agent out of
# reading its own task prompt or the canonical glossary).
_EXAMPLES_POLICY_FORCED_TRUE: set[str] = {"task_prompt", "glossary"}


def _check_examples_policy_dict(
    policy: object, label: str, result: CellResult, *, strict: bool,
) -> None:
    """Validate a single examples_policy mapping (cell-level or per-variant)."""
    fail = result.fail if strict else result.warn
    if not isinstance(policy, dict):
        fail(
            f"{label} must be a mapping with keys {sorted(_EXAMPLES_POLICY_KEYS)}; "
            f"got {type(policy).__name__}"
        )
        return
    keys = set(policy.keys())
    missing = _EXAMPLES_POLICY_KEYS - keys
    extra = keys - _EXAMPLES_POLICY_KEYS
    if missing:
        fail(f"{label} missing required keys {sorted(missing)}")
    if extra:
        fail(
            f"{label} contains unknown keys {sorted(extra)}; "
            f"allowed: {sorted(_EXAMPLES_POLICY_KEYS)}"
        )
    for k in _EXAMPLES_POLICY_KEYS & keys:
        if not isinstance(policy[k], bool):
            fail(
                f"{label}.{k} must be a bool, got "
                f"{type(policy[k]).__name__!s}={policy[k]!r}"
            )
    for k in _EXAMPLES_POLICY_FORCED_TRUE & keys:
        if policy.get(k) is False:
            fail(
                f"{label}.{k} must be true (task_prompt and glossary are "
                "always allowed; see docs/prompt-template.md and "
                "docs/evaluation-methodology.md \"Baseline validity\")"
            )


def _check_examples_policy(
    cell: dict, result: CellResult, *, strict: bool,
) -> None:
    """Validate that the cell declares an examples_policy mapping."""
    if "examples_policy" not in cell:
        (result.fail if strict else result.warn)(
            "missing examples_policy (Phase 2 Stage 2.3; see "
            "docs/prompt-template.md \"Cross-references\")"
        )
        return
    _check_examples_policy_dict(
        cell["examples_policy"], "examples_policy", result, strict=strict,
    )


def _check_prompt_variants(cell: dict, result: CellResult) -> None:
    """Enforce Phase 0 protocol-axis requirement: every cell must define
    both ``prompt_variants.minimal`` and ``prompt_variants.guided`` so
    the collector can drive P2 (minimal+skills_off) and P3/P4/P6 (guided)
    legs without falling back to the legacy primary ``prompt``.

    ``prompt_variants.oracle_guided`` (Phase 2 Stage 2.3) may be either a
    bare prompt string OR a mapping with ``prompt`` (string) and an
    optional per-variant ``examples_policy`` override. The collector
    normalizes both shapes to a string before sending to the agent (see
    ``tests/tools/collect_generative_evidence.py``::``_variant_prompt``).
    """
    variants = cell.get("prompt_variants") or {}
    if not isinstance(variants, dict):
        result.fail("prompt_variants must be a mapping when present")
        return
    minimal = variants.get("minimal")
    guided = variants.get("guided")
    if not (isinstance(minimal, str) and minimal.strip()):
        result.fail(
            "missing prompt_variants.minimal "
            "(required for P2 leg; see docs/evaluation-methodology.md "
            "\"Protocol-axis CI mapping (Phase 0)\")"
        )
    if not (isinstance(guided, str) and guided.strip()):
        result.fail(
            "missing prompt_variants.guided "
            "(required for P3/P4/P6 legs; see docs/evaluation-methodology.md "
            "\"Protocol-axis CI mapping (Phase 0)\")"
        )

    og = variants.get("oracle_guided")
    if og is None:
        return
    if isinstance(og, str):
        if not og.strip():
            result.fail(
                "prompt_variants.oracle_guided is an empty string; either "
                "drop the variant or populate it (see docs/prompt-template.md)"
            )
        return
    if not isinstance(og, dict):
        result.fail(
            "prompt_variants.oracle_guided must be a string or a mapping "
            "(see docs/prompt-template.md \"oracle_guided\")"
        )
        return
    og_prompt = og.get("prompt")
    if not (isinstance(og_prompt, str) and og_prompt.strip()):
        result.fail(
            "prompt_variants.oracle_guided.prompt missing or empty (dict form)"
        )
    og_policy = og.get("examples_policy")
    if og_policy is not None:
        _check_examples_policy_dict(
            og_policy,
            "prompt_variants.oracle_guided.examples_policy",
            result,
            strict=True,
        )
    og_extra = set(og.keys()) - {"prompt", "examples_policy"}
    if og_extra:
        result.fail(
            f"prompt_variants.oracle_guided has unknown keys "
            f"{sorted(og_extra)}; allowed: prompt, examples_policy"
        )


def _check_generative(cell: dict, result: CellResult, soft_runtime: bool = False) -> None:
    status = cell.get("generative_status", "untested")

    if status == "confirmed":
        evidence_ref = cell.get("generative_evidence")
        if not evidence_ref:
            result.fail("generative confirmed but no 'generative_evidence' path")
        else:
            evidence_path = REPO_ROOT / evidence_ref
            ok, detail = _validate_evidence(evidence_path, expected_kind="generative")
            if not ok:
                result.fail(f"generative evidence: {detail}")
            else:
                with open(evidence_path) as f:
                    ev_data = json.load(f)
                rt_status = ev_data.get("verification", {}).get("status", "")
                if rt_status != "pass":
                    msg = (f"generative confirmed but runtime status is '{rt_status}',"
                           f" not 'pass' (drift — likely the nightly-bot has updated"
                           f" since you last pulled; run"
                           f" `python3 tests/tools/sync_capabilities.py` and commit,"
                           f" or pull and rebase)")
                    if soft_runtime:
                        result.warn(msg)
                    else:
                        result.fail(msg)
                else:
                    result.note(f"generative evidence valid + runtime pass: {evidence_ref}")

    elif status == "pending":
        prompt = cell.get("prompt")
        if prompt:
            result.warn("generative pending — prompt defined, no evidence yet")
        else:
            result.warn("generative pending — no prompt defined")

    elif status == "untested":
        result.note("generative untested")

    elif status == "blocked":
        notes = cell.get("notes", "no notes")
        result.warn(f"generative blocked: {notes}")


def check_structure(data: dict) -> list[str]:
    """Validate v3 structural requirements. Returns list of fatal errors."""
    errors: list[str] = []

    tiers = data.get("tiers")
    if not tiers or not isinstance(tiers, dict):
        errors.append("missing or invalid 'tiers' block")
        return errors

    for t_name, t_info in tiers.items():
        if "level" not in t_info:
            errors.append(f"tier '{t_name}' missing 'level'")
        if "description" not in t_info:
            errors.append(f"tier '{t_name}' missing 'description'")

    valid_tiers = set(tiers.keys())
    for op in data.get("operations", []):
        op_name = op.get("name", "?")
        op_tier = op.get("tier")
        if not op_tier:
            errors.append(f"operation '{op_name}' missing 'tier' field")
        elif op_tier not in valid_tiers:
            errors.append(f"operation '{op_name}' references unknown tier '{op_tier}'")

    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate capabilities.yaml (v3) consistency.")
    parser.add_argument("--json", action="store_true", help="JSON output")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show info-level notes")
    parser.add_argument(
        "--soft-runtime",
        action="store_true",
        help=(
            "Demote 'generative confirmed but runtime fail' from FAIL to WARN."
            " Used at PR-gate time so that transient nightly-vs-local drift in"
            " generative_status does not block unrelated commits. Hard failures"
            " (missing files, invalid JSON, schema violations, broken golden"
            " static verify) still fail. Merge-gate and nightly-gate run the"
            " strict variant."
        ),
    )
    parser.add_argument(
        "--strict-metadata",
        dest="strict_metadata",
        action="store_true",
        default=True,
        help=(
            "Hard-fail on missing or invalid Phase 1 capability-cell metadata"
            " fields (shape_regime, reduce_axis, output_shape,"
            " accumulator_dtype, identity, tail_behavior, padding,"
            " partitioning, unsupported_regimes). This is the default after"
            " Stage 1.2 lands. See docs/glossary.md §6 for the allowed values."
        ),
    )
    parser.add_argument(
        "--no-strict-metadata",
        dest="strict_metadata",
        action="store_false",
        help=(
            "Demote Phase 1 metadata violations from FAIL to WARN. Use while"
            " a partial-rollout branch is in flight."
        ),
    )
    args = parser.parse_args()

    if not CAPABILITIES_FILE.exists():
        if args.json:
            print(json.dumps({"status": "fail", "error": "capabilities.yaml not found"}))
        else:
            print(f"FAIL: {CAPABILITIES_FILE} not found")
        sys.exit(1)

    data = _load_yaml(CAPABILITIES_FILE)

    schema = data.get("schema_version", "1")
    if schema != "3":
        msg = f"capabilities.yaml schema_version is '{schema}', expected '3'"
        if args.json:
            print(json.dumps({"status": "fail", "error": msg}))
        else:
            print(f"FAIL: {msg}")
        sys.exit(1)

    struct_errors = check_structure(data)
    if struct_errors:
        if args.json:
            print(json.dumps({"status": "fail", "structural_errors": struct_errors}))
        else:
            print("FAIL: Structural validation errors:")
            for e in struct_errors:
                print(f"  - {e}")
        sys.exit(1)

    operations = data.get("operations", [])
    tiers = data.get("tiers", {})

    results: list[CellResult] = []
    for op in operations:
        op_name = op.get("name", "unknown")
        op_tier = op.get("tier", "unknown")
        for cell in op.get("cells", []):
            dtype = cell.get("dtype", "unknown")
            result = CellResult(op_name, dtype, op_tier)
            _check_prompt_variants(cell, result)
            _check_cell_metadata(
                cell, op_name, op_tier, result,
                strict=args.strict_metadata,
            )
            _check_examples_policy(
                cell, result, strict=args.strict_metadata,
            )
            _check_golden(cell, result)
            _check_generative(cell, result, soft_runtime=args.soft_runtime)
            results.append(result)

    failures = [r for r in results if not r.passed]
    warnings = [r for r in results if r.warnings]

    tier_counts: dict[str, dict] = {}
    for r in results:
        if r.tier not in tier_counts:
            tier_counts[r.tier] = {"total": 0, "golden_confirmed": 0, "gen_confirmed": 0}
        tc = tier_counts[r.tier]
        tc["total"] += 1
        for op in operations:
            if op.get("name") != r.op:
                continue
            for cell in op.get("cells", []):
                if cell.get("dtype") == r.dtype:
                    if cell.get("golden_status") == "confirmed":
                        tc["golden_confirmed"] += 1
                    if cell.get("generative_status") == "confirmed":
                        tc["gen_confirmed"] += 1
                    break
            break

    if args.json:
        out = {
            "status": "pass" if not failures else "fail",
            "schema_version": "3",
            "total_cells": len(results),
            "tier_counts": tier_counts,
            "failures": [
                {"op": r.op, "dtype": r.dtype, "tier": r.tier, "issues": r.issues}
                for r in failures
            ],
            "warnings": [
                {"op": r.op, "dtype": r.dtype, "tier": r.tier, "warnings": r.warnings}
                for r in warnings
            ],
        }
        print(json.dumps(out, indent=2))
    else:
        print("=" * 60)
        print("  Capabilities Matrix Validation (v3 — tier-based)")
        print("=" * 60)
        print()

        for r in results:
            tag = "PASS" if r.passed else "FAIL"
            line = f"  [{tag}] {r.op}/{r.dtype} (tier: {r.tier})"
            if r.issues:
                line += f" — {'; '.join(r.issues)}"
            if r.warnings:
                line += f" — {'; '.join(r.warnings)}"
            if args.verbose and r.info:
                line += f" — {'; '.join(r.info)}"
            print(line)

        print()
        print(f"  Cells: {len(results)} total")
        for t_name in sorted(tier_counts, key=lambda k: tiers.get(k, {}).get("level", 99)):
            tc = tier_counts[t_name]
            lvl = tiers.get(t_name, {}).get("level", "?")
            print(f"  Tier {lvl} ({t_name}): golden {tc['golden_confirmed']}/{tc['total']}, "
                  f"gen {tc['gen_confirmed']}/{tc['total']}")
        print()

        if failures:
            print(f"  FAIL: {len(failures)} cell(s) have broken artifacts")
        else:
            print("  PASS: All confirmed cells are consistent")

    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
