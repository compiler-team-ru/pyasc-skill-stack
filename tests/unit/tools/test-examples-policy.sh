#!/usr/bin/env bash
# =============================================================================
# L1 Unit Test: Phase 2 Stage 2.3 — examples_policy + dict-form oracle_guided.
#
# Validates two things:
#   1. Every cell in capabilities.yaml declares an examples_policy mapping
#      with the canonical six keys, task_prompt=glossary=true.
#   2. The four oracle-carrying cells store oracle_guided as a dict with a
#      ``prompt`` string and (optionally) a per-variant examples_policy
#      override; check_capabilities.py accepts both string and dict forms;
#      collect_generative_evidence.load_prompt_from_capabilities()
#      normalizes the dict form to the inner ``prompt`` string.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../lib/test-helpers.sh"

echo "=== Test: examples_policy + dict-form oracle_guided ==="

CAPS_FILE="$SKILLS_DIR/capabilities.yaml"
CHECK_TOOL="$TOOLS_DIR/check_capabilities.py"
COLLECT_TOOL="$TOOLS_DIR/collect_generative_evidence.py"

[ -f "$CAPS_FILE" ] || { print_fail "capabilities.yaml not found at $CAPS_FILE"; exit 1; }
[ -f "$CHECK_TOOL" ] || { print_fail "check_capabilities.py not found"; exit 1; }
[ -f "$COLLECT_TOOL" ] || { print_fail "collect_generative_evidence.py not found"; exit 1; }

# -----------------------------------------------------------------------------
# Case 1: every cell has the canonical examples_policy.
# -----------------------------------------------------------------------------
$PYTHON - <<'PY'
import sys, yaml
d = yaml.safe_load(open("capabilities.yaml"))
expected = {"task_prompt", "glossary", "golden_kernels", "golden_docs",
            "external_web", "human_hints"}
errors = []
for op in d["operations"]:
    for cell in op["cells"]:
        ep = cell.get("examples_policy")
        cid = f"{op['name']}/{cell['dtype']}"
        if not isinstance(ep, dict):
            errors.append(f"{cid}: examples_policy missing or not dict")
            continue
        keys = set(ep.keys())
        missing = expected - keys
        extra = keys - expected
        if missing:
            errors.append(f"{cid}: missing {sorted(missing)}")
        if extra:
            errors.append(f"{cid}: extra {sorted(extra)}")
        if ep.get("task_prompt") is not True:
            errors.append(f"{cid}: task_prompt must be true")
        if ep.get("glossary") is not True:
            errors.append(f"{cid}: glossary must be true")
        for k in expected & keys:
            if not isinstance(ep[k], bool):
                errors.append(f"{cid}: {k} is not bool")
if errors:
    print("FAIL: cell-level examples_policy violations:")
    for e in errors:
        print(f"  {e}")
    sys.exit(1)
print("  [OK] 12 cell(s) declare canonical examples_policy")
PY

# -----------------------------------------------------------------------------
# Case 2: four cells carry dict-form oracle_guided with prompt + examples_policy.
# -----------------------------------------------------------------------------
$PYTHON - <<'PY'
import sys, yaml
d = yaml.safe_load(open("capabilities.yaml"))
expected_cells = {("gelu", "float32"), ("matmul", "float16"),
                  ("rms_norm", "float16"), ("rms_norm", "float32")}
seen = set()
errors = []
for op in d["operations"]:
    for cell in op["cells"]:
        key = (op["name"], cell["dtype"])
        og = (cell.get("prompt_variants") or {}).get("oracle_guided")
        if og is None:
            continue
        if not isinstance(og, dict):
            errors.append(f"{op['name']}/{cell['dtype']}: oracle_guided not a mapping (got {type(og).__name__})")
            continue
        if not isinstance(og.get("prompt"), str) or not og["prompt"].strip():
            errors.append(f"{op['name']}/{cell['dtype']}: oracle_guided.prompt missing or empty")
        ep = og.get("examples_policy")
        if ep is not None and not isinstance(ep, dict):
            errors.append(f"{op['name']}/{cell['dtype']}: oracle_guided.examples_policy not a mapping")
        seen.add(key)
missing_cells = expected_cells - seen
if missing_cells:
    errors.append(f"missing oracle_guided on cells: {sorted(missing_cells)}")
if errors:
    print("FAIL: oracle_guided dict-form violations:")
    for e in errors:
        print(f"  {e}")
    sys.exit(1)
print(f"  [OK] {len(seen)} oracle_guided variant(s) in dict form with policy")
PY

# -----------------------------------------------------------------------------
# Case 3: check_capabilities.py reports pass with strict-metadata default-on.
# -----------------------------------------------------------------------------
status=$($PYTHON "$CHECK_TOOL" --json 2>/dev/null | $PYTHON -c "import json,sys; print(json.load(sys.stdin)['status'])")
if [ "$status" = "pass" ]; then
    echo "  [OK] check_capabilities.py reports pass (strict-metadata + examples_policy + oracle_guided)"
else
    print_fail "check_capabilities.py reports $status (expected pass)"
    exit 1
fi

# -----------------------------------------------------------------------------
# Case 4: collect_generative_evidence.load_prompt_from_capabilities() returns a
# bare string for an oracle_guided variant stored as dict.
# -----------------------------------------------------------------------------
$PYTHON - <<'PY'
import sys, importlib.util
from pathlib import Path
# collect_generative_evidence imports sibling helpers from tests/tools/, so
# put that directory on sys.path before loading it.
sys.path.insert(0, str(Path("tests/tools").resolve()))
spec = importlib.util.spec_from_file_location(
    "collect_ge", "tests/tools/collect_generative_evidence.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
for op, dtype in [("matmul", "float16"), ("rms_norm", "float32")]:
    p = mod.load_prompt_from_capabilities(op, dtype, variant="oracle_guided")
    if not isinstance(p, str):
        print(f"FAIL: {op}/{dtype} oracle_guided returned {type(p).__name__}, not str")
        sys.exit(1)
    if not p.strip():
        print(f"FAIL: {op}/{dtype} oracle_guided returned empty string")
        sys.exit(1)
print("  [OK] load_prompt_from_capabilities normalizes dict-form oracle_guided to str")
PY

echo ""
print_pass "examples_policy + dict-form oracle_guided checks passed"
