#!/usr/bin/env bash
# =============================================================================
# L1 Unit Test: skills-value aggregator + dashboard renderer smoke
# Runs tests/tools/skills_value_smoke.py — no external deps beyond Python +
# the workspace itself. Exercises:
#   * _classify_validity (all branches)
#   * _classify_failure_mode (F0/F7/F8/F9/F10/F12/F13 + pass)
#   * _attempts_to_pass (every branch)
#   * _wilson_ci bounds
#   * aggregator end-to-end (no clean baseline + clean baseline + unlock)
#   * --partial-run / --legs-status-file plumbing
#   * merge_evidence_artifacts.sh stale-overwrite race protection
#   * detect_partial_run.py both branches
#   * generate_dashboard.py inlines the new explainability + efficiency keys
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../lib/test-helpers.sh"

echo "=== Test: skills-value aggregator + dashboard smoke ==="

SMOKE="$TOOLS_DIR/skills_value_smoke.py"
if [ ! -f "$SMOKE" ]; then
    print_fail "skills_value_smoke.py not found at $SMOKE"
    exit 1
fi

# Use python3 (not $PYTHON which defaults to python3.10) so the smoke
# runs on whatever the runner has — the script only needs stdlib.
if command -v python3 >/dev/null 2>&1; then
    PY=python3
else
    PY="$PYTHON"
fi

if "$PY" "$SMOKE"; then
    print_pass "skills-value smoke (10 cases) passed"
else
    print_fail "skills-value smoke failed (see output above)"
    exit 1
fi

# Schema v4 gate: every current-schema generative evidence file under
# evidence/ must carry a non-empty pyasc_revision.sha. Legacy quarantined
# files under evidence/legacy-* and the on-disk runtime-archive are
# excluded (they were emitted before the field existed).
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
EVIDENCE_DIR="$REPO_ROOT/evidence"
if [ -d "$EVIDENCE_DIR" ]; then
    if ! "$PY" - "$EVIDENCE_DIR" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
EXCLUDE_PARTS = {"legacy-static-only", "legacy-cann-mirror-wip",
                 "runtime-archive"}
missing = []
for path in sorted(root.rglob("*.json")):
    if set(path.relative_to(root).parts) & EXCLUDE_PARTS:
        continue
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        continue
    if data.get("kind") != "generative":
        continue
    if str(data.get("schema_version", "")) < "4":
        continue
    rev = data.get("pyasc_revision") or {}
    if not rev.get("sha"):
        missing.append(str(path.relative_to(root)))

if missing:
    print("FAIL: schema_version>=4 evidence missing pyasc_revision.sha:",
          file=sys.stderr)
    for p in missing:
        print(f"  {p}", file=sys.stderr)
    sys.exit(1)
PY
    then
        print_fail "pyasc_revision gate (schema v4) failed"
        exit 1
    fi
    print_pass "pyasc_revision gate: every schema-v4 evidence file carries an SHA"
fi
