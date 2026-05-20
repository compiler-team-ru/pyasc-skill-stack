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
