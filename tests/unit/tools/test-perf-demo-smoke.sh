#!/usr/bin/env bash
# =============================================================================
# L1 Unit Test: perf-vs-AscendC demo harness smoke
# Runs tests/tools/perf_demo_smoke.py — camodel-free, stdlib-only. Validates
# the repo-aware reference runner (OP_SPECS), the generated-side runner
# (arg-spec builders + launch selection), and the demo orchestrator (CELLS)
# are structurally consistent and that all 5 requested operators are wired in.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../lib/test-helpers.sh"

echo "=== Test: perf-vs-AscendC demo harness smoke ==="

SMOKE="$TOOLS_DIR/perf_demo_smoke.py"
if [ ! -f "$SMOKE" ]; then
    print_fail "perf_demo_smoke.py not found at $SMOKE"
    exit 1
fi

if command -v python3 >/dev/null 2>&1; then
    PY=python3
else
    PY="$PYTHON"
fi

if "$PY" "$SMOKE"; then
    print_pass "perf demo harness smoke passed"
else
    print_fail "perf demo harness smoke failed (see output above)"
    exit 1
fi
