#!/usr/bin/env bash
# =============================================================================
# L1 Unit Test: Phase 0 protocol-axis derivation table
# Asserts that collect_generative_evidence.PROTOCOL_TABLE matches the
# canonical mapping documented in docs/evaluation-methodology.md
# §"Protocol-axis CI mapping (Phase 0)". The collector reads from this
# table for --protocol-id resolution, so a drift between code and docs
# would silently mismeasure a CI leg.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../lib/test-helpers.sh"

echo "=== Test: Phase 0 protocol-axis derivation table ==="

if command -v python3 >/dev/null 2>&1; then
    PY=python3
else
    PY="$PYTHON"
fi

"$PY" - <<'PYEOF'
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent if False else Path.cwd()
# Locate the repo root by walking up to the first directory that contains
# tests/tools/collect_generative_evidence.py.
here = Path(__file__).resolve() if False else Path.cwd()
# We are executed with cwd at the repo root by run-tests.sh / the shell
# wrapper, but be defensive: walk up from this file's directory if not.
candidate = Path.cwd()
while candidate != candidate.parent:
    if (candidate / "tests" / "tools" / "collect_generative_evidence.py").exists():
        break
    candidate = candidate.parent
else:
    print("FAIL: cannot find repo root from cwd", file=sys.stderr)
    raise SystemExit(1)

sys.path.insert(0, str(candidate / "tests" / "tools"))
import collect_generative_evidence as cge

EXPECTED = {
    "P2": {
        "name": "opencode-skills-off-minimal",
        "skills_mode": "off",
        "prompt_variant": "minimal",
        "agents_md": False,
    },
    "P3": {
        "name": "opencode-skills-off-guided",
        "skills_mode": "off",
        "prompt_variant": "guided",
        "agents_md": False,
    },
    "P4": {
        "name": "opencode-skills-off-agents-md",
        "skills_mode": "off",
        "prompt_variant": "guided",
        "agents_md": True,
    },
    "P6": {
        "name": "opencode-skills-on-guided",
        "skills_mode": "on",
        "prompt_variant": "guided",
        "agents_md": False,
    },
}

assert set(cge.PROTOCOL_TABLE.keys()) == set(EXPECTED.keys()), (
    f"PROTOCOL_TABLE keys mismatch: expected {sorted(EXPECTED)}, "
    f"got {sorted(cge.PROTOCOL_TABLE)}"
)

for pid, want in EXPECTED.items():
    got = cge.PROTOCOL_TABLE[pid]
    for key, value in want.items():
        assert got.get(key) == value, (
            f"{pid}.{key}: expected {value!r}, got {got.get(key)!r}"
        )
    derived = cge.derive_protocol(pid)
    for key, value in want.items():
        assert derived.get(key) == value, (
            f"derive_protocol({pid!r}).{key}: expected {value!r}, "
            f"got {derived.get(key)!r}"
        )
    print(f"  [OK] {pid}: skills_mode={want['skills_mode']}, "
          f"prompt_variant={want['prompt_variant']}, "
          f"agents_md={want['agents_md']}")

try:
    cge.derive_protocol("P0")
except KeyError:
    print("  [OK] derive_protocol rejects unknown id")
else:
    print("FAIL: derive_protocol should raise KeyError on unknown id",
          file=sys.stderr)
    raise SystemExit(1)

print("ALL DERIVATION CASES PASS")
PYEOF

if [ "$?" -eq 0 ]; then
    print_pass "Phase 0 protocol-derivation table matches the doc mapping"
else
    print_fail "Phase 0 protocol-derivation table drifted from doc mapping"
    exit 1
fi
