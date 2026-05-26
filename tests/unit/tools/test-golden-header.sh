#!/usr/bin/env bash
# =============================================================================
# L1 Unit Test: Phase 1 golden-kernel header ↔ capabilities.yaml drift
#
# For every cell in capabilities.yaml that has a confirmed (or golden_only)
# golden, the corresponding golden/kernels/*.py file MUST have a top-level
# docstring that names the same shape_regime / tail_behavior / partitioning
# values. This is a typo-guard: a metadata edit without a matching golden
# header edit (or vice versa) is rejected at the PR gate.
#
# Cheap to write, hard to defeat — implemented as a Python helper that
# walks the YAML once, opens each golden, and greps for the three strings.
#
# See: .cursor/plans/phase_1_spec_hygiene.plan.md "Risks" §"Drift between
#      cell metadata and golden kernel header"
#      docs/glossary.md §6 "Cell metadata enums"
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../lib/test-helpers.sh"

echo "=== Test: Phase 1 golden-header ↔ capabilities.yaml drift ==="

if command -v python3 >/dev/null 2>&1; then
    PY=python3
else
    PY="$PYTHON"
fi

"$PY" - <<'PYEOF'
import sys
from pathlib import Path

candidate = Path.cwd()
while candidate != candidate.parent:
    if (candidate / "capabilities.yaml").exists():
        break
    candidate = candidate.parent
else:
    print("FAIL: cannot find repo root (no capabilities.yaml)", file=sys.stderr)
    raise SystemExit(1)

import yaml
data = yaml.safe_load((candidate / "capabilities.yaml").read_text())

errors: list[str] = []
checks = 0

for op in data.get("operations", []):
    op_name = op.get("name", "?")
    for cell in op.get("cells", []):
        golden_path = cell.get("golden")
        if not golden_path:
            continue
        full = candidate / golden_path
        if not full.exists():
            errors.append(f"{op_name}/{cell.get('dtype')}: golden file missing {golden_path}")
            continue
        text = full.read_text()
        try:
            module_doc_end = text.index('"""', text.index('"""') + 3)
            header = text[: module_doc_end + 3]
        except ValueError:
            errors.append(f"{op_name}/{cell.get('dtype')}: {golden_path} has no top-level docstring")
            continue

        for field in ("shape_regime", "tail_behavior", "partitioning"):
            value = cell.get(field)
            if value is None:
                errors.append(f"{op_name}/{cell.get('dtype')}: cell {field} is missing (Stage 1.2)")
                continue
            needle = f"{field}: {value}"
            if needle not in header:
                errors.append(
                    f"{op_name}/{cell.get('dtype')} ({golden_path}): docstring header is "
                    f"missing '{needle}' — header and capabilities.yaml have drifted"
                )
        checks += 1

if errors:
    print(f"FAIL: {len(errors)} drift issue(s) across {checks} cell(s) with goldens:",
          file=sys.stderr)
    for e in errors:
        print(f"  - {e}", file=sys.stderr)
    raise SystemExit(1)

print(f"  [OK] {checks} cell(s) with goldens; "
      f"shape_regime/tail_behavior/partitioning match")
print("ALL GOLDEN-HEADER CHECKS PASS")
PYEOF

if [ "$?" -eq 0 ]; then
    print_pass "Phase 1 golden-kernel headers match capabilities.yaml"
else
    print_fail "Phase 1 golden-kernel headers drift from capabilities.yaml"
    exit 1
fi
