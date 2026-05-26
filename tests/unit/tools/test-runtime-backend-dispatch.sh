#!/usr/bin/env bash
# =============================================================================
# L1 Unit Test: --runtime-backend dispatch in collect_generative_evidence.py.
#
# Verifies four things:
#   1. RUNTIME_BACKEND_CHOICES exposes ("auto", "host", "docker").
#   2. resolve_runtime_backend() pass-through behavior on explicit values.
#   3. resolve_runtime_backend("auto") picks "host" or "docker" based on a
#      monkeypatched host_runtime_available() probe (both branches).
#   4. invoke_runtime_verify() dispatches to the right verify function with
#      monkeypatched run_host_verify / run_docker_verify, and refuses
#      "auto" as an input (must be resolved first).
#
# Pure unit test, no opencode / no Docker / no CANN required.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../lib/test-helpers.sh"

echo "=== Test: --runtime-backend dispatch (Stage A.4) ==="

COLLECT_TOOL="$TOOLS_DIR/collect_generative_evidence.py"
[ -f "$COLLECT_TOOL" ] || { print_fail "collect_generative_evidence.py not found"; exit 1; }

$PYTHON - <<'PY'
import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path("tests/tools").resolve()))
spec = importlib.util.spec_from_file_location(
    "collect_ge", "tests/tools/collect_generative_evidence.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

errors = []

# Case 1: RUNTIME_BACKEND_CHOICES surface.
expected_choices = ("auto", "host", "docker")
if mod.RUNTIME_BACKEND_CHOICES != expected_choices:
    errors.append(
        f"RUNTIME_BACKEND_CHOICES={mod.RUNTIME_BACKEND_CHOICES!r}, "
        f"expected {expected_choices!r}"
    )
else:
    print("  [OK] RUNTIME_BACKEND_CHOICES exposes (auto, host, docker)")

# Case 2: explicit pass-through.
if mod.resolve_runtime_backend("host") != "host":
    errors.append("resolve_runtime_backend('host') did not pass through")
if mod.resolve_runtime_backend("docker") != "docker":
    errors.append("resolve_runtime_backend('docker') did not pass through")
try:
    mod.resolve_runtime_backend("garbage")
except ValueError:
    print("  [OK] resolve_runtime_backend rejects unknown values")
else:
    errors.append("resolve_runtime_backend did not raise on unknown value")

# Case 3a: auto -> host when host_runtime_available reports True.
real_probe = mod.host_runtime_available
mod.host_runtime_available = lambda platform="Ascend950PR_9599": (True, "ok-test")
try:
    resolved = mod.resolve_runtime_backend("auto")
    if resolved != "host":
        errors.append(f"auto+host-available -> {resolved!r} (expected 'host')")
    else:
        print("  [OK] auto picks 'host' when probe succeeds")
finally:
    mod.host_runtime_available = real_probe

# Case 3b: auto -> docker when host_runtime_available reports False.
mod.host_runtime_available = lambda platform="Ascend950PR_9599": (False, "no-host-test")
try:
    resolved = mod.resolve_runtime_backend("auto")
    if resolved != "docker":
        errors.append(f"auto+no-host -> {resolved!r} (expected 'docker')")
    else:
        print("  [OK] auto falls back to 'docker' when probe fails")
finally:
    mod.host_runtime_available = real_probe

# Case 4: invoke_runtime_verify dispatch.
calls = []
real_host = mod.run_host_verify
real_docker = mod.run_docker_verify

def _fake_host(kernel_path, project_dir, timeout=300, platform="Ascend950PR_9599"):
    calls.append(("host", str(kernel_path), platform, timeout))
    return {"mode": "simulator", "backend": "Model", "platform": platform,
            "status": "pass", "shapes_verified": []}

def _fake_docker(kernel_path, project_dir, timeout=300, platform="Ascend950PR_9599"):
    calls.append(("docker", str(kernel_path), platform, timeout))
    return {"mode": "simulator", "backend": "Model", "platform": platform,
            "status": "pass", "shapes_verified": []}

mod.run_host_verify = _fake_host
mod.run_docker_verify = _fake_docker
try:
    from pathlib import Path as _P
    res_h = mod.invoke_runtime_verify(
        backend="host",
        kernel_path=_P("/tmp/x.py"),
        project_dir=_P("/tmp"),
        timeout=240,
        platform="Ascend950PR_9599",
    )
    res_d = mod.invoke_runtime_verify(
        backend="docker",
        kernel_path=_P("/tmp/y.py"),
        project_dir=_P("/tmp"),
        timeout=180,
        platform="Ascend950PR_9599",
    )
    if calls != [("host", "/tmp/x.py", "Ascend950PR_9599", 240),
                 ("docker", "/tmp/y.py", "Ascend950PR_9599", 180)]:
        errors.append(f"dispatch order/args unexpected: {calls!r}")
    else:
        print("  [OK] invoke_runtime_verify dispatches to host and docker")
    if res_h.get("mode") != "simulator" or res_d.get("mode") != "simulator":
        errors.append("dispatch result missing mode=simulator")
    try:
        mod.invoke_runtime_verify(
            backend="auto",
            kernel_path=_P("/tmp/z.py"),
            project_dir=_P("/tmp"),
            timeout=60,
            platform="Ascend950PR_9599",
        )
    except ValueError:
        print("  [OK] invoke_runtime_verify refuses 'auto' (must resolve first)")
    else:
        errors.append("invoke_runtime_verify did not raise on backend='auto'")
finally:
    mod.run_host_verify = real_host
    mod.run_docker_verify = real_docker

if errors:
    print("FAIL: runtime-backend dispatch:")
    for e in errors:
        print(f"  {e}")
    sys.exit(1)
PY

echo ""
print_pass "--runtime-backend dispatch checks passed"
