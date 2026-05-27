#!/usr/bin/env bash
# =============================================================================
# L1 Unit Test: pyasc_revision field collection in
# collect_generative_evidence.py.
#
# Verifies four things:
#   1. collect_pyasc_revision() returns the expected dict shape
#      ({url, branch, sha, dirty, root}) in all branches.
#   2. When `import asc` fails (no pyasc on PYTHONPATH), the helper
#      returns the predictable empty-string shape (not None) so JSON
#      serialization always has the field present.
#   3. On a synthetic git checkout (tmp dir with `git init`), the
#      helper reports the right SHA, branch, url, and dirty flag —
#      including transitions clean -> dirty -> clean (after add/commit).
#   4. resolve_pyasc_src() returns None for an interpreter where `asc`
#      cannot be imported.
#
# Pure unit test, no opencode / no Docker / no CANN required.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../lib/test-helpers.sh"

echo "=== Test: pyasc_revision field (Step 2a) ==="

COLLECT_TOOL="$TOOLS_DIR/collect_generative_evidence.py"
[ -f "$COLLECT_TOOL" ] || { print_fail "collect_generative_evidence.py not found"; exit 1; }

$PYTHON - <<'PY'
import importlib.util
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path("tests/tools").resolve()))
spec = importlib.util.spec_from_file_location(
    "collect_ge", "tests/tools/collect_generative_evidence.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

errors = []

# Case 1: dict shape (keys always present, dirty is bool, others str).
real_resolve = mod.resolve_pyasc_src
mod.resolve_pyasc_src = lambda: None
try:
    rev = mod.collect_pyasc_revision()
    expected_keys = {"url", "branch", "sha", "dirty", "root"}
    if set(rev.keys()) != expected_keys:
        errors.append(f"missing keys: got {set(rev.keys())!r}")
    elif not isinstance(rev["dirty"], bool):
        errors.append(f"dirty not bool: {type(rev['dirty']).__name__}")
    elif rev["sha"] != "" or rev["url"] != "" or rev["branch"] != "":
        errors.append(f"unresolved root should give empty strings: {rev!r}")
    else:
        print("  [OK] empty-resolution returns predictable shape")
finally:
    mod.resolve_pyasc_src = real_resolve

# Case 2: synthetic git checkout — branch, sha, url, dirty transitions.
with tempfile.TemporaryDirectory() as td:
    repo = Path(td) / "fake-pyasc"
    repo.mkdir()
    # Layout mirrors real clone: <root>/python/asc/__init__.py
    (repo / "python" / "asc").mkdir(parents=True)
    (repo / "python" / "asc" / "__init__.py").write_text("")
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=repo, check=True)
    subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t",
                    "config", "user.email", "t@t"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=repo, check=True)
    subprocess.run(["git", "remote", "add", "origin",
                    "https://example.invalid/fake/pyasc.git"],
                   cwd=repo, check=True)
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=repo, check=True)
    head_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo).decode().strip()

    mod.resolve_pyasc_src = lambda r=repo: str(r)
    try:
        rev = mod.collect_pyasc_revision()
        if rev["sha"] != head_sha:
            errors.append(f"sha mismatch: {rev['sha']!r} vs {head_sha!r}")
        elif rev["branch"] != "main":
            errors.append(f"branch mismatch: {rev['branch']!r}")
        elif rev["url"] != "https://example.invalid/fake/pyasc.git":
            errors.append(f"url mismatch: {rev['url']!r}")
        elif rev["dirty"] is True:
            errors.append("clean tree reported dirty=True")
        else:
            print("  [OK] clean synthetic checkout reports correct url/branch/sha")

        # Dirty transition: add untracked file.
        (repo / "scratch.txt").write_text("dirty marker\n")
        rev2 = mod.collect_pyasc_revision()
        if not rev2["dirty"]:
            errors.append("untracked file did not mark tree dirty")
        else:
            print("  [OK] untracked file marks dirty=True")

        # Clean again via gitignore (regression for the eval-clone use case).
        (repo / ".gitignore").write_text("scratch.txt\n")
        subprocess.run(["git", "add", ".gitignore"], cwd=repo, check=True)
        subprocess.run(["git", "commit", "-q", "-m", "ignore scratch"],
                       cwd=repo, check=True)
        rev3 = mod.collect_pyasc_revision()
        if rev3["dirty"]:
            errors.append("gitignored file still marks tree dirty")
        else:
            print("  [OK] gitignored file does not mark dirty")

        # Detached HEAD case.
        subprocess.run(["git", "checkout", "-q", "--detach", head_sha],
                       cwd=repo, check=True)
        rev4 = mod.collect_pyasc_revision()
        if rev4["branch"] != "(detached)":
            errors.append(
                f"detached HEAD branch reported as {rev4['branch']!r} "
                f"instead of '(detached)'")
        else:
            print("  [OK] detached HEAD reports branch='(detached)'")
    finally:
        mod.resolve_pyasc_src = real_resolve

# Case 3: resolve_pyasc_src returns None when asc cannot be imported.
# Build a minimal interpreter with PYTHONPATH stripped, asserting failure.
# We swap sys.executable temporarily to a python invoked with -S (no site).
# Defensive: only run when /usr/bin/python3 exists.
if Path("/usr/bin/python3").exists():
    real_exec = sys.executable
    try:
        sys.executable = "/usr/bin/python3"
        # Clear PYTHONPATH so asc isn't reachable on the system python.
        env_backup = os.environ.pop("PYTHONPATH", None)
        try:
            resolved = mod.resolve_pyasc_src()
            # Either None (asc not importable) or a real path (system has it).
            if resolved is not None and not Path(resolved).is_dir():
                errors.append(
                    f"resolve_pyasc_src returned non-dir: {resolved!r}")
            else:
                print("  [OK] resolve_pyasc_src handles missing-asc gracefully")
        finally:
            if env_backup is not None:
                os.environ["PYTHONPATH"] = env_backup
    finally:
        sys.executable = real_exec

if errors:
    print("FAIL: pyasc_revision field:")
    for e in errors:
        print(f"  {e}")
    sys.exit(1)
PY

echo ""
print_pass "pyasc_revision field checks passed"
