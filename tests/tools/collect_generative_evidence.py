#!/usr/bin/env python3
"""Drive an opencode agent run for a given prompt, verify the generated kernel,
and write a generative evidence JSON file.

Flow:
  1. Read prompt from --prompt or from capabilities.yaml for the given --op/--dtype
  2. Create a clean test project directory
  3. Run: opencode run "<prompt>" --dir <project>
  4. Search the project for the generated kernel.py
  5. Run score_kernel.py and verify_kernel.py (static, on host)
  6. Optionally run simulator verification inside Docker (--runtime)
  7. Write evidence/<op>-<dtype>-generative.json
  8. Clean up

Exit codes: 0 = pass, 1 = fail, 2 = skip (opencode unavailable)

Usage:
    python collect_generative_evidence.py --op abs --dtype float16 [--prompt "..."] [--runtime] [--timeout 300]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[assignment]

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
EVIDENCE_DIR = REPO_ROOT / "evidence"
VERIFY_SCRIPT = SCRIPT_DIR / "verify_kernel.py"
SCORE_SCRIPT = SCRIPT_DIR / "score_kernel.py"
RUN_VERIFY_SCRIPT = SCRIPT_DIR / "run_and_verify.py"
CAPABILITIES_FILE = REPO_ROOT / "capabilities.yaml"
PYTHON = "python3.10"

DOCKER_IMAGE = os.environ.get(
    "PYASC_SIM_IMAGE", "ghcr.io/aloschilov/pyasc-sim:py3.11"
)


def _run(cmd: list[str], timeout: int = 30) -> tuple[int, str, str]:
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
        )
        return result.returncode, result.stdout.strip(), result.stderr.strip()
    except subprocess.TimeoutExpired:
        return 1, "", "timeout"
    except FileNotFoundError:
        return 2, "", f"not found: {cmd[0]}"
    except Exception as exc:
        return 1, "", str(exc)


def load_prompt_from_capabilities(op: str, dtype: str,
                                   variant: str | None = None) -> str | None:
    """Extract the prompt for a given op/dtype from capabilities.yaml.

    If *variant* is given, look up that key in ``prompt_variants`` instead
    of using the primary ``prompt``.
    """
    if not CAPABILITIES_FILE.exists():
        return None

    if yaml is not None:
        with open(CAPABILITIES_FILE) as f:
            data = yaml.safe_load(f)
    else:
        code, out, _ = _run(
            [PYTHON, "-c",
             f"import yaml,json; print(json.dumps(yaml.safe_load(open('{CAPABILITIES_FILE}'))))"],
            timeout=10,
        )
        if code != 0:
            return None
        data = json.loads(out)

    for operation in data.get("operations", []):
        if operation.get("name") != op:
            continue
        for cell in operation.get("cells", []):
            if cell.get("dtype") == dtype:
                if variant:
                    variants = cell.get("prompt_variants", {})
                    return variants.get(variant)
                return cell.get("prompt")
    return None


def create_test_project(prefix: str) -> Path:
    """Create a fresh isolated project directory for one agent run.

    Read-only assets (skills, golden) are symlinked; writable assets (teams)
    are deep-copied so each run starts clean and cannot pollute other runs.
    opencode.json is copied so skill discovery works.
    """
    tmp = Path(tempfile.mkdtemp(prefix=f"{prefix}."))

    for subdir in ("skills", "golden"):
        src = REPO_ROOT / subdir
        if src.exists():
            (tmp / subdir).symlink_to(src)

    teams_src = REPO_ROOT / "teams"
    if teams_src.exists():
        shutil.copytree(teams_src, tmp / "teams", symlinks=True)

    opencode_cfg = REPO_ROOT / "opencode.json"
    if opencode_cfg.exists():
        shutil.copy2(opencode_cfg, tmp / "opencode.json")

    subprocess.run(
        ["git", "init", "--quiet"],
        cwd=str(tmp), capture_output=True, timeout=10,
    )
    subprocess.run(
        ["git", "config", "user.email", "test@pyasc.test"],
        cwd=str(tmp), capture_output=True, timeout=10,
    )
    subprocess.run(
        ["git", "config", "user.name", "test"],
        cwd=str(tmp), capture_output=True, timeout=10,
    )
    return tmp


def find_kernel(project_dir: Path, op: str) -> Path | None:
    """Search for the generated kernel.py, preferring paths containing the op name.

    Excludes symlinked read-only directories (skills/, golden/) to avoid
    picking up golden reference kernels instead of the agent-generated one.
    """
    exclude_prefixes = (
        str(project_dir / "skills"),
        str(project_dir / "golden"),
    )

    def _is_excluded(p: str) -> bool:
        return any(p.startswith(pfx) for pfx in exclude_prefixes)

    op_clean = op.replace("_", "")
    dtype_suffixes = ["f16", "f32", "float16", "float32"]
    candidates = []
    for sfx in dtype_suffixes:
        candidates.append(project_dir / "kernels" / f"{op}_{sfx}" / "kernel.py")
        candidates.append(project_dir / f"{op}_{sfx}" / "kernel.py")
        candidates.append(project_dir / "teams" / "pyasc-kernel-dev-team" / "kernels" / f"{op}_{sfx}" / "kernel.py")
    candidates.append(project_dir / "kernel.py")
    for c in candidates:
        if c.is_file():
            return c

    all_kernels = glob.glob(str(project_dir / "**" / "kernel.py"), recursive=True)
    all_kernels = [m for m in all_kernels if not _is_excluded(m)]

    op_matches = [m for m in all_kernels if op in Path(m).parent.name or op_clean in Path(m).parent.name.replace("_", "")]
    if op_matches:
        return Path(op_matches[0])
    if all_kernels:
        print(f"  WARN: no kernel.py in a directory named after '{op}'; "
              f"using first match: {all_kernels[0]}", file=sys.stderr)
        return Path(all_kernels[0])

    py_files = glob.glob(str(project_dir / "**" / "*.py"), recursive=True)
    py_files = [f for f in py_files
                if not Path(f).name.startswith("__") and not _is_excluded(f)]
    if py_files:
        return Path(py_files[0])

    return None


def find_artifacts(project_dir: Path) -> list[str]:
    """List workflow artifacts found in the project."""
    found = []
    for name in ("kernel.py", "design.md", "self_review.md",
                  "acceptance_review.md", "verification.md"):
        matches = glob.glob(str(project_dir / "**" / name), recursive=True)
        if matches:
            found.append(name)
    return found


def run_score(kernel_path: Path, op: str | None = None,
              dtype: str | None = None) -> dict | None:
    cmd = [PYTHON, str(SCORE_SCRIPT), str(kernel_path), "--json"]
    if op:
        cmd += ["--op", op]
    if dtype:
        cmd += ["--dtype", dtype]
    code, out, _ = _run(cmd)
    try:
        return json.loads(out)
    except (json.JSONDecodeError, TypeError):
        pass
    return None


def run_static_verify(kernel_path: Path) -> str:
    code, out, _ = _run([PYTHON, str(VERIFY_SCRIPT), str(kernel_path), "--json"])
    if code == 0:
        try:
            data = json.loads(out)
            return "pass" if data.get("passed", False) else "fail"
        except json.JSONDecodeError:
            pass
    return "fail"


OP_SEMANTIC_MARKERS: dict[str, list[str]] = {
    "abs": ["asc2.abs"],
    "exp": ["asc2.exp"],
    "log": ["asc2.log"],
    "sqrt": ["asc2.sqrt"],
    "relu": ["asc2.relu"],
    "erf": ["asc2.erf"],
    "add": ["x + y", "x+y", "+ y", "+y"],
    "sub": ["x - y", "x-y", "- y", "-y"],
    "mul": ["x * y", "x*y", "* y", "*y"],
    "div": ["x / y", "x/y", "/ y", "/y"],
    "reduce_sum": ["asc2.reduce_sum", ".sum("],
    "reduce_max": ["asc2.reduce_max", ".max("],
    "reduce_min": [".min("],
    "gelu": ["asc2.erf", "erf(", "gelu", "0.5 * x", "0.5*x"],
    "leaky_relu": ["asc2.where"],
    "softmax": ["asc2.softmax", "asc2.exp", "softmax"],
    "matmul": ["asc2.matmul", "@ "],
}


def check_op_semantics(kernel_path: Path, op: str) -> dict:
    """Check whether the kernel source contains API calls expected for the operation.

    Returns {"passed": bool, "detail": str, "markers_found": list[str]}.
    """
    markers = OP_SEMANTIC_MARKERS.get(op)
    if markers is None:
        return {"passed": True, "detail": f"no semantic markers defined for '{op}'",
                "markers_found": []}

    try:
        source = kernel_path.read_text()
    except OSError:
        return {"passed": False, "detail": "could not read kernel source",
                "markers_found": []}

    found = [m for m in markers if m in source]
    passed = len(found) > 0
    detail = (f"found {found}" if passed
              else f"none of {markers} found in kernel source")
    return {"passed": passed, "detail": detail, "markers_found": found}


def run_docker_verify(kernel_path: Path, project_dir: Path, timeout: int = 300) -> dict:
    """Run simulator verification inside the Docker container.

    Mounts the repo at /repo (for tool scripts) and the project at /workspace
    (for the generated kernel). Runs run_and_verify.py from /repo.
    """
    rel_kernel = kernel_path.relative_to(project_dir)
    cmd = [
        "docker", "run", "--rm",
        "-v", f"{REPO_ROOT}:/repo:ro",
        "-v", f"{project_dir}:/workspace",
        "-w", "/workspace",
        DOCKER_IMAGE,
        "python3.11", "/repo/tests/tools/run_and_verify.py",
        str(rel_kernel), "--mode", "simulator", "--json",
    ]
    code, out, err = _run(cmd, timeout=timeout)
    result = {
        "mode": "simulator", "backend": "Model", "platform": "Ascend910B1",
    }
    if code == 0:
        result["status"] = "pass"
    elif code == 2:
        result["status"] = "skip"
    else:
        result["status"] = "fail"

    try:
        parsed = json.loads(out)
        result["detail"] = parsed.get("detail", "")
        result["shapes_verified"] = parsed.get("shapes_verified", [])
    except (json.JSONDecodeError, TypeError):
        raw = err or out
        traceback_line = ""
        for line in reversed(raw.splitlines()):
            stripped = line.strip()
            if stripped and not stripped.startswith("^") and not stripped.startswith("~"):
                traceback_line = stripped
                break
        detail = raw[-1000:] if len(raw) > 1000 else raw
        if traceback_line and traceback_line not in detail:
            detail = f"... {traceback_line}\n{detail}"
        result["detail"] = detail
        result["shapes_verified"] = []

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect generative evidence via opencode run.",
    )
    parser.add_argument("--op", required=True, help="Operation name (e.g. abs)")
    parser.add_argument("--dtype", required=True, help="Data type (e.g. float16)")
    parser.add_argument("--prompt", default=None,
                        help="Prompt to use (default: read from capabilities.yaml)")
    parser.add_argument("--timeout", type=int, default=300,
                        help="Agent timeout in seconds (default: 300)")
    parser.add_argument("--runtime", action="store_true",
                        help="Run simulator verification in Docker after generation")
    parser.add_argument("--docker-timeout", type=int, default=300,
                        help="Docker verify timeout in seconds (default: 300)")
    parser.add_argument("--notes", default="", help="Optional notes")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print JSON to stdout, don't write file")
    parser.add_argument("--keep-project", action="store_true",
                        help="Don't delete the test project after run")
    parser.add_argument("--archive-dir", default=None,
                        help="Copy generated kernel directory here for archival")
    parser.add_argument("--ci-run-url", default=None,
                        help="URL to the CI run for linking from evidence")
    parser.add_argument("--prompt-variant", default=None,
                        help="Named variant from prompt_variants in capabilities.yaml")
    parser.add_argument("--output-suffix", default=None,
                        help="Suffix appended to evidence filename (e.g. 'minimal' -> abs-f16-generative-minimal.json)")
    args = parser.parse_args()

    prompt = args.prompt
    if not prompt:
        prompt = load_prompt_from_capabilities(args.op, args.dtype,
                                               variant=args.prompt_variant)
    if not prompt:
        print(f"ERROR: No prompt provided and none found in capabilities.yaml "
              f"for {args.op}/{args.dtype}", file=sys.stderr)
        sys.exit(1)

    if shutil.which("opencode") is None:
        print("SKIP: opencode CLI not found on PATH", file=sys.stderr)
        sys.exit(2)

    print(f"  Generative evidence for {args.op}/{args.dtype}")
    print(f"  Prompt: {prompt[:80]}...")
    print()

    project = create_test_project(f"gen-{args.op}-{args.dtype}")
    print(f"  Project: {project}")

    output_file = project / "agent-output.txt"
    agent_completed = False
    try:
        env = os.environ.copy()
        env["NODE_TLS_REJECT_UNAUTHORIZED"] = "0"

        opencode_cmd = f'opencode run "{prompt}" --dir "{project}"'
        cmd = ["script", "-qc", opencode_cmd, "/dev/null"]

        print(f"  Running opencode (timeout={args.timeout}s)...")
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=args.timeout, env=env,
        )
        with open(output_file, "w") as f:
            f.write(result.stdout)
            f.write(result.stderr)
        agent_completed = True
        print("  Agent completed.")
    except subprocess.TimeoutExpired:
        print(f"  Agent timed out after {args.timeout}s")
    except Exception as exc:
        print(f"  Agent error: {exc}")

    kernel = find_kernel(project, args.op)
    artifacts = find_artifacts(project)
    print(f"  Kernel: {kernel}")
    print(f"  Artifacts: {artifacts}")

    score_data = None
    static_result = "fail"
    verification: dict = {
        "mode": "static_only", "status": "fail", "shapes_verified": [],
    }

    semantic_check: dict = {"passed": False, "detail": "no kernel found",
                            "markers_found": []}

    if kernel and kernel.is_file():
        score_data = run_score(kernel, op=args.op, dtype=args.dtype)
        static_result = run_static_verify(kernel)
        semantic_check = check_op_semantics(kernel, args.op)
        print(f"  Static verify: {static_result}")
        print(f"  Semantic check: {'pass' if semantic_check['passed'] else 'FAIL'}"
              f" — {semantic_check['detail']}")
        if score_data:
            print(f"  Score: {score_data.get('score', '?')}/10")
        verification = {
            "mode": "static_only", "status": static_result, "shapes_verified": [],
        }

        if args.runtime:
            print("  Running simulator in Docker...")
            rt = run_docker_verify(kernel, project, timeout=args.docker_timeout)
            verification = rt
            print(f"  Runtime: {rt['status']}")
    else:
        print("  No kernel found — generation failed")

    try:
        kernel_rel = str(kernel.relative_to(project)) if kernel else ""
    except ValueError:
        kernel_rel = str(kernel) if kernel else ""

    evidence: dict = {
        "schema_version": "2",
        "kind": "generative",
        "operation": args.op,
        "dtype": args.dtype,
        "prompt": prompt,
        "kernel_path": kernel_rel,
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "agent": {
            "platform": "opencode",
            "timeout_s": args.timeout,
            "completed": agent_completed,
            "artifacts_found": artifacts,
        },
        "verification": verification,
        "semantic_check": semantic_check,
        "score": {
            "value": score_data.get("score", 0.0) if score_data else 0.0,
            "threshold": 12,
            "accepted": score_data.get("accepted", False) if score_data else False,
            "checks": score_data.get("checks", {}) if score_data else {},
        },
        "static_verify": static_result,
        "notes": args.notes,
    }
    if args.ci_run_url:
        evidence["ci_run_url"] = args.ci_run_url

    dtype_short = args.dtype.replace("float", "f")
    suffix = f"-{args.output_suffix}" if args.output_suffix else ""
    out_name = f"{args.op}-{dtype_short}-generative{suffix}.json"
    out_path = EVIDENCE_DIR / out_name

    history: list[dict] = []
    if out_path.exists():
        try:
            with open(out_path) as f:
                prev = json.load(f)
            history = prev.get("history", [])
            prev_runtime_ok = prev.get("verification", {}).get("status") == "pass" if prev.get("verification", {}).get("mode") != "static_only" else True
            history.append({
                "date": prev.get("date", ""),
                "overall_pass": (
                    prev.get("static_verify") == "pass"
                    and prev.get("score", {}).get("accepted", False)
                    and prev.get("semantic_check", {}).get("passed", False)
                    and bool(prev.get("kernel_path"))
                    and prev_runtime_ok
                ),
                "score": prev.get("score", {}).get("value", 0),
                "static": prev.get("static_verify", ""),
                "runtime": prev.get("verification", {}).get("status", ""),
                "semantic": prev.get("semantic_check", {}).get("passed", False),
                "skipped": prev.get("verification", {}).get("status") == "skip",
            })
            if len(history) > 30:
                history = history[-30:]
        except (json.JSONDecodeError, OSError, KeyError):
            pass

    evidence["history"] = history

    if args.dry_run:
        print()
        print(json.dumps(evidence, indent=2))
    else:
        EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(evidence, f, indent=2)
            f.write("\n")
        print(f"  Written: {out_path.relative_to(REPO_ROOT)}")

    if args.archive_dir and kernel and kernel.is_file():
        archive_dest = Path(args.archive_dir) / f"{args.op}-{dtype_short}"
        archive_dest.mkdir(parents=True, exist_ok=True)
        kernel_dir = kernel.parent
        for item in kernel_dir.iterdir():
            if item.is_file():
                shutil.copy2(item, archive_dest / item.name)
        print(f"  Archived kernel to: {archive_dest}")

    if not args.keep_project:
        shutil.rmtree(project, ignore_errors=True)
        print("  Cleaned up project directory")
    else:
        print(f"  Project kept at: {project}")

    runtime_ok = verification.get("status") == "pass" if args.runtime else True
    overall_pass = (
        static_result == "pass"
        and evidence["score"]["accepted"]
        and semantic_check["passed"]
        and kernel is not None
        and runtime_ok
    )
    print(f"  Overall: {'pass' if overall_pass else 'fail'}")
    sys.exit(0 if overall_pass else 1)


if __name__ == "__main__":
    main()
