#!/usr/bin/env python3
"""Drive an opencode agent run for a given prompt, verify the generated kernel,
and write a generative evidence JSON file.

Flow:
  1. Read prompt from --prompt or from capabilities.yaml for the given --op/--dtype
  2. For each attempt:
     a. Create a clean test project directory (layout depends on --skills-mode)
     b. Run: opencode run "<prompt>" --dir <project> --format json
     c. Parse the agent's JSON event stream for tokens and model
     d. Search the project for the generated kernel.py
     e. Run score_kernel.py and verify_kernel.py (static, on host)
     f. Optionally run simulator verification inside Docker (--runtime)
     g. Clean up
  3. Write evidence/<op>-<dtype>-generative[-<profile>-<mode>].json

Exit codes: 0 = pass, 1 = fail, 2 = skip (opencode unavailable)

Usage:
    python collect_generative_evidence.py --op abs --dtype float16 \
        [--prompt "..."] [--runtime] [--timeout 300] \
        [--skills-mode on|off] [--model-profile cloud-default|local-...] \
        [--opencode-config /path/to/config.json] \
        [--max-attempts 3] [--fallback-variant guided]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from string import Template

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
DEFAULT_AGENTS_MD = REPO_ROOT / "docs" / "baseline" / "pyasc-fork-AGENTS.md"
PYTHON = "python3.10"


def _snapshot_asc_file() -> str | None:
    """Phase 10: capture the resolved location of the ``asc`` package.

    Prefers the in-process value (``sys.modules['asc'].__file__``) but
    falls back to ``importlib.util.find_spec`` so we still get a signal
    even when ``asc`` was never imported by this process (e.g. running
    on a host with the package installed system-wide but not yet
    referenced). Returns ``None`` only when ``asc`` is genuinely
    unresolvable.

    Used in pairs around the ``opencode run`` subprocess to catch the
    pip-install-e diversion captured in Appendix B of
    ``docs/skill-value-q1-findings.md``: a P4 baseline AGENTS.md
    routinely runs ``pip install -e .`` against the agent's working
    directory, which on disk swaps the resolved ``asc`` to a different
    clone. The next 18 trials in the matrix then quietly silently fail
    with "pyasc working tree at ... is dirty". A pre/post snapshot
    catches the mutation immediately and the trial is failed cleanly
    with ``failure_category: pyasc_root_mutated_during_run``.
    """
    mod = sys.modules.get("asc")
    in_proc = getattr(mod, "__file__", None) if mod is not None else None
    if in_proc:
        return in_proc
    try:
        import importlib.util as _imp_util
        spec = _imp_util.find_spec("asc")
        return spec.origin if spec else None
    except (ImportError, ValueError):
        return None


# Phase 0 protocol-axis mapping. Single source of truth for what each
# protocol id resolves to. See docs/evaluation-methodology.md
# "Protocol-axis CI mapping (Phase 0)" for the full contract.
PROTOCOL_TABLE: dict[str, dict] = {
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


def derive_protocol(protocol_id: str) -> dict:
    """Return the resolved knobs for ``protocol_id`` (Phase 0).

    Raises ``KeyError`` for unknown ids. Callers that want a graceful
    error should validate against ``PROTOCOL_TABLE.keys()`` first.
    """
    return dict(PROTOCOL_TABLE[protocol_id])

DOCKER_IMAGE = os.environ.get(
    "PYASC_SIM_IMAGE", "ghcr.io/aloschilov/pyasc-sim:py3.11"
)

PROFILE_DIR = REPO_ROOT / "docker" / "opencode-profiles"
KNOWN_PROFILES = {
    "cloud-default",
    "local-qwen-coder-7b",
    "local-llama-3.1-8b",
}
SCHEMA_VERSION = "4"
DEFAULT_PYASC_EVAL_ROOT = "/home/aloschilov/workspace/pyasc-v2-eval"


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


def _load_capabilities_data() -> dict | None:
    """Load capabilities.yaml as a dict (or None if unavailable)."""
    if not CAPABILITIES_FILE.exists():
        return None
    if yaml is not None:
        with open(CAPABILITIES_FILE) as f:
            return yaml.safe_load(f)
    code, out, _ = _run(
        [PYTHON, "-c",
         f"import yaml,json; print(json.dumps(yaml.safe_load(open('{CAPABILITIES_FILE}'))))"],
        timeout=10,
    )
    if code != 0:
        return None
    return json.loads(out)


def _find_cell(data: dict, op: str, dtype: str) -> dict | None:
    for operation in data.get("operations", []):
        if operation.get("name") != op:
            continue
        for cell in operation.get("cells", []):
            if cell.get("dtype") == dtype:
                return cell
    return None


def load_prompt_from_capabilities(op: str, dtype: str,
                                   variant: str | None = None) -> str | None:
    """Extract the prompt for a given op/dtype from capabilities.yaml.

    If *variant* is given, look up that key in ``prompt_variants`` instead
    of using the primary ``prompt``.
    """
    data = _load_capabilities_data()
    if data is None:
        return None
    cell = _find_cell(data, op, dtype)
    if cell is None:
        return None
    if variant:
        variants = cell.get("prompt_variants", {})
        val = variants.get(variant)
        # Phase 2 Stage 2.3: oracle_guided may be a mapping with
        # ``prompt`` + ``examples_policy``. Normalize to the prompt
        # string here; the policy is reported by check_capabilities.py
        # and is consumed by the harness at allowed_context resolution
        # time, not by the agent runtime.
        if isinstance(val, dict):
            return val.get("prompt")
        return val
    return cell.get("prompt")


def load_platform_from_capabilities(op: str, dtype: str,
                                     default: str = "Ascend950PR_9599") -> str:
    """Look up the simulator platform for a given op/dtype cell.

    All goldens target Ascend950PR_9599 (C310). Falls back to the default
    if the cell or field is missing.
    """
    data = _load_capabilities_data()
    if data is None:
        return default
    cell = _find_cell(data, op, dtype)
    if cell is None:
        return default
    return cell.get("platform") or default


_MINIMAL_OPENCODE_JSON: dict = {
    "$schema": "https://opencode.ai/config.json",
    "permission": {
        "read": "allow",
        "edit": "allow",
        "bash": "allow",
        "glob": "allow",
        "grep": "allow",
        "list": "allow",
        "skill": "allow",
        "task": "allow",
    },
}


def resolve_opencode_config(
    profile: str, override_path: str | None = None,
) -> tuple[dict, Path | None]:
    """Resolve an opencode config dict for a given profile.

    Resolution order:
      1. If ``override_path`` is given, load that JSON template directly.
      2. Otherwise, look up the profile in ``docker/opencode-profiles/<profile>.json``.
      3. If neither exists, fall back to ``_MINIMAL_OPENCODE_JSON``.

    Template files may reference environment variables using ``${VAR}`` —
    they are substituted from the current process environment (or replaced
    with an empty string when unset). This lets us point local-model
    profiles at a sidecar Ollama endpoint via ``OLLAMA_BASE_URL`` without
    baking secrets into the repo.

    Returns the merged dict and the path of the source template (or None
    when the minimal fallback is used).
    """
    if override_path:
        path = Path(override_path)
    else:
        path = PROFILE_DIR / f"{profile}.json"

    if not path.exists():
        return dict(_MINIMAL_OPENCODE_JSON), None

    raw = path.read_text()
    expanded = Template(raw).safe_substitute(os.environ)
    try:
        data = json.loads(expanded)
    except json.JSONDecodeError as exc:
        print(f"  WARN: profile template {path} not valid JSON after "
              f"env substitution ({exc}); falling back to minimal config",
              file=sys.stderr)
        return dict(_MINIMAL_OPENCODE_JSON), None
    return data, path


def build_opencode_json(
    profile: str, skills_mode: str, override_path: str | None = None,
) -> dict:
    """Build the per-project opencode.json for a (profile, mode) combination.

    When ``skills_mode`` is ``"on"`` the resulting config enables skill
    discovery from ``./skills``. When ``"off"`` the skills section is
    removed entirely so the agent operates as raw opencode with no
    skill-stack hints.
    """
    cfg, _ = resolve_opencode_config(profile, override_path)
    if skills_mode == "on":
        cfg["skills"] = {"paths": ["./skills"]}
    else:
        cfg.pop("skills", None)
    return cfg


def create_test_project(
    prefix: str,
    skills_mode: str = "on",
    profile: str = "cloud-default",
    opencode_config_override: str | None = None,
    agents_md_path: Path | None = None,
) -> Path:
    """Create a fresh isolated project directory for one agent run.

    Layout depends on ``skills_mode``:
      * ``"on"`` — symlink ``skills/`` and ``golden/``; copy ``teams/``
        (which contains the skill-stack ``AGENTS.md`` that activates
        the skill workflow).
      * ``"off"`` — symlink only ``golden/`` (kept as reference data so
        the input surface is comparable); do not copy ``teams/`` or link
        ``skills/``. The resulting workspace has no ``AGENTS.md`` and no
        ``SKILL.md`` files visible to the agent.

    When ``agents_md_path`` is given (Phase 0 ``P4`` protocol layout),
    that file is copied to ``<project>/AGENTS.md`` *after* the layout
    work above. The caller is responsible for refusing the conflicting
    combination of ``skills_mode='on'`` + ``agents_md_path is not None``
    (which would mount two different AGENTS.md files and conflate the
    skill-stack AGENTS.md value with the baseline AGENTS.md value); see
    ``docs/evaluation-methodology.md`` "Protocol-axis CI mapping".

    The on-disk ``opencode.json`` is generated from the chosen profile
    template (resolved via ``resolve_opencode_config``) with the skill
    section toggled to match ``skills_mode``.
    """
    tmp = Path(tempfile.mkdtemp(prefix=f"{prefix}."))

    if skills_mode == "on":
        for subdir in ("skills", "golden"):
            src = REPO_ROOT / subdir
            if src.exists():
                (tmp / subdir).symlink_to(src)
        teams_src = REPO_ROOT / "teams"
        if teams_src.exists():
            # Copy team metadata (AGENTS.md, quickstart, …) but NOT the
            # checked-in kernels/ workspace. Otherwise find_kernel() would
            # pick up a stale kernel from a prior manual run as if the
            # agent had just generated it, producing a false-positive pass
            # (e.g. when the agent timed out or otherwise failed).
            shutil.copytree(
                teams_src, tmp / "teams", symlinks=True,
                ignore=shutil.ignore_patterns("kernels"),
            )
            # Recreate the empty kernels/ workspace each agent expects.
            for team_dir in teams_src.iterdir():
                if team_dir.is_dir() and (team_dir / "kernels").is_dir():
                    (tmp / "teams" / team_dir.name / "kernels").mkdir(
                        parents=True, exist_ok=True,
                    )
    else:
        # Skills-off baseline: keep `golden/` available so reference data is
        # comparable, but no skills, no teams/AGENTS.md.
        src = REPO_ROOT / "golden"
        if src.exists():
            (tmp / "golden").symlink_to(src)

    if agents_md_path is not None:
        src = Path(agents_md_path)
        if not src.exists():
            raise FileNotFoundError(
                f"--agents-md-source not found: {src}"
            )
        shutil.copy2(src, tmp / "AGENTS.md")

    cfg = build_opencode_json(profile, skills_mode, opencode_config_override)
    (tmp / "opencode.json").write_text(json.dumps(cfg, indent=2) + "\n")

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
        print(f"  WARN: found kernel.py files but none in a directory matching '{op}': "
              f"{[Path(m).parent.name for m in all_kernels]}", file=sys.stderr)

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


from semantic_markers import OP_SEMANTIC_MARKERS


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


RUNTIME_BACKEND_CHOICES = ("auto", "host", "docker")


def _git_capture(args: list[str], *, cwd: str) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", cwd, *args],
            capture_output=True, text=True, timeout=10,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return ""
    if out.returncode != 0:
        return ""
    return out.stdout.strip()


def resolve_pyasc_src() -> str | None:
    """Resolve the on-disk root of the imported pyasc git checkout.

    Starts from ``asc.__file__`` and walks up at most 5 levels looking
    for a ``.git`` directory or file (worktree). For a normal editable
    install rooted at ``.../pyasc-v2-eval/python/asc/__init__.py`` this
    lands on ``.../pyasc-v2-eval``. Returns ``None`` when ``asc`` cannot
    be imported or no ``.git`` is found along the path. Kept
    dependency-free; the harness already shells out to ``sys.executable
    -c "import asc"`` elsewhere.
    """
    probe_script = (
        "import asc, os, sys\n"
        "p = os.path.abspath(asc.__file__)\n"
        "for _ in range(5):\n"
        "    p = os.path.dirname(p)\n"
        "    if os.path.exists(os.path.join(p, '.git')):\n"
        "        print(p)\n"
        "        sys.exit(0)\n"
        "sys.exit(1)\n"
    )
    try:
        probe = subprocess.run(
            [sys.executable, "-c", probe_script],
            capture_output=True, text=True, timeout=15,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None
    if probe.returncode != 0:
        return None
    path = probe.stdout.strip()
    return path or None


def collect_pyasc_revision() -> dict:
    """Capture ``{url, branch, sha, dirty, root}`` of the imported pyasc.

    Returns a dict with empty strings when something is missing so the
    field always lands in the JSON with a predictable shape. The
    ``dirty`` flag respects ``.git/info/exclude`` and ``.gitignore`` —
    the eval clone keeps ``EVAL-ONLY.README.md`` excluded so its presence
    does not mark the tree dirty.
    """
    root = resolve_pyasc_src()
    if not root or not os.path.exists(os.path.join(root, ".git")):
        return {"url": "", "branch": "", "sha": "", "dirty": False, "root": root or ""}
    url = _git_capture(["config", "--get", "remote.origin.url"], cwd=root)
    sha = _git_capture(["rev-parse", "HEAD"], cwd=root)
    branch = _git_capture(["symbolic-ref", "--short", "HEAD"], cwd=root)
    if not branch:
        branch = "(detached)"
    porcelain = _git_capture(["status", "--porcelain"], cwd=root)
    dirty = bool(porcelain.strip())
    return {
        "url": url,
        "branch": branch,
        "sha": sha,
        "dirty": dirty,
        "root": root,
    }


def host_runtime_available(platform: str = "Ascend950PR_9599") -> tuple[bool, str]:
    """Probe whether the host has a usable CANN simulator install.

    Returns ``(ok, reason)`` where ``reason`` is a short human-readable
    explanation suitable for logging when host runtime is unavailable.

    The probe matches the conditions ``run_and_verify._simulator_env``
    relies on: ``ASCEND_HOME_PATH`` set, the platform-specific lib dir
    present (``$ASCEND_HOME_PATH/tools/simulator/<platform>/lib``), and
    the ``asc`` Python package importable by the current interpreter.
    Kept cheap so it can be called once per ``--runtime-backend=auto``
    invocation.
    """
    ascend = os.environ.get("ASCEND_HOME_PATH")
    if not ascend:
        return False, "ASCEND_HOME_PATH is not set"
    sim_lib = os.path.join(ascend, "tools", "simulator", platform, "lib")
    if not os.path.isdir(sim_lib):
        return False, f"simulator lib dir missing: {sim_lib}"
    try:
        probe = subprocess.run(
            [sys.executable, "-c", "import asc"],
            capture_output=True, text=True, timeout=15,
        )
    except subprocess.TimeoutExpired:
        return False, "`import asc` probe timed out"
    except FileNotFoundError:
        return False, f"interpreter not found: {sys.executable}"
    if probe.returncode != 0:
        snippet = (probe.stderr or probe.stdout or "").strip()[-200:]
        return False, f"`import asc` failed: {snippet}"
    return True, "host CANN runtime available"


def resolve_runtime_backend(
    backend: str, platform: str = "Ascend950PR_9599",
) -> str:
    """Resolve ``--runtime-backend`` to a concrete backend.

    ``auto`` picks ``host`` when :func:`host_runtime_available` succeeds
    and falls back to ``docker`` otherwise. ``host`` and ``docker`` are
    pass-through. Unknown values raise ``ValueError``.
    """
    if backend not in RUNTIME_BACKEND_CHOICES:
        raise ValueError(
            f"unknown runtime backend {backend!r}; "
            f"choices: {RUNTIME_BACKEND_CHOICES}"
        )
    if backend == "host":
        return "host"
    if backend == "docker":
        return "docker"
    ok, reason = host_runtime_available(platform)
    if ok:
        return "host"
    print(
        f"  runtime-backend=auto: host unavailable ({reason}); using docker",
        file=sys.stderr,
    )
    return "docker"


def host_pyasc_pin_error(
    pyasc_revision: dict,
    *,
    eval_root: str,
    allow_dirty: bool,
) -> tuple[str | None, str | None]:
    """Validate the host pyasc pin for a ``--runtime`` collection.

    Only meaningful for the ``host`` backend: the docker backend carries
    its own pinned pyasc inside the image, so callers MUST skip this check
    when the resolved backend is ``docker`` (otherwise CI runners, which
    have no host ``asc``, would abort before ever spawning the container).

    Returns ``(error, warning)``. ``error`` is a fatal message (caller
    should print + exit) or ``None`` when the pin is acceptable.
    ``warning`` is an advisory message (caller should print but continue)
    or ``None``.
    """
    if not pyasc_revision["sha"]:
        return (
            "--runtime requires an importable pyasc; "
            "`import asc` did not resolve to a git checkout. "
            "Install the editable from /home/aloschilov/workspace/pyasc-v2-eval "
            "(see docs/cann-setup.md).",
            None,
        )
    eval_basename = os.path.basename(eval_root.rstrip("/"))
    if not pyasc_revision["root"].rstrip("/").endswith(eval_basename):
        if allow_dirty:
            return (
                None,
                f"imported pyasc is at {pyasc_revision['root']!r}, "
                f"not under {eval_root!r}. Recording anyway because "
                f"--allow-dirty-pyasc is set; result is NOT comparable "
                f"to the canonical baseline.",
            )
        return (
            f"--runtime resolved pyasc to {pyasc_revision['root']!r} but "
            f"expected a tree under {eval_root!r}. Something (likely an "
            f"opencode skill step) overwrote the editable install. "
            f"Re-pin with:\n    pip install -e {eval_root}\n"
            f"or pass --allow-dirty-pyasc to record evidence anyway.",
            None,
        )
    if pyasc_revision["dirty"] and not allow_dirty:
        return (
            f"pyasc working tree at {pyasc_revision['root']} is dirty; "
            f"refuse to record evidence against an un-pinnable revision. "
            f"Pass --allow-dirty-pyasc to override (e.g. while iterating "
            f"locally).",
            None,
        )
    return None, None


def run_host_verify(kernel_path: Path, project_dir: Path, timeout: int = 300,
                    platform: str = "Ascend950PR_9599") -> dict:
    """Run simulator verification directly on the host via run_and_verify.py.

    Mirrors :func:`run_docker_verify` but invokes ``run_and_verify.py``
    in the current Python interpreter so the host's editable
    ``asc``/``asc2`` packages and the CANN simulator libs (resolved
    via ``ASCEND_HOME_PATH`` + ``_simulator_env``) are used directly —
    no Docker. Pattern copied from
    :func:`tests.tools.collect_evidence.collect_runtime`.

    Returns the same evidence-block shape as ``run_docker_verify`` so
    callers don't need to special-case the backend.

    ``project_dir`` is accepted (and ignored) for parity with
    ``run_docker_verify``; the host path can read the kernel via its
    absolute path without any bind mounts.
    """
    _ = project_dir
    inner_timeout = max(60, timeout - 30)
    cmd = [
        sys.executable,
        str(RUN_VERIFY_SCRIPT),
        str(kernel_path),
        "--mode", "simulator",
        "--backend", "Model",
        "--platform", platform,
        "--timeout", str(inner_timeout),
        "--json",
    ]
    code, out, err = _run(cmd, timeout=timeout)
    result = {
        "mode": "simulator", "backend": "Model", "platform": platform,
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


def invoke_runtime_verify(
    *,
    backend: str,
    kernel_path: Path,
    project_dir: Path,
    timeout: int,
    platform: str,
) -> dict:
    """Dispatch a runtime verification to the chosen backend.

    ``backend`` must already be resolved (``host`` or ``docker``);
    callers that pass ``auto`` should first run it through
    :func:`resolve_runtime_backend`.
    """
    if backend == "host":
        return run_host_verify(
            kernel_path, project_dir, timeout=timeout, platform=platform,
        )
    if backend == "docker":
        return run_docker_verify(
            kernel_path, project_dir, timeout=timeout, platform=platform,
        )
    raise ValueError(
        f"invoke_runtime_verify: unresolved backend {backend!r} "
        f"(must be 'host' or 'docker'; resolve 'auto' first)"
    )


def run_docker_verify(kernel_path: Path, project_dir: Path, timeout: int = 300,
                      platform: str = "Ascend950PR_9599") -> dict:
    """Run simulator verification inside the Docker container.

    Mounts the repo at /repo (for tool scripts) and the project at /workspace
    (for the generated kernel). Runs run_and_verify.py from /repo.

    *platform* is forwarded to run_and_verify.py via ``--platform``. The
    default ``Ascend950PR_9599`` (C310) is the only platform the stack
    targets. Heavy CANN simulator runs need a high open-files ulimit.

    *timeout* bounds both the outer ``docker run`` subprocess and the inner
    simulator-kernel subprocess (forwarded as ``run_and_verify.py
    --timeout``). They must agree, otherwise the inner default of 300 s
    short-circuits the outer wait — see the 2026-05-07 nightly post-mortem
    in docs/perf-methodology/.
    """
    rel_kernel = kernel_path.relative_to(project_dir)
    cmd = [
        "docker", "run", "--rm",
        "--ulimit", "nofile=65536:65536",
        "-v", f"{REPO_ROOT}:/repo:ro",
        "-v", f"{project_dir}:/workspace",
        "-w", "/workspace",
        DOCKER_IMAGE,
        "python3.11", "/repo/tests/tools/run_and_verify.py",
        str(rel_kernel), "--mode", "simulator", "--json",
        "--platform", platform,
        "--timeout", str(max(60, timeout - 30)),
    ]
    code, out, err = _run(cmd, timeout=timeout)
    result = {
        "mode": "simulator", "backend": "Model", "platform": platform,
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


def _coerce_int(value) -> int:
    """Return ``int(value)`` or 0 when the value isn't an int-like scalar."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def extract_tokens_and_model(agent_output_path: Path) -> dict:
    """Best-effort extraction of token usage and model id from opencode output.

    The opencode CLI is invoked with ``--format json`` (1.4.x), which writes
    a stream of JSON events on stdout. Each ``step_finish`` event carries a
    ``part.tokens`` object with the shape::

        {"total": int, "input": int, "output": int, "reasoning": int,
         "cache": {"read": int, "write": int}}

    and a ``part.cost`` float in USD. The event stream does *not* carry a
    model id today; that is recorded separately (from the resolved
    opencode.json) in the caller. We still scan for ``modelID`` / ``model``
    keys defensively so future opencode releases that surface the model in
    the stream "just work".

    We aggregate token counts across events because usage is reported per
    assistant turn (i.e. per ``step_finish``); one ``opencode run`` can
    contain many turns.

    The function is tolerant of:
      * malformed lines (skipped silently)
      * the ``script -qc`` TTY wrapper (header + footer lines)
      * single-document JSON instead of JSONL
      * the legacy ``usage.input_tokens`` shape used by older session
        exports — kept as a fallback so external session JSON still works.

    Returns ``{"input": int, "output": int, "cache_read": int,
    "total": int, "cost_usd": float, "model": str | None}``.
    """
    result: dict = {
        "input": 0, "output": 0, "cache_read": 0, "total": 0,
        "cost_usd": 0.0, "model": None,
    }
    if not agent_output_path.exists():
        return result

    try:
        data = agent_output_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return result

    def _visit(entry) -> None:
        if isinstance(entry, dict):
            tokens = entry.get("tokens")
            if isinstance(tokens, dict):
                # opencode --format json schema (per step_finish event).
                result["input"] += _coerce_int(tokens.get("input", 0))
                result["output"] += _coerce_int(tokens.get("output", 0))
                cache = tokens.get("cache")
                if isinstance(cache, dict):
                    result["cache_read"] += _coerce_int(cache.get("read", 0))
            cost = entry.get("cost")
            if isinstance(cost, (int, float)):
                result["cost_usd"] += float(cost)
            usage = entry.get("usage")
            if isinstance(usage, dict):
                # Legacy shape (older opencode session exports, kept for
                # backward compatibility with analyze-token-usage outputs).
                result["input"] += _coerce_int(usage.get("input_tokens", 0))
                result["output"] += _coerce_int(usage.get("output_tokens", 0))
                result["cache_read"] += _coerce_int(
                    usage.get("cache_read_input_tokens", 0)
                )
            for key in ("modelID", "model_id", "model"):
                m = entry.get(key)
                if isinstance(m, str) and m and result["model"] is None:
                    result["model"] = m
            for v in entry.values():
                _visit(v)
        elif isinstance(entry, list):
            for v in entry:
                _visit(v)

    try:
        doc = json.loads(data)
        _visit(doc)
    except json.JSONDecodeError:
        for line in data.splitlines():
            line = line.strip()
            if not line or line[0] not in "{[":
                continue
            try:
                _visit(json.loads(line))
            except json.JSONDecodeError:
                continue

    if result["model"] is None:
        match = re.search(r'"(?:modelID|model_id|model)"\s*:\s*"([^"]+)"', data)
        if match:
            result["model"] = match.group(1)

    result["total"] = result["input"] + result["output"]
    result["cost_usd"] = round(result["cost_usd"], 6)
    return result


def _first_provider_model(cfg: dict) -> str | None:
    providers = cfg.get("provider")
    if not isinstance(providers, dict):
        return None
    for provider_id, pcfg in providers.items():
        if not isinstance(pcfg, dict):
            continue
        models = pcfg.get("models")
        if isinstance(models, dict) and models:
            model_id = next(iter(models.keys()), None)
            if model_id:
                return f"{provider_id}/{model_id}"
    return None


def resolve_configured_model(project_dir: Path) -> str | None:
    """Resolve the LLM model id used by opencode for this attempt.

    opencode's ``--format json`` event stream does not surface the model
    id, so we fall back to reading the resolved ``opencode.json`` that
    actually drove the run. We look at, in order:

      1. The project-local ``<project>/opencode.json`` (which is what we
         wrote via ``build_opencode_json``). Local profiles pin a model
         here explicitly.
      2. The global ``~/.config/opencode/opencode.json``. The cloud
         profile inherits its provider/model from there via the CI's
         ``OPENCODE_CONFIG`` secret, so this is where we discover the
         model id for cloud-default runs.

    For each candidate we take the first
    ``provider.<provider_id>.models.<model_id>`` entry — each profile
    pins exactly one model, so the first match is deterministic. The
    returned id is namespaced as ``<provider_id>/<model_id>`` to match
    the format the user passes to ``opencode run --model``.

    Returns ``None`` only if neither config declares a model.
    """
    candidates = [
        project_dir / "opencode.json",
        Path.home() / ".config" / "opencode" / "opencode.json",
    ]
    for cfg_path in candidates:
        if not cfg_path.exists():
            continue
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        model = _first_provider_model(cfg)
        if model:
            return model
    return None


def run_one_attempt(
    *,
    attempt_num: int,
    prompt: str,
    op: str,
    dtype: str,
    timeout: int,
    runtime: bool,
    runtime_backend: str,
    docker_timeout: int,
    skills_mode: str,
    profile: str,
    opencode_config_override: str | None,
    archive_dir: str | None,
    keep_project: bool,
    agent_format: str = "json",
    agents_md_path: Path | None = None,
) -> dict:
    """Run a single agent attempt end-to-end and return a result dict.

    The result is shaped to merge directly into the evidence document and
    contains everything the aggregator (compare_skills_value.py) needs:
    attempt metadata, tokens, model, verification outcomes, kernel path,
    and the live project directory (so the caller can archive it).
    """
    project = create_test_project(
        f"gen-{op}-{dtype}-a{attempt_num}",
        skills_mode=skills_mode,
        profile=profile,
        opencode_config_override=opencode_config_override,
        agents_md_path=agents_md_path,
    )
    agent_output = project / "agent-output.txt"
    agent_completed = False
    exit_code = 1
    elapsed = 0.0

    asc_path_before = _snapshot_asc_file()
    asc_path_after: str | None = asc_path_before
    asc_root_mutated = False

    try:
        env = os.environ.copy()
        env["NODE_TLS_REJECT_UNAUTHORIZED"] = "0"

        fmt_flag = f" --format {agent_format}" if agent_format else ""
        opencode_cmd = f'opencode run "{prompt}" --dir "{project}"{fmt_flag}'
        cmd = ["script", "-qc", opencode_cmd, "/dev/null"]

        print(f"  [attempt {attempt_num}] running opencode "
              f"(profile={profile}, skills_mode={skills_mode}, "
              f"timeout={timeout}s)...")
        t0 = time.monotonic()
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True,
                timeout=timeout, env=env,
            )
            exit_code = result.returncode
            agent_completed = True
        except subprocess.TimeoutExpired as exc:
            elapsed = time.monotonic() - t0
            print(f"  [attempt {attempt_num}] timed out after {timeout}s")
            partial_out = exc.stdout or b""
            partial_err = exc.stderr or b""
            if isinstance(partial_out, bytes):
                partial_out = partial_out.decode("utf-8", errors="replace")
            if isinstance(partial_err, bytes):
                partial_err = partial_err.decode("utf-8", errors="replace")
            agent_output.write_text(partial_out + partial_err)
            asc_path_after = _snapshot_asc_file()
            asc_root_mutated = (
                bool(asc_path_before) and bool(asc_path_after)
                and asc_path_before != asc_path_after
            )
            if asc_root_mutated:
                print(f"  [attempt {attempt_num}] CRITICAL: pyasc root mutated "
                      f"during agent run: {asc_path_before} -> {asc_path_after}")
            return _finalize_attempt(
                attempt_num=attempt_num, elapsed=elapsed, exit_code=124,
                project=project, op=op, dtype=dtype,
                runtime=runtime, runtime_backend=runtime_backend,
                docker_timeout=docker_timeout,
                agent_output_path=agent_output, agent_completed=False,
                keep_project=keep_project, archive_dir=archive_dir,
                skills_mode=skills_mode, profile=profile,
                asc_path_before=asc_path_before,
                asc_path_after=asc_path_after,
                asc_root_mutated=asc_root_mutated,
            )
        elapsed = time.monotonic() - t0
        agent_output.write_text((result.stdout or "") + (result.stderr or ""))
        print(f"  [attempt {attempt_num}] agent exited (code={exit_code}, "
              f"{elapsed:.1f}s)")
        if elapsed < 10:
            print(f"  [attempt {attempt_num}] WARNING: opencode exited "
                  f"suspiciously fast — possible API/config issue")
    except Exception as exc:
        elapsed = time.monotonic() - t0 if "t0" in dir() else 0.0
        print(f"  [attempt {attempt_num}] agent error: {exc}")

    asc_path_after = _snapshot_asc_file()
    asc_root_mutated = (
        bool(asc_path_before) and bool(asc_path_after)
        and asc_path_before != asc_path_after
    )
    if asc_root_mutated:
        print(f"  [attempt {attempt_num}] CRITICAL: pyasc root mutated "
              f"during agent run: {asc_path_before} -> {asc_path_after}")

    return _finalize_attempt(
        attempt_num=attempt_num, elapsed=elapsed, exit_code=exit_code,
        project=project, op=op, dtype=dtype,
        runtime=runtime, runtime_backend=runtime_backend,
        docker_timeout=docker_timeout,
        agent_output_path=agent_output, agent_completed=agent_completed,
        keep_project=keep_project, archive_dir=archive_dir,
        skills_mode=skills_mode, profile=profile,
        asc_path_before=asc_path_before,
        asc_path_after=asc_path_after,
        asc_root_mutated=asc_root_mutated,
    )


def _finalize_attempt(
    *,
    attempt_num: int,
    elapsed: float,
    exit_code: int,
    project: Path,
    op: str,
    dtype: str,
    runtime: bool,
    runtime_backend: str,
    docker_timeout: int,
    agent_output_path: Path,
    agent_completed: bool,
    keep_project: bool,
    archive_dir: str | None,
    skills_mode: str,
    profile: str,
    asc_path_before: str | None = None,
    asc_path_after: str | None = None,
    asc_root_mutated: bool = False,
) -> dict:
    """Score, verify, and clean up after one attempt.

    Split out from run_one_attempt so the timeout path and the normal
    path share identical post-processing logic.
    """
    kernel = find_kernel(project, op)
    artifacts = find_artifacts(project)
    print(f"  [attempt {attempt_num}] kernel={kernel}, artifacts={artifacts}")

    score_data = None
    static_result = "fail"
    semantic_check: dict = {"passed": False, "detail": "no kernel found",
                            "markers_found": []}
    verification: dict = {
        "mode": "static_only", "status": "fail", "shapes_verified": [],
    }

    if kernel and kernel.is_file():
        score_data = run_score(kernel, op=op, dtype=dtype)
        static_result = run_static_verify(kernel)
        semantic_check = check_op_semantics(kernel, op)
        print(f"  [attempt {attempt_num}] static={static_result}, "
              f"semantic={'pass' if semantic_check['passed'] else 'fail'}, "
              f"score={score_data.get('score', '?') if score_data else '?'}")
        verification = {
            "mode": "static_only", "status": static_result, "shapes_verified": [],
        }
        if runtime:
            platform = load_platform_from_capabilities(op, dtype)
            resolved_backend = resolve_runtime_backend(
                runtime_backend, platform=platform,
            )
            print(f"  [attempt {attempt_num}] running simulator "
                  f"(platform={platform}, backend={resolved_backend})...")
            verification = invoke_runtime_verify(
                backend=resolved_backend,
                kernel_path=kernel,
                project_dir=project,
                timeout=docker_timeout,
                platform=platform,
            )
            print(f"  [attempt {attempt_num}] runtime={verification['status']}")
    else:
        print(f"  [attempt {attempt_num}] no kernel found — generation failed")

    try:
        kernel_rel = str(kernel.relative_to(project)) if kernel else ""
    except ValueError:
        kernel_rel = str(kernel) if kernel else ""

    accepted = bool(score_data.get("accepted", False)) if score_data else False
    runtime_ok = (
        verification.get("status") == "pass"
        if runtime
        else True
    )
    overall_pass = (
        static_result == "pass"
        and accepted
        and semantic_check["passed"]
        and kernel is not None
        and runtime_ok
        and not asc_root_mutated
    )

    tokens = extract_tokens_and_model(agent_output_path)
    if tokens.get("model") is None:
        # opencode --format json does not emit the model id; fall back to
        # the model that the project's resolved opencode.json declares.
        tokens["model"] = resolve_configured_model(project)

    if archive_dir and kernel and kernel.is_file() and overall_pass:
        dtype_short = dtype.replace("float", "f")
        archive_dest = (
            Path(archive_dir)
            / f"{op}-{dtype_short}-{profile}-{skills_mode}-a{attempt_num}"
        )
        archive_dest.mkdir(parents=True, exist_ok=True)
        for item in kernel.parent.iterdir():
            if item.is_file():
                shutil.copy2(item, archive_dest / item.name)
        print(f"  [attempt {attempt_num}] archived kernel to: {archive_dest}")

    record = {
        "n": attempt_num,
        "elapsed_s": round(elapsed, 2),
        "exit": exit_code,
        "outcome": "pass" if overall_pass else "fail",
        "tokens": {k: v for k, v in tokens.items() if k != "model"},
        "model": tokens.get("model"),
        "agent_completed": agent_completed,
        "kernel_path": kernel_rel,
        "artifacts_found": artifacts,
        "static_verify": static_result,
        "semantic_check": semantic_check,
        "verification": verification,
        "score_data": score_data or {},
        "_project": project,
        "_keep_project": keep_project,
    }
    if asc_root_mutated:
        record["failure_category"] = "pyasc_root_mutated_during_run"
        record["pyasc_root_mutation"] = {
            "before": asc_path_before,
            "after": asc_path_after,
        }
    return record


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
                        help="Run simulator verification after generation")
    parser.add_argument("--runtime-backend",
                        choices=list(RUNTIME_BACKEND_CHOICES), default="auto",
                        help="Where to run simulator verify: 'host' invokes "
                             "run_and_verify.py directly using the local CANN "
                             "install (requires ASCEND_HOME_PATH and "
                             "`import asc`); 'docker' spawns a container from "
                             "the pyasc-sim image; 'auto' (default) prefers "
                             "host when available, otherwise docker.")
    parser.add_argument("--allow-dirty-pyasc", action="store_true",
                        help="Permit --runtime collection even when the "
                             "imported pyasc working tree has uncommitted "
                             "changes. Off by default so nightly evidence "
                             "always pins to a recorded SHA. Override only "
                             "for local iteration; CI must never set this.")
    parser.add_argument("--docker-timeout", type=int, default=300,
                        help="Runtime verify timeout in seconds, applies to "
                             "both docker and host backends (default: 300)")
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
    parser.add_argument("--skills-mode", choices=["on", "off"], default=None,
                        help="Whether to install skills/AGENTS.md into the "
                             "test project. When --protocol-id is given the "
                             "value is derived from the protocol table and "
                             "any conflicting --skills-mode is rejected. "
                             "(default: on when neither flag is given)")
    parser.add_argument("--model-profile", default="cloud-default",
                        help="Named opencode profile (default: cloud-default). "
                             "Templates live in docker/opencode-profiles/")
    parser.add_argument("--opencode-config", default=None,
                        help="Override the profile template with a JSON file at "
                             "this path (env vars are substituted)")
    parser.add_argument("--max-attempts", type=int, default=1,
                        help="Number of primary-prompt attempts before giving "
                             "up or falling back (default: 1)")
    parser.add_argument("--fallback-variant", default=None,
                        help="Prompt-variant to retry with after the primary "
                             "prompt exhausts --max-attempts (e.g. 'guided')")
    parser.add_argument("--agent-format", choices=["default", "json"],
                        default="json",
                        help="opencode --format mode (default: json so tokens "
                             "can be extracted)")
    parser.add_argument("--protocol-id", choices=sorted(PROTOCOL_TABLE.keys()),
                        default=None,
                        help="Phase 0 protocol id (P2/P3/P4/P6). Derives "
                             "--skills-mode, --prompt-variant, and whether "
                             "the baseline AGENTS.md is mounted; see "
                             "docs/evaluation-methodology.md \"Protocol-axis "
                             "CI mapping\". Conflicting explicit flags exit 1.")
    parser.add_argument("--agents-md-source", default=str(DEFAULT_AGENTS_MD),
                        help="Path to a baseline AGENTS.md to copy into the "
                             "test project under the P4 layout. Defaults to "
                             "docs/baseline/pyasc-fork-AGENTS.md (vendored).")
    parser.add_argument("--no-agents-md", action="store_true",
                        help="Suppress mounting the baseline AGENTS.md even "
                             "when the resolved protocol asks for it. Useful "
                             "for local ablations.")
    args = parser.parse_args()

    protocol_resolved: dict | None = None
    if args.protocol_id:
        protocol_resolved = derive_protocol(args.protocol_id)
        derived_mode = protocol_resolved["skills_mode"]
        derived_variant = protocol_resolved["prompt_variant"]
        if args.skills_mode is not None and args.skills_mode != derived_mode:
            print(
                f"ERROR: --skills-mode={args.skills_mode!r} contradicts "
                f"--protocol-id={args.protocol_id} (expected "
                f"skills_mode={derived_mode!r}). See "
                f"docs/evaluation-methodology.md \"Protocol-axis CI "
                f"mapping\".",
                file=sys.stderr,
            )
            sys.exit(1)
        if args.prompt_variant is not None and args.prompt_variant != derived_variant:
            print(
                f"ERROR: --prompt-variant={args.prompt_variant!r} "
                f"contradicts --protocol-id={args.protocol_id} (expected "
                f"prompt_variant={derived_variant!r}).",
                file=sys.stderr,
            )
            sys.exit(1)
        args.skills_mode = derived_mode
        args.prompt_variant = derived_variant
        if args.protocol_id == "P4" and args.skills_mode == "on":
            print(
                "ERROR: --protocol-id P4 with skills enabled is not allowed: "
                "it would mount both the baseline AGENTS.md and the "
                "skill-stack AGENTS.md and conflate the two.",
                file=sys.stderr,
            )
            sys.exit(1)
    else:
        if args.skills_mode is None:
            args.skills_mode = "on"

    if args.protocol_id == "P6" and not args.no_agents_md \
            and Path(args.agents_md_source) != DEFAULT_AGENTS_MD:
        # Defensive guard: P6 does not mount the baseline AGENTS.md by
        # design. If a caller explicitly pointed --agents-md-source at
        # something non-default under P6, refuse — they probably meant
        # P4.
        print(
            "ERROR: --protocol-id P6 explicitly received an --agents-md-source "
            "override. P6 does not mount the baseline AGENTS.md; the "
            "skill-stack AGENTS.md is already part of the skills mount. "
            "Use --protocol-id P4 if you want the baseline AGENTS.md.",
            file=sys.stderr,
        )
        sys.exit(1)

    agents_md_path: Path | None = None
    if protocol_resolved is not None:
        if protocol_resolved["agents_md"] and not args.no_agents_md:
            agents_md_path = Path(args.agents_md_source)

    primary_prompt = args.prompt
    if not primary_prompt:
        primary_prompt = load_prompt_from_capabilities(
            args.op, args.dtype, variant=args.prompt_variant,
        )
    if not primary_prompt:
        print(f"ERROR: No prompt provided and none found in capabilities.yaml "
              f"for {args.op}/{args.dtype}", file=sys.stderr)
        sys.exit(1)

    fallback_prompt = None
    if args.fallback_variant:
        fallback_prompt = load_prompt_from_capabilities(
            args.op, args.dtype, variant=args.fallback_variant,
        )
        if fallback_prompt is None:
            print(f"  WARN: fallback variant '{args.fallback_variant}' not "
                  f"defined in capabilities.yaml; ignoring", file=sys.stderr)

    if shutil.which("opencode") is None:
        print("SKIP: opencode CLI not found on PATH", file=sys.stderr)
        sys.exit(2)

    # Capture pyasc revision up-front so downstream JSON always has it.
    pyasc_revision = collect_pyasc_revision()
    if args.runtime:
        # The host-pyasc pin guards only make sense for the host backend.
        # The docker backend carries its own pinned pyasc inside the
        # pyasc-sim image, so a CI runner (no host ASCEND_HOME_PATH / no
        # `import asc`) legitimately resolves --runtime-backend=auto to
        # docker and MUST NOT be aborted by the host-checkout checks.
        resolved_backend = resolve_runtime_backend(
            args.runtime_backend,
            platform=load_platform_from_capabilities(args.op, args.dtype),
        )
        if resolved_backend == "host":
            eval_root = os.environ.get(
                "PYASC_EVAL_ROOT", DEFAULT_PYASC_EVAL_ROOT)
            error, warning = host_pyasc_pin_error(
                pyasc_revision,
                eval_root=eval_root,
                allow_dirty=args.allow_dirty_pyasc,
            )
            if warning:
                print(f"  WARN: {warning}", file=sys.stderr)
            if error:
                print(f"ERROR: {error}", file=sys.stderr)
                sys.exit(1)

    print(f"  Generative evidence for {args.op}/{args.dtype}")
    print(f"  Profile: {args.model_profile}, skills_mode: {args.skills_mode}")
    if args.protocol_id:
        print(
            f"  Protocol: {args.protocol_id} "
            f"({protocol_resolved['name']}) — "
            f"prompt_variant={args.prompt_variant}, "
            f"agents_md={bool(agents_md_path)}"
        )
    print(f"  Prompt: {primary_prompt[:80]}...")
    if fallback_prompt:
        print(f"  Fallback ({args.fallback_variant}): {fallback_prompt[:60]}...")
    print()

    attempts: list[dict] = []
    winning: dict | None = None

    plan: list[tuple[str, str]] = [
        (primary_prompt, args.prompt_variant or "primary")
    ] * args.max_attempts
    if fallback_prompt:
        plan.append((fallback_prompt, args.fallback_variant or "fallback"))

    for idx, (this_prompt, label) in enumerate(plan, start=1):
        print(f"  --- attempt {idx}/{len(plan)} ({label}) ---")
        rec = run_one_attempt(
            attempt_num=idx,
            prompt=this_prompt,
            op=args.op,
            dtype=args.dtype,
            timeout=args.timeout,
            runtime=args.runtime,
            runtime_backend=args.runtime_backend,
            docker_timeout=args.docker_timeout,
            skills_mode=args.skills_mode,
            profile=args.model_profile,
            opencode_config_override=args.opencode_config,
            archive_dir=args.archive_dir,
            keep_project=args.keep_project,
            agent_format=args.agent_format,
            agents_md_path=agents_md_path,
        )
        rec["prompt_label"] = label
        attempts.append(rec)
        if rec["outcome"] == "pass":
            winning = rec
            break

    best = winning or attempts[-1]
    project = best.get("_project")
    keep_project = best.get("_keep_project", False)

    static_result = best["static_verify"]
    verification = best["verification"]
    semantic_check = best["semantic_check"]
    score_data = best.get("score_data") or {}
    kernel_rel = best["kernel_path"]
    artifacts = best["artifacts_found"]
    agent_completed = best["agent_completed"]

    sum_tokens = {"input": 0, "output": 0, "cache_read": 0, "total": 0}
    elapsed_total = 0.0
    resolved_model: str | None = None
    for rec in attempts:
        for k in sum_tokens:
            sum_tokens[k] += int(rec.get("tokens", {}).get(k, 0) or 0)
        elapsed_total += float(rec.get("elapsed_s", 0.0) or 0.0)
        if rec.get("model") and resolved_model is None:
            resolved_model = rec["model"]

    attempts_serializable = []
    for rec in attempts:
        attempts_serializable.append({
            "n": rec["n"],
            "prompt_label": rec.get("prompt_label", "primary"),
            "elapsed_s": rec["elapsed_s"],
            "exit": rec["exit"],
            "outcome": rec["outcome"],
            "tokens": rec["tokens"],
            "model": rec.get("model"),
            "kernel_found": bool(rec.get("kernel_path")),
            "static_verify": rec.get("static_verify", "fail"),
            "runtime_status": rec.get("verification", {}).get("status", ""),
        })

    evidence: dict = {
        "schema_version": SCHEMA_VERSION,
        "kind": "generative",
        "operation": args.op,
        "dtype": args.dtype,
        "prompt": primary_prompt,
        "kernel_path": kernel_rel,
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "pyasc_revision": pyasc_revision,
        "agent": {
            "platform": "opencode",
            "timeout_s": args.timeout,
            "completed": agent_completed,
            "artifacts_found": artifacts,
        },
        "skills_mode": args.skills_mode,
        "model_profile": args.model_profile,
        "model": resolved_model,
        "tokens": sum_tokens,
        "elapsed_total_s": round(elapsed_total, 2),
        "attempts": attempts_serializable,
        "verification": verification,
        "semantic_check": semantic_check,
        "score": {
            "value": score_data.get("score", 0.0) if score_data else 0.0,
            "threshold": 12,
            "accepted": score_data.get("accepted", False) if score_data else False,
            "checks": score_data.get("checks", {}) if score_data else {},
            "failure_category": (
                score_data.get("failure_category") if score_data else None
            ),
            "gating_detail": (
                score_data.get("gating_detail", {}) if score_data else {}
            ),
        },
        "static_verify": static_result,
        "notes": args.notes,
    }
    if args.ci_run_url:
        evidence["ci_run_url"] = args.ci_run_url

    if protocol_resolved is not None:
        evidence["protocol"] = {
            "id": args.protocol_id,
            "name": protocol_resolved["name"],
            "prompt_variant": args.prompt_variant,
            "skills_enabled": args.skills_mode == "on",
            "allowed_context": {
                "task_prompt": True,
                "agents_md": bool(agents_md_path),
                "skills": args.skills_mode == "on",
                "golden_kernels": False,
            },
        }

    dtype_short = args.dtype.replace("float", "f")
    suffix = f"-{args.output_suffix}" if args.output_suffix else ""
    # Filename selection (Phase 0):
    #   * --protocol-id given: <op>-<dtype>-generative-<profile>-<pid_lower>.json
    #     (legacy short name is no longer used in this code path).
    #   * --protocol-id absent: today's behavior unchanged — the legacy
    #     short name for (cloud-default, on), and <profile>-<mode>.json
    #     otherwise.
    if args.protocol_id:
        out_name = (
            f"{args.op}-{dtype_short}-generative"
            f"-{args.model_profile}-{args.protocol_id.lower()}{suffix}.json"
        )
    elif args.model_profile == "cloud-default" and args.skills_mode == "on":
        out_name = f"{args.op}-{dtype_short}-generative{suffix}.json"
    else:
        out_name = (
            f"{args.op}-{dtype_short}-generative"
            f"-{args.model_profile}-{args.skills_mode}{suffix}.json"
        )
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
                "tokens_total": prev.get("tokens", {}).get("total", 0),
                "elapsed_s": prev.get("elapsed_total_s", 0),
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

    # Clean up every attempt's project directory (best already covered).
    for rec in attempts:
        rec_project = rec.get("_project")
        rec_keep = rec.get("_keep_project", False)
        if rec_project is None:
            continue
        if rec_keep:
            print(f"  Project kept at: {rec_project}")
        else:
            shutil.rmtree(rec_project, ignore_errors=True)

    runtime_ok = verification.get("status") == "pass" if args.runtime else True
    overall_pass = (
        static_result == "pass"
        and evidence["score"]["accepted"]
        and semantic_check["passed"]
        and bool(kernel_rel)
        and runtime_ok
    )
    print(f"  Overall: {'pass' if overall_pass else 'fail'}  "
          f"(attempts={len(attempts)}, "
          f"tokens={sum_tokens['total']}, "
          f"elapsed={elapsed_total:.1f}s, "
          f"profile={args.model_profile}, skills={args.skills_mode})")
    sys.exit(0 if overall_pass else 1)


if __name__ == "__main__":
    main()
