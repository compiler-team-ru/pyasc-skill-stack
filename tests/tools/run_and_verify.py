#!/usr/bin/env python3
"""Runtime verification for pyasc kernels.

Supports three strategies:

* **jit** — Runs ``pytest_verify_kernel.py``: mocks ``Launcher.run``, no simulator.
* **simulator** — Runs ``python kernel.py -r <backend> -v <platform>`` with
  simulator ``LD_LIBRARY_PATH`` derived from ``ASCEND_HOME_PATH``.
* **auto** — Tries simulator first; on SKIP (exit 2), falls back to JIT.

Exit codes: 0 = PASS, 1 = FAIL, 2 = SKIP.

Usage::

    python run_and_verify.py <kernel.py> \\
        [--mode jit|simulator|auto] \\
        [--backend Model|NPU] \\
        [--platform Ascend910B1] \\
        [--json]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

PYTHON = sys.executable or "python3"
SCRIPT_DIR = Path(__file__).resolve().parent
JIT_VERIFY_SCRIPT = SCRIPT_DIR / "pytest_verify_kernel.py"


def _can_import(module_name: str) -> bool:
    try:
        __import__(module_name)
        return True
    except ImportError:
        return False


def _simulator_env(platform: str, base: dict[str, str]) -> dict[str, str]:
    env = dict(base)
    env.setdefault("PYASC_DUMP_PATH", "/tmp/pyasc-eval-dump")
    ascend = env.get("ASCEND_HOME_PATH") or os.environ.get("ASCEND_HOME_PATH")
    if ascend:
        sim_lib = os.path.join(ascend, "tools", "simulator", platform, "lib")
        if os.path.isdir(sim_lib):
            prev = env.get("LD_LIBRARY_PATH", "")
            env["LD_LIBRARY_PATH"] = f"{sim_lib}:{prev}" if prev else sim_lib
    return env


def _report(
    use_json: bool,
    path: str,
    status: str,
    detail: str,
    mode: str | None = None,
) -> None:
    if use_json:
        payload: dict[str, str | None] = {
            "file": path,
            "status": status,
            "detail": detail,
        }
        if mode is not None:
            payload["mode"] = mode
        print(json.dumps(payload, indent=2))
    else:
        mode_s = f" [{mode}]" if mode else ""
        print(f"  [{status}]{mode_s} {Path(path).name}: {detail}")


def _run_jit(kernel_abs: str, use_json: bool) -> tuple[int, str]:
    cmd = [PYTHON, str(JIT_VERIFY_SCRIPT), kernel_abs]
    if use_json:
        cmd.append("--json")
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(Path(kernel_abs).parent),
        )
    except subprocess.TimeoutExpired:
        return 1, "JIT verify timeout after 120s"
    except FileNotFoundError:
        return 2, f"{PYTHON} or pytest_verify_kernel.py not found"
    except Exception as exc:
        return 1, f"JIT verify error: {exc}"

    combined = (result.stdout or "") + (result.stderr or "")
    if result.returncode == 0:
        return 0, "JIT compilation verified (Launcher.run mocked)"
    if result.returncode == 2:
        return 2, combined.strip() or "pyasc (asc) not importable"
    return 1, combined.strip()[:500] or f"JIT verify exit {result.returncode}"


def _run_simulator(
    kernel_abs: str,
    backend: str,
    platform: str,
    use_json: bool,
) -> tuple[int, str]:
    cmd = [
        PYTHON,
        kernel_abs,
        "-r",
        backend,
        "-v",
        platform,
    ]
    env = _simulator_env(platform, os.environ.copy())
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
            env=env,
            cwd=str(Path(kernel_abs).parent),
        )
    except subprocess.TimeoutExpired:
        return 1, "Timeout after 120s"
    except FileNotFoundError:
        return 2, f"{PYTHON} not found on PATH"
    except Exception as exc:
        return 1, f"Execution error: {exc}"

    combined = result.stdout + result.stderr

    if result.returncode != 0:
        if (
            "Runtime library is not available" in combined
            or "LD_LIBRARY_PATH" in combined
        ):
            return 2, "CANN runtime not available (simulator libs missing)"
        return (
            1,
            f"Exit code {result.returncode}: {combined[:500]}",
        )

    passed = False
    if "PASS" in combined or (
        "allclose" in combined.lower() and "true" in combined.lower()
    ):
        passed = True
    if "FAIL" in combined and "allclose" in combined.lower():
        passed = False

    if passed:
        return 0, f"Kernel executed on {backend}/{platform}, verification passed"
    return (
        1,
        f"Kernel executed but verification unclear: {combined[:300]}",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify pyasc kernel via simulator and/or JIT mock.",
    )
    parser.add_argument("kernel_py", help="Path to kernel.py")
    parser.add_argument(
        "--mode",
        choices=("jit", "simulator", "auto"),
        default="auto",
        help="jit=mocked launch only; simulator=run kernel script; auto=simulator then JIT on SKIP",
    )
    parser.add_argument(
        "--backend",
        default="Model",
        help="Backend passed as -r (default: Model)",
    )
    parser.add_argument(
        "--platform",
        default="Ascend910B1",
        help="Platform passed as -v for simulator mode (default: Ascend910B1)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="JSON report on stdout",
    )
    args = parser.parse_args()

    kernel_path = args.kernel_py
    use_json = args.json

    if not Path(kernel_path).exists():
        _report(use_json, kernel_path, "FAIL", "File not found", args.mode)
        sys.exit(1)

    if not _can_import("asc"):
        _report(
            use_json,
            kernel_path,
            "SKIP",
            "pyasc (asc) not importable",
            args.mode,
        )
        sys.exit(2)

    kernel_abs = str(Path(kernel_path).resolve())

    if args.mode == "jit":
        code, detail = _run_jit(kernel_abs, use_json)
        status = {0: "PASS", 1: "FAIL", 2: "SKIP"}[code]
        _report(use_json, kernel_path, status, detail, "jit")
        sys.exit(code)

    if args.mode == "simulator":
        code, detail = _run_simulator(
            kernel_abs,
            args.backend,
            args.platform,
            use_json,
        )
        status = {0: "PASS", 1: "FAIL", 2: "SKIP"}[code]
        _report(use_json, kernel_path, status, detail, "simulator")
        sys.exit(code)

    # auto: simulator first, JIT fallback on SKIP only
    sim_code, sim_detail = _run_simulator(
        kernel_abs,
        args.backend,
        args.platform,
        use_json,
    )
    if sim_code == 0:
        _report(use_json, kernel_path, "PASS", sim_detail, "auto (simulator)")
        sys.exit(0)
    if sim_code != 2:
        _report(use_json, kernel_path, "FAIL", sim_detail, "auto (simulator)")
        sys.exit(1)

    jit_code, jit_detail = _run_jit(kernel_abs, use_json)
    if jit_code == 0:
        combined = f"{sim_detail}; fallback: {jit_detail}"
        _report(use_json, kernel_path, "PASS", combined, "auto (jit fallback)")
        sys.exit(0)
    if jit_code == 2:
        _report(
            use_json,
            kernel_path,
            "SKIP",
            f"Simulator: {sim_detail}; JIT: {jit_detail}",
            "auto",
        )
        sys.exit(2)

    _report(
        use_json,
        kernel_path,
        "FAIL",
        f"Simulator skipped ({sim_detail}); JIT: {jit_detail}",
        "auto (jit fallback failed)",
    )
    sys.exit(1)


if __name__ == "__main__":
    main()
