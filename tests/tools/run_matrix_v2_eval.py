#!/usr/bin/env python3
"""Orchestrator: re-run Stage 3.3 (48 cells) + Stage 3.4 (32 trials)
against the pinned pyasc-v2-eval baseline.

Used once after the Step 1/2/3a wiring of plan
``pyasc-fork_repoint_and_q1_re-run``. Sequential by default; pass
``--parallel 2`` to run two concurrent opencode invocations (probe in
the prior session confirmed two concurrent simulator runs are safe).

Invocation per cell mirrors what
``collect_generative_evidence.py --protocol-id PN`` accepts. The
orchestrator manages:

  * Stage 3.3 matrix: 12 cells x 4 protocols (P2/P3/P4/P6), single
    attempt per cell with ``--max-attempts 3``. Cells listed in
    ``capabilities.yaml`` order.

  * Stage 3.4 stability sweep: 4 boundary cells
    (abs/f16, gelu/f16, matmul/f16, rms_norm/f16) x 4 protocols x
    2 trials (r2, r3) with the same per-cell budget.

It assumes:
  * The CANN simulator env vars are already exported in the shell.
  * ``pip install -e /home/aloschilov/workspace/pyasc-v2-eval`` has
    been run; ``python -c "import asc"`` lands there.
  * ``opencode`` CLI is on PATH.

Output: writes a CSV log of (stage, op, dtype, protocol, trial,
outcome, elapsed_s, status) into ``./run-matrix-log.csv`` (timestamped
filename).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
COLLECT = REPO_ROOT / "tests" / "tools" / "collect_generative_evidence.py"
LOG_DIR = REPO_ROOT
PYTHON = os.environ.get(
    "PYASC_PYTHON",
    "/home/aloschilov/.pyenv/versions/3.11.10/bin/python3",
)
# Hard-pin asc/asc2 imports to the eval clone so an opencode skill
# step inside one of the cells cannot re-install editable from a
# dev tree (observed once on 2026-05-27: pyasc-fork was silently
# substituted partway through Stage C, causing 18 trials to abort).
PYASC_EVAL_ROOT = os.environ.get(
    "PYASC_EVAL_ROOT",
    "/home/aloschilov/workspace/pyasc-v2-eval",
)
PYASC_EVAL_PYTHON_PATH = str(Path(PYASC_EVAL_ROOT) / "python")

# 12 cells (op, dtype) from capabilities.yaml, in declared order.
STAGE_B_CELLS: list[tuple[str, str]] = [
    ("abs", "float16"),
    ("abs", "float32"),
    ("add", "float16"),
    ("reduce_sum", "float32"),
    ("reduce_sum", "float16"),
    ("gelu", "float16"),
    ("gelu", "float32"),
    ("leaky_relu", "float16"),
    ("softmax", "float16"),
    ("matmul", "float16"),
    ("rms_norm", "float16"),
    ("rms_norm", "float32"),
]

# Boundary cells for Stage 3.4 stability sweep.
STAGE_C_CELLS: list[tuple[str, str]] = [
    ("abs", "float16"),
    ("gelu", "float16"),
    ("matmul", "float16"),
    ("rms_norm", "float16"),
]

PROTOCOLS = ("P2", "P3", "P4", "P6")
STAGE_C_TRIALS = ("r2", "r3")


def run_one(
    op: str, dtype: str, protocol: str, trial: str | None,
    *, timeout: int, docker_timeout: int, max_attempts: int,
    dry_run: bool,
) -> dict:
    """Invoke collect_generative_evidence.py for one cell/protocol/trial.

    Returns a dict with keys: op, dtype, protocol, trial, exit_code,
    elapsed_s, status, evidence_file, started_at, ended_at.
    """
    started = time.time()
    started_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    cmd = [
        PYTHON, str(COLLECT),
        "--op", op,
        "--dtype", dtype,
        "--protocol-id", protocol,
        "--runtime",
        "--runtime-backend", "host",
        "--max-attempts", str(max_attempts),
        "--timeout", str(timeout),
        "--docker-timeout", str(docker_timeout),
    ]
    if trial is not None:
        cmd += ["--output-suffix", trial]
    print(
        f"  [{started_at}] >> {op}/{dtype} {protocol}"
        + (f"/{trial}" if trial else "")
        + f" (timeout={timeout}s)",
        flush=True,
    )
    exit_code = -1
    status = "unknown"
    last_lines = ""
    if not dry_run:
        env = os.environ.copy()
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            PYASC_EVAL_PYTHON_PATH + (os.pathsep + existing if existing else "")
        )
        env["PYASC_EVAL_ROOT"] = PYASC_EVAL_ROOT
        proc = subprocess.run(
            cmd,
            capture_output=True, text=True,
            timeout=timeout * (max_attempts + 1) + 600,
            env=env,
        )
        exit_code = proc.returncode
        text = proc.stdout + proc.stderr
        last_lines = "\n".join(text.splitlines()[-3:])
        if "Overall: pass" in text:
            status = "pass"
        elif "Overall: fail" in text:
            status = "fail"
        else:
            status = "error"
    elapsed = time.time() - started
    ended_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    dtype_short = dtype.replace("float", "f")
    proto_lower = protocol.lower()
    name = f"{op}-{dtype_short}-generative-cloud-default-{proto_lower}"
    if trial is not None:
        name += f"-{trial}"
    name += ".json"
    evidence_file = str(REPO_ROOT / "evidence" / name)
    print(
        f"  [{ended_at}] << {op}/{dtype} {protocol}"
        + (f"/{trial}" if trial else "")
        + f" => {status} in {elapsed:.0f}s  {last_lines[-200:]}",
        flush=True,
    )
    return {
        "op": op, "dtype": dtype, "protocol": protocol,
        "trial": trial or "",
        "exit_code": exit_code, "status": status,
        "elapsed_s": round(elapsed, 2),
        "evidence_file": evidence_file,
        "started_at": started_at, "ended_at": ended_at,
    }


def plan_jobs(stages: list[str]) -> list[dict]:
    jobs: list[dict] = []
    if "B" in stages:
        for op, dtype in STAGE_B_CELLS:
            for proto in PROTOCOLS:
                jobs.append({
                    "stage": "B", "op": op, "dtype": dtype,
                    "protocol": proto, "trial": None,
                })
    if "C" in stages:
        for op, dtype in STAGE_C_CELLS:
            for proto in PROTOCOLS:
                for trial in STAGE_C_TRIALS:
                    jobs.append({
                        "stage": "C", "op": op, "dtype": dtype,
                        "protocol": proto, "trial": trial,
                    })
    return jobs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stages", default="B,C",
        help="Comma-sep stage selection from {B,C}. Default: B,C.",
    )
    parser.add_argument(
        "--parallel", type=int, default=1,
        help="Concurrent collect_generative_evidence.py invocations. "
             "1 (default) keeps single-stream forensic clarity; "
             "2 is empirically safe with the simulator.",
    )
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=240,
                        help="Agent timeout per attempt (seconds).")
    parser.add_argument("--docker-timeout", type=int, default=180,
                        help="Runtime/simulator verify timeout (seconds).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print job list and skip execution.")
    parser.add_argument(
        "--resume-from", default=None,
        help="Path to a previous run-matrix-log-*.csv. When given, only "
             "rows whose status is in {error, fail-stale, unknown} are "
             "rerun; pass/fail rows are skipped. Use after a partial "
             "matrix run.",
    )
    args = parser.parse_args()

    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    for s in stages:
        if s not in ("B", "C"):
            print(f"unknown stage: {s!r}", file=sys.stderr)
            return 2

    if shutil.which("opencode") is None:
        print("opencode not on PATH", file=sys.stderr)
        return 2

    jobs = plan_jobs(stages)
    if args.resume_from:
        keep_keys: set[tuple[str, str, str, str]] = set()
        with open(args.resume_from) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["status"] not in ("error", "fail-stale", "unknown", ""):
                    continue
                keep_keys.add((
                    row["op"], row["dtype"], row["protocol"],
                    row["trial"] or "",
                ))
        jobs = [
            j for j in jobs
            if (j["op"], j["dtype"], j["protocol"], j["trial"] or "")
            in keep_keys
        ]
        print(f"  resume-from: kept {len(jobs)} errored/incomplete jobs",
              flush=True)
    print(f"  plan: {len(jobs)} jobs (parallel={args.parallel})",
          flush=True)
    for j in jobs:
        suffix = f"/{j['trial']}" if j["trial"] else ""
        print(
            f"    {j['stage']} {j['op']}/{j['dtype']} "
            f"{j['protocol']}{suffix}",
            flush=True,
        )

    if args.dry_run:
        return 0

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = LOG_DIR / f"run-matrix-log-{ts}.csv"
    fieldnames = [
        "stage", "op", "dtype", "protocol", "trial",
        "started_at", "ended_at", "exit_code", "status",
        "elapsed_s", "evidence_file",
    ]
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        passed = 0
        failed = 0
        errored = 0

        def execute(job: dict) -> dict:
            res = run_one(
                job["op"], job["dtype"], job["protocol"], job["trial"],
                timeout=args.timeout,
                docker_timeout=args.docker_timeout,
                max_attempts=args.max_attempts,
                dry_run=args.dry_run,
            )
            res.update({"stage": job["stage"]})
            return res

        if args.parallel <= 1:
            for job in jobs:
                row = execute(job)
                writer.writerow({k: row.get(k, "") for k in fieldnames})
                f.flush()
                if row["status"] == "pass":
                    passed += 1
                elif row["status"] == "fail":
                    failed += 1
                else:
                    errored += 1
                print(f"  progress: pass={passed} fail={failed} err={errored} "
                      f"(of {len(jobs)})", flush=True)
        else:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=args.parallel,
            ) as pool:
                for row in pool.map(execute, jobs):
                    writer.writerow({k: row.get(k, "") for k in fieldnames})
                    f.flush()
                    if row["status"] == "pass":
                        passed += 1
                    elif row["status"] == "fail":
                        failed += 1
                    else:
                        errored += 1
                    print(f"  progress: pass={passed} fail={failed} err={errored} "
                          f"(of {len(jobs)})", flush=True)
    print(f"  done. log: {log_path}", flush=True)
    print(f"  summary: pass={passed} fail={failed} error={errored}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
