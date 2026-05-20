#!/usr/bin/env python3
"""Detect whether the most recent nightly was partial.

Used by the ``skills-value-report`` job in
``.github/workflows/ci.yml`` to classify a CI run as "complete" or
"partial" based on the per-job conclusions of the
``nightly-gate`` and ``local-stability-gate`` matrix legs in the same
GitHub Actions run.

A run is "partial" iff at least one matrix leg of either job has a
conclusion other than ``success`` or ``skipped`` (``skipped`` is a
legitimate outcome, e.g. when ``nightly-gate`` didn't fire because the
push event short-circuited it; in that case there are no evidence
files to fold in either, so it's not a partial-run signal). Cancelled
and failed legs are partial because their fresh evidence never made
it to an artifact.

Reads the GitHub Actions ``GET /repos/{owner}/{repo}/actions/runs/{run_id}/jobs``
payload from stdin (or from a file passed via ``--jobs-file``) and
writes a structured JSON document to stdout / ``--output``:

    {
        "needs": {
            "nightly-gate": "<result>",
            "local-stability-gate": "<result>"
        },
        "partial_run": true|false,
        "legs": [
            {"name": "<job-name>", "conclusion": "<conclusion-or-status>"}
        ]
    }

The ``--github-output`` flag appends ``partial_run=true|false`` to
the file at ``$GITHUB_OUTPUT`` so subsequent workflow steps can gate
on the result.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

MATRIX_JOB_PREFIXES = ("nightly-gate", "local-stability-gate")
TERMINAL_OK = ("success", "skipped")


def classify(jobs: list[dict],
             nightly_result: str | None,
             local_result: str | None) -> dict:
    """Build the partial-run record from a list of GitHub Actions jobs."""
    legs: list[dict] = []
    partial = False
    for job in jobs:
        name = job.get("name") or ""
        if not any(name.startswith(p) for p in MATRIX_JOB_PREFIXES):
            continue
        # ``conclusion`` is set once the job ends; fall back to
        # ``status`` for the (rare) in-progress case so we don't
        # silently lose the leg.
        concl = job.get("conclusion") or job.get("status")
        legs.append({"name": name, "conclusion": concl})
        if concl not in TERMINAL_OK:
            partial = True
    return {
        "needs": {
            "nightly-gate": nightly_result,
            "local-stability-gate": local_result,
        },
        "partial_run": partial,
        "legs": legs,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--jobs-file", default="-",
        help="Path to the GH API jobs JSON (default: stdin).",
    )
    parser.add_argument(
        "--output", default="-",
        help="Path to write the classification JSON (default: stdout).",
    )
    parser.add_argument(
        "--github-output", default=None,
        help=(
            "Optional path of $GITHUB_OUTPUT to append "
            "`partial_run=true|false` to."
        ),
    )
    parser.add_argument(
        "--nightly-result", default=os.environ.get("NIGHTLY_RESULT"),
        help="Conclusion of needs.nightly-gate (passed as env or CLI).",
    )
    parser.add_argument(
        "--local-result", default=os.environ.get("LOCAL_RESULT"),
        help="Conclusion of needs.local-stability-gate.",
    )
    args = parser.parse_args()

    if args.jobs_file == "-":
        raw = sys.stdin.read()
    else:
        raw = Path(args.jobs_file).read_text()
    data = json.loads(raw)
    jobs = data.get("jobs", []) if isinstance(data, dict) else []
    record = classify(jobs, args.nightly_result, args.local_result)

    payload = json.dumps(record, indent=2)
    if args.output == "-":
        print(payload)
    else:
        Path(args.output).write_text(payload + "\n")
        # Echo to stdout too so the workflow log shows the verdict.
        print(payload)

    if args.github_output:
        with open(args.github_output, "a") as fh:
            fh.write(
                "partial_run=" + ("true" if record["partial_run"] else "false") + "\n"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
