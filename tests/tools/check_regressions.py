#!/usr/bin/env python3
"""Detect regressions by comparing the latest evidence run against history.

Reads all evidence/*-generative.json files. For each file that contains a
non-empty ``history`` array, checks whether the previous run was passing and
the current run is now failing. Reports regressions as [REGRESSION] lines.

Exit codes:
  0 = no regressions detected
  1 = at least one regression found
  2 = no evidence files found

Usage:
    python check_regressions.py [--evidence-dir evidence/] [--json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_EVIDENCE_DIR = REPO_ROOT / "evidence"


def _current_pass(ev: dict) -> bool:
    return (
        ev.get("static_verify") == "pass"
        and ev.get("score", {}).get("accepted", False)
        and ev.get("semantic_check", {}).get("passed", False)
        and ev.get("agent", {}).get("completed", False)
        and bool(ev.get("kernel_path"))
    )


def check_regressions(evidence_dir: Path) -> list[dict]:
    regressions: list[dict] = []

    ev_files = sorted(evidence_dir.glob("*-generative.json"))
    if not ev_files:
        return regressions

    for ev_path in ev_files:
        try:
            with open(ev_path) as f:
                ev = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        history = ev.get("history", [])
        if not history:
            continue

        prev = history[-1]
        current = _current_pass(ev)
        prev_passed = prev.get("overall_pass", False)

        if prev_passed and not current:
            regressions.append({
                "file": ev_path.name,
                "operation": ev.get("operation", "?"),
                "dtype": ev.get("dtype", "?"),
                "previous_date": prev.get("date", "?"),
                "current_date": ev.get("date", "?"),
                "prev_score": prev.get("score", 0),
                "current_score": ev.get("score", {}).get("value", 0),
            })

    return regressions


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect generative evidence regressions.")
    parser.add_argument("--evidence-dir", type=Path, default=DEFAULT_EVIDENCE_DIR)
    parser.add_argument("--json", action="store_true", help="Output JSON")
    args = parser.parse_args()

    ev_dir = args.evidence_dir
    if not ev_dir.exists():
        print("No evidence directory found", file=sys.stderr)
        sys.exit(2)

    ev_files = list(ev_dir.glob("*-generative.json"))
    if not ev_files:
        print("No generative evidence files found", file=sys.stderr)
        sys.exit(2)

    regressions = check_regressions(ev_dir)

    if args.json:
        print(json.dumps({"regressions": regressions, "count": len(regressions)}, indent=2))
    else:
        total = len(ev_files)
        print(f"Checked {total} evidence files for regressions")
        if regressions:
            for r in regressions:
                print(f"  [REGRESSION] {r['operation']}/{r['dtype']}: "
                      f"was passing ({r['previous_date']}), now failing ({r['current_date']})")
        else:
            print("  No regressions detected")

    sys.exit(1 if regressions else 0)


if __name__ == "__main__":
    main()
