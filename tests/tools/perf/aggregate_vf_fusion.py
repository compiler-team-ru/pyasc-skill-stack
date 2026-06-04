#!/usr/bin/env python3
"""Aggregate --cce-simd-vf-fusion A/B evidence into a dashboard-ready summary.

Companion to ``aggregate_perf.py`` for the compiler-fusion axis. It walks the
A/B records written by ``demo_vf_fusion.py`` under ``evidence/vf-fusion/*.json``
(latest record per cell) and writes ``evidence/vf-fusion-summary.json``, which
``generate_dashboard.py`` renders in the "Compiler SIMD fusion" panel.

Unlike the perf summary there is no curated baseline in ``capabilities.yaml``:
the A/B is a pure measurement (off vs on of the same generated kernel), so the
summary is empty until a run lands evidence. The dashboard panel hides itself
when the summary is absent.

Pure stdlib; no camodel required, so it runs in CI's report job and the smoke
test.

Usage:
    python aggregate_vf_fusion.py [--output evidence/vf-fusion-summary.json]
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
VF_DIR = REPO_ROOT / "evidence" / "vf-fusion"
DEFAULT_OUTPUT = REPO_ROOT / "evidence" / "vf-fusion-summary.json"

# evidence/vf-fusion/<op>-<dtype>-<YYYYMMDDTHHMMSS>.json
_REC_RE = re.compile(r"^(?P<cell>.+)-(?P<ts>\d{8}T\d{6})\.json$")

_STATUS = ("improved", "neutral", "regressed", "gen_blocked")

# Carried-forward fields straight from each A/B record (already dashboard-shaped).
_CARRY = (
    "cell", "op", "dtype", "shape", "arch", "ref_ticks",
    "ticks_off", "ticks_on", "fusion_speedup", "ratio_off", "ratio_on",
    "gate", "status", "flag",
)


def _latest() -> dict[str, tuple[str, dict]]:
    """Latest A/B record per ``cell``. Returns {cell: (timestamp, record)}."""
    latest: dict[str, tuple[str, dict]] = {}
    if not VF_DIR.is_dir():
        return latest
    for path in sorted(VF_DIR.glob("*.json")):
        m = _REC_RE.match(path.name)
        if not m:
            continue
        try:
            with open(path) as f:
                rec = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        cell = rec.get("cell")
        if not cell:
            continue
        ts = m.group("ts")
        prev = latest.get(cell)
        if prev is None or ts > prev[0]:
            latest[cell] = (ts, {"_file": path.name, **rec})
    return latest


def _norm_status(rec: dict) -> str:
    status = str(rec.get("status", "")).lower()
    return status if status in _STATUS else "gen_blocked"


def build_summary() -> dict:
    latest = _latest()
    cells: list[dict] = []
    for cell_key in sorted(latest):
        ts, rec = latest[cell_key]
        out = {k: rec.get(k) for k in _CARRY}
        out["cell"] = cell_key
        out["status"] = _norm_status(rec)
        out["flag"] = rec.get("flag", "--cce-simd-vf-fusion")
        out["evidence_file"] = f"evidence/vf-fusion/{rec['_file']}"
        out["measured_at"] = (
            f"{ts[0:4]}-{ts[4:6]}-{ts[6:8]}T{ts[9:11]}:{ts[11:13]}:{ts[13:15]}Z"
        )
        out["source"] = "measured"
        cells.append(out)

    counts = {s: 0 for s in _STATUS}
    counts["total"] = len(cells)
    for c in cells:
        counts[c["status"]] = counts.get(c["status"], 0) + 1

    # Median fusion_speedup over cells that produced a number (context for the
    # one-line verdict; reduction-heavy ops can regress, so this can be < 1).
    speedups = sorted(c["fusion_speedup"] for c in cells
                      if isinstance(c.get("fusion_speedup"), (int, float)))
    median_speedup = None
    if speedups:
        mid = len(speedups) // 2
        median_speedup = (speedups[mid] if len(speedups) % 2
                          else round((speedups[mid - 1] + speedups[mid]) / 2, 4))

    return {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "flag": "--cce-simd-vf-fusion",
        "counts": counts,
        "median_speedup": median_speedup,
        "cells": cells,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                    help="Where to write the vf-fusion summary JSON.")
    ap.add_argument("-q", "--quiet", action="store_true")
    args = ap.parse_args()

    summary = build_summary()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    if not args.quiet:
        c = summary["counts"]
        print(f"vf-fusion-summary: {c['total']} cells "
              f"(improved={c['improved']}, neutral={c['neutral']}, "
              f"regressed={c['regressed']}, gen_blocked={c['gen_blocked']}); "
              f"median speedup={summary['median_speedup']} -> {args.output}")


if __name__ == "__main__":
    main()
