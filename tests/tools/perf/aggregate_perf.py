#!/usr/bin/env python3
"""Aggregate perf-vs-AscendC evidence into a single dashboard-ready summary.

Mirrors the role of ``compare_skills_value.py`` for the perf axis: it walks the
curated ``perf_ratio_demo`` blocks in ``capabilities.yaml`` (always present, the
offline source of truth) and overlays the latest *measured* combined record per
cell from ``evidence/perf-vs-ascendc/*.json`` when one exists. The result,
``evidence/perf-summary.json``, is what ``generate_dashboard.py`` renders in the
perf panel.

Pure stdlib + PyYAML; no camodel required, so it runs in CI's report job and in
the dashboard smoke test.

Usage:
    python aggregate_perf.py [--output evidence/perf-summary.json]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
CAPABILITIES_FILE = REPO_ROOT / "capabilities.yaml"
PERF_VS_DIR = REPO_ROOT / "evidence" / "perf-vs-ascendc"
DEFAULT_OUTPUT = REPO_ROOT / "evidence" / "perf-summary.json"

# evidence/perf-vs-ascendc/<op>-<dtype>-<YYYYMMDDTHHMMSS>.json
_COMBINED_RE = re.compile(r"^(?P<cell>.+)-(?P<ts>\d{8}T\d{6})\.json$")


def _load_yaml(path: Path) -> dict:
    try:
        import yaml
        with open(path) as f:
            return yaml.safe_load(f)
    except ImportError:
        import subprocess
        result = subprocess.run(
            ["python3", "-c",
             f"import yaml,json; print(json.dumps(yaml.safe_load(open('{path}'))))"],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode == 0:
            return json.loads(result.stdout)
        sys.stderr.write("ERROR: PyYAML is required. pip install pyyaml\n")
        sys.exit(2)


def _curated_cells(cap: dict) -> dict[str, dict]:
    """Build {``op/dtype``: curated-perf-record} from capabilities.yaml."""
    out: dict[str, dict] = {}
    for op in cap.get("operations", []):
        op_name = op.get("name", "?")
        for cell in op.get("cells", []):
            prd = cell.get("perf_ratio_demo")
            if not prd:
                continue
            dtype = cell.get("dtype", "?")
            key = f"{op_name}/{dtype}"
            status = str(prd.get("status", "")).lower()
            out[key] = {
                "cell": key,
                "op": op_name,
                "dtype": dtype,
                "shape": prd.get("shape"),
                "arch": prd.get("arch"),
                "gate": prd.get("gate", 0.70),
                "ref_ticks": prd.get("ref_ticks"),
                "gen_ticks": prd.get("gen_ticks"),
                "ratio": prd.get("last_ratio"),
                "status": status if status in ("pass", "fail", "gen_blocked") else "unknown",
                "reference_source": prd.get("ref_source", ""),
                "kernel_source": prd.get("kernel_source", ""),
                "perf_miss_note": prd.get("perf_miss_note", ""),
                "comparability_note": prd.get("comparability_note", ""),
                "evidence_file": prd.get("evidence", ""),
                "source": "curated",
            }
    return out


def _latest_measured() -> dict[str, tuple[str, dict]]:
    """Latest combined record per ``cell`` from evidence/perf-vs-ascendc/.

    Returns {cell: (timestamp, record)}.
    """
    latest: dict[str, tuple[str, dict]] = {}
    if not PERF_VS_DIR.is_dir():
        return latest
    for path in sorted(PERF_VS_DIR.glob("*.json")):
        m = _COMBINED_RE.match(path.name)
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
    if status == "gen_blocked":
        return "gen_blocked"
    if rec.get("passed") is True:
        return "pass"
    if rec.get("passed") is False:
        return "fail"
    return status if status in ("pass", "fail", "gen_blocked") else "unknown"


def build_summary() -> dict:
    cap = _load_yaml(CAPABILITIES_FILE)
    cells = _curated_cells(cap)
    measured = _latest_measured()

    for cell_key, (ts, rec) in measured.items():
        base = cells.get(cell_key)
        if base is None:
            # A measured cell with no curated block: still surface it.
            base = {
                "cell": cell_key,
                "op": rec.get("ref_detail", {}).get("op", cell_key.split("/")[0]),
                "dtype": cell_key.split("/")[-1],
                "perf_miss_note": "",
                "comparability_note": "",
                "kernel_source": "",
            }
            cells[cell_key] = base
        base["shape"] = rec.get("shape", base.get("shape"))
        base["arch"] = rec.get("arch", base.get("arch"))
        base["gate"] = rec.get("gate", base.get("gate", 0.70))
        base["ref_ticks"] = rec.get("ref_ticks", base.get("ref_ticks"))
        base["gen_ticks"] = rec.get("gen_ticks", base.get("gen_ticks"))
        base["ratio"] = rec.get("ratio", base.get("ratio"))
        base["status"] = _norm_status(rec)
        ref_detail = rec.get("ref_detail") or {}
        if ref_detail.get("reference_source"):
            base["reference_source"] = ref_detail["reference_source"]
        base["evidence_file"] = f"evidence/perf-vs-ascendc/{rec['_file']}"
        base["measured_at"] = (
            f"{ts[0:4]}-{ts[4:6]}-{ts[6:8]}T{ts[9:11]}:{ts[11:13]}:{ts[13:15]}Z"
        )
        base["source"] = "measured"

    cell_list = [cells[k] for k in sorted(cells)]
    counts = {"pass": 0, "fail": 0, "gen_blocked": 0, "unknown": 0, "total": len(cell_list)}
    for c in cell_list:
        counts[c.get("status", "unknown")] = counts.get(c.get("status", "unknown"), 0) + 1

    return {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "gate": 0.70,
        "counts": counts,
        "cells": cell_list,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                    help="Where to write the perf summary JSON.")
    ap.add_argument("-q", "--quiet", action="store_true")
    args = ap.parse_args()

    summary = build_summary()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    if not args.quiet:
        c = summary["counts"]
        print(f"perf-summary: {c['pass']}/{c['total']} cells clear the "
              f"{summary['gate']} gate "
              f"(fail={c['fail']}, gen_blocked={c['gen_blocked']}) -> {args.output}")


if __name__ == "__main__":
    main()
