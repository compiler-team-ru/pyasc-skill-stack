#!/usr/bin/env python3
"""Generate an interactive HTML dashboard from capabilities.yaml (v3) and evidence/*.json.

Reads the tier-based capabilities matrix and any evidence artifacts, then
produces a self-contained index.html with embedded data, tier progress bars,
representative_of tags, prompt display, and expand/collapse evidence details.

Usage:
    python generate_dashboard.py [--output-dir _site]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[assignment]

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
CAPABILITIES_FILE = REPO_ROOT / "capabilities.yaml"
EVIDENCE_DIR = REPO_ROOT / "evidence"
SKILLS_VALUE_FILE = EVIDENCE_DIR / "skills-value-summary.json"


def _load_yaml(path: Path) -> dict:
    if yaml is not None:
        with open(path) as f:
            return yaml.safe_load(f)
    import subprocess
    result = subprocess.run(
        ["python3", "-c",
         f"import yaml,json; print(json.dumps(yaml.safe_load(open('{path}'))))"],
        capture_output=True, text=True, timeout=10,
    )
    if result.returncode == 0:
        return json.loads(result.stdout)
    sys.stderr.write("ERROR: PyYAML is required. pip install pyyaml\n")
    sys.exit(2)


def _load_evidence(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def build_data(cap: dict) -> dict:
    tiers = cap.get("tiers", {})
    operations = cap.get("operations", [])

    tier_progress: dict[str, dict] = {}
    for t_name, t_info in tiers.items():
        tier_progress[t_name] = {
            "level": t_info.get("level", 0),
            "description": t_info.get("description", ""),
            "note": t_info.get("note", ""),
            "golden_confirmed": 0,
            "generative_confirmed": 0,
            "total_cells": 0,
        }

    all_dtypes: list[str] = []
    rows: list[dict] = []

    for op in operations:
        op_name = op.get("name", "?")
        op_tier = op.get("tier", "?")
        asc2_api = op.get("asc2_api", "")
        representative_of = op.get("representative_of", [])
        note = op.get("note", "")

        for cell in op.get("cells", []):
            dtype = cell.get("dtype", "?")
            if dtype not in all_dtypes:
                all_dtypes.append(dtype)

            gs = cell.get("golden_status", "untested")
            gen_s = cell.get("generative_status", "untested")
            prompt = cell.get("prompt", "")

            if op_tier in tier_progress:
                tier_progress[op_tier]["total_cells"] += 1
                if gs == "confirmed":
                    tier_progress[op_tier]["golden_confirmed"] += 1
                if gen_s == "confirmed":
                    tier_progress[op_tier]["generative_confirmed"] += 1

            golden_ev = None
            gen_ev = None
            ge_ref = cell.get("golden_evidence")
            if ge_ref:
                golden_ev = _load_evidence(REPO_ROOT / ge_ref)
            gn_ref = cell.get("generative_evidence")
            if gn_ref:
                gen_ev = _load_evidence(REPO_ROOT / gn_ref)

            prompt_variants = cell.get("prompt_variants", {})

            row: dict = {
                "op": op_name,
                "tier": op_tier,
                "asc2_api": asc2_api,
                "dtype": dtype,
                "golden_status": gs,
                "generative_status": gen_s,
                "representative_of": representative_of,
                "note": note,
                "prompt": prompt,
                "prompt_variants": prompt_variants,
            }

            if golden_ev:
                gv_status = golden_ev.get("verification", {}).get("status", "")
                row["golden_evidence"] = {
                    "date": golden_ev.get("date", ""),
                    "score": golden_ev.get("score", {}).get("value", 0),
                    "verification_mode": golden_ev.get("verification", {}).get("mode", ""),
                    "verification_status": gv_status,
                    "runtime_verified": gv_status == "pass",
                    "shapes": golden_ev.get("verification", {}).get("shapes_verified", []),
                    "notes": golden_ev.get("notes", ""),
                    "kernel_path": golden_ev.get("kernel_path", ""),
                }

            if gen_ev:
                v_status = gen_ev.get("verification", {}).get("status", "")
                v_detail = gen_ev.get("verification", {}).get("detail", "")
                failure_reason = ""
                if v_status not in ("pass", "skip", "") and v_detail:
                    for line in v_detail.strip().splitlines():
                        line = line.strip()
                        if line and not line.startswith("Traceback") and not line.startswith("File "):
                            failure_reason = line[:200]
                            break
                    if not failure_reason:
                        failure_reason = v_detail.strip()[-200:]

                history = gen_ev.get("history", [])
                trend = ""
                verification = gen_ev.get("verification", {})
                runtime_ok = (
                    verification.get("status") == "pass"
                    if verification.get("mode") != "static_only"
                    else True
                )
                curr_pass = (
                    gen_ev.get("static_verify") == "pass"
                    and gen_ev.get("score", {}).get("accepted", False)
                    and gen_ev.get("semantic_check", {}).get("passed", False)
                    and bool(gen_ev.get("kernel_path"))
                    and runtime_ok
                )
                if history:
                    prev_pass = history[-1].get("overall_pass", False)
                    if prev_pass and not curr_pass:
                        trend = "regression"
                    elif not prev_pass and curr_pass:
                        trend = "improvement"

                row["generative_evidence"] = {
                    "date": gen_ev.get("date", ""),
                    "score": gen_ev.get("score", {}).get("value", 0),
                    "verification_mode": gen_ev.get("verification", {}).get("mode", ""),
                    "verification_status": v_status,
                    "runtime_verified": v_status == "pass",
                    "shapes": gen_ev.get("verification", {}).get("shapes_verified", []),
                    "notes": gen_ev.get("notes", ""),
                    "kernel_path": gen_ev.get("kernel_path", ""),
                    "agent_platform": gen_ev.get("agent", {}).get("platform", ""),
                    "agent_completed": gen_ev.get("agent", {}).get("completed", False),
                    "artifacts": gen_ev.get("agent", {}).get("artifacts_found", []),
                    "semantic_check": gen_ev.get("semantic_check", {}),
                    "history": history,
                    "ci_run_url": gen_ev.get("ci_run_url", ""),
                    "failure_reason": failure_reason,
                    "trend": trend,
                }

            rows.append(row)

    rt_pass = rt_static = rt_skip = rt_none = 0
    for row in rows:
        ev = row.get("generative_evidence")
        if not ev:
            rt_none += 1
        elif ev.get("runtime_verified"):
            rt_pass += 1
        elif ev.get("verification_status") == "skip":
            rt_skip += 1
        else:
            rt_static += 1

    skills_value = _load_evidence(SKILLS_VALUE_FILE)

    return {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "tiers": tier_progress,
        "dtypes": all_dtypes,
        "total_cells": len(rows),
        "rows": rows,
        "runtime_summary": {
            "runtime_pass": rt_pass,
            "static_only": rt_static,
            "skipped": rt_skip,
            "no_evidence": rt_none,
            "total": len(rows),
        },
        "skills_value": skills_value,
    }


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>pyasc-skill-stack Capabilities</title>
<style>
:root {
  --bg: #ffffff;
  --bg-alt: #f6f8fa;
  --fg: #1f2328;
  --fg-muted: #656d76;
  --border: #d0d7de;
  --confirmed: #1a7f37;
  --confirmed-bg: #dafbe1;
  --golden-only: #0969da;
  --golden-only-bg: #ddf4ff;
  --pending: #9a6700;
  --pending-bg: #fff8c5;
  --claimed: #bf8700;
  --claimed-bg: #fff1cc;
  --untested: #656d76;
  --untested-bg: #eaeef2;
  --blocked: #cf222e;
  --blocked-bg: #ffebe9;
  --radius: 6px;
  --shadow: 0 1px 3px rgba(0,0,0,0.08);
  --accent: #0969da;
  --bar-bg: #eaeef2;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #0d1117;
    --bg-alt: #161b22;
    --fg: #e6edf3;
    --fg-muted: #8b949e;
    --border: #30363d;
    --confirmed: #3fb950;
    --confirmed-bg: #0d2818;
    --golden-only: #58a6ff;
    --golden-only-bg: #0c2d6b;
    --pending: #d29922;
    --pending-bg: #3b2300;
    --claimed: #d29922;
    --claimed-bg: #3b2300;
    --untested: #8b949e;
    --untested-bg: #21262d;
    --blocked: #f85149;
    --blocked-bg: #3d1214;
    --shadow: 0 1px 3px rgba(0,0,0,0.3);
    --accent: #58a6ff;
    --bar-bg: #21262d;
  }
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
  background: var(--bg);
  color: var(--fg);
  line-height: 1.5;
  padding: 24px;
  max-width: 1400px;
  margin: 0 auto;
}
h1 { font-size: 24px; margin-bottom: 4px; }
.subtitle { color: var(--fg-muted); margin-bottom: 12px; font-size: 14px; }
.runtime-banner {
  display: flex;
  flex-wrap: wrap;
  gap: 16px;
  padding: 10px 16px;
  margin-bottom: 20px;
  background: var(--bg-alt);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  font-size: 13px;
  align-items: center;
}
.runtime-banner .rb-label { font-weight: 600; color: var(--fg-muted); }
.runtime-banner .rb-item { display: flex; align-items: center; gap: 4px; }
.runtime-banner .rb-dot {
  width: 10px; height: 10px; border-radius: 50%; display: inline-block;
}
.rb-dot-pass { background: var(--confirmed); }
.rb-dot-static { background: var(--golden-only); }
.rb-dot-skip { background: var(--pending); }
.rb-dot-none { background: var(--untested); }
.skills-value-banner {
  display: none;
  padding: 16px 18px;
  margin-bottom: 20px;
  background: var(--bg-alt);
  border: 1px solid var(--border);
  border-radius: var(--radius);
}
.skills-value-banner.has-data { display: block; }
.svb-title {
  font-size: 14px;
  font-weight: 600;
  margin-bottom: 2px;
  color: var(--fg);
}
.svb-subtitle {
  font-size: 12px;
  color: var(--fg-muted);
  margin-bottom: 14px;
  max-width: 820px;
}
.svb-legend {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  margin-right: 12px;
}
.svb-legend-swatch {
  width: 10px;
  height: 10px;
  border-radius: 2px;
  display: inline-block;
}
.svb-legend-swatch.good { background: var(--confirmed); }
.svb-legend-swatch.bad { background: var(--blocked); }
.svb-legend-swatch.neutral { background: var(--fg-muted); }
.svb-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 12px;
}
.svb-card {
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 12px 14px;
  display: flex;
  flex-direction: column;
  gap: 10px;
}
.svb-card-head {
  display: flex;
  flex-direction: column;
  gap: 2px;
  padding-bottom: 8px;
  border-bottom: 1px solid var(--border);
}
.svb-card-profile {
  font-family: "SFMono-Regular", Consolas, monospace;
  font-size: 12px;
  color: var(--accent);
  font-weight: 600;
}
.svb-card-model {
  font-size: 11px;
  color: var(--fg-muted);
  font-style: italic;
}
.svb-verdict {
  font-size: 13px;
  font-weight: 600;
  display: flex;
  align-items: center;
  gap: 6px;
  margin-top: 2px;
}
.svb-verdict-arrow {
  font-size: 14px;
  line-height: 1;
}
.svb-section-label {
  font-size: 10px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.4px;
  color: var(--fg-muted);
  margin-bottom: 4px;
}
.svb-stat {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  font-size: 12px;
  margin: 3px 0;
  color: var(--fg-muted);
  gap: 8px;
}
.svb-stat .svb-val {
  color: var(--fg);
  font-weight: 500;
  font-variant-numeric: tabular-nums;
}
.svb-stat-label {
  display: inline-flex;
  align-items: center;
  gap: 3px;
}
.svb-delta-good { color: var(--confirmed); font-weight: 600; font-variant-numeric: tabular-nums; }
.svb-delta-bad { color: var(--blocked); font-weight: 600; font-variant-numeric: tabular-nums; }
.svb-delta-neutral { color: var(--fg-muted); font-variant-numeric: tabular-nums; }
.svb-pass-row {
  display: grid;
  grid-template-columns: 28px 1fr auto;
  gap: 6px;
  align-items: center;
  font-size: 11px;
  margin: 3px 0;
  color: var(--fg-muted);
  font-variant-numeric: tabular-nums;
}
.svb-pass-side { font-weight: 600; color: var(--fg-muted); }
.svb-validity-pills {
  display: inline-flex;
  gap: 4px;
  margin-left: 6px;
  vertical-align: middle;
}
.svb-validity-pill {
  display: inline-flex;
  align-items: center;
  font-size: 10px;
  padding: 1px 6px;
  border-radius: 8px;
  background: var(--bg-alt);
  color: var(--fg-muted);
  border: 1px solid var(--border);
  font-variant-numeric: tabular-nums;
  white-space: nowrap;
  position: relative;
}
.svb-validity-pill.infra { color: var(--blocked); border-color: var(--blocked); }
.svb-validity-pill.incomplete { color: var(--pending); border-color: var(--pending); }
.svb-validity-pill .svb-tip { display: none; }
.svb-validity-pill:hover .svb-tip,
.svb-validity-pill:focus-visible .svb-tip { display: block; }
.svb-fchips {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 4px;
  margin-top: 4px;
  font-size: 11px;
  color: var(--fg-muted);
}
.svb-fchips-label {
  font-weight: 600;
  color: var(--fg-muted);
  text-transform: uppercase;
  font-size: 10px;
  letter-spacing: 0.3px;
  margin-right: 2px;
}
.svb-fchips-sep {
  display: inline-block;
  width: 1px;
  height: 12px;
  background: var(--border);
  margin: 0 4px;
}
.svb-fchip {
  display: inline-flex;
  align-items: center;
  font-size: 10px;
  padding: 1px 6px;
  border-radius: 8px;
  background: var(--bg-alt);
  color: var(--fg-muted);
  border: 1px solid var(--border);
  font-variant-numeric: tabular-nums;
  white-space: nowrap;
  position: relative;
}
.svb-fchip.warn { color: var(--pending); border-color: var(--pending); }
.svb-fchip.danger { color: var(--blocked); border-color: var(--blocked); }
.svb-fchip .svb-tip { display: none; }
.svb-fchip:hover .svb-tip,
.svb-fchip:focus-visible .svb-tip { display: block; }
.svb-ci-hint {
  font-size: 11px;
  font-weight: 400;
  color: var(--fg-muted);
  font-variant-numeric: tabular-nums;
}
.svb-baseline-note {
  font-size: 12px;
  line-height: 1.45;
  color: var(--fg-muted);
  background: var(--bg-alt);
  border-left: 3px solid var(--pending);
  border-radius: 4px;
  padding: 8px 10px;
  margin-top: 4px;
}
.svb-baseline-note strong { color: var(--fg); font-weight: 600; }
.svb-pass-bar {
  height: 8px;
  background: var(--bar-bg);
  border-radius: 4px;
  overflow: hidden;
  position: relative;
}
.svb-pass-fill {
  height: 100%;
  border-radius: 4px;
  transition: width 0.3s;
}
.svb-pass-fill.on { background: var(--confirmed); }
.svb-pass-fill.off { background: var(--blocked); }
.svb-help {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 14px;
  height: 14px;
  border-radius: 50%;
  border: 1px solid var(--border);
  background: var(--bg-alt);
  color: var(--fg-muted);
  font-size: 9px;
  font-weight: 700;
  cursor: help;
  position: relative;
  user-select: none;
}
.svb-help::before { content: "i"; line-height: 1; font-family: Georgia, "Times New Roman", serif; font-style: italic; }
.svb-help:hover { color: var(--fg); border-color: var(--fg-muted); }
.svb-help:hover .svb-tip,
.svb-help:focus-visible .svb-tip { display: block; }
.svb-tip {
  display: none;
  position: absolute;
  bottom: calc(100% + 8px);
  right: -8px;
  background: var(--fg);
  color: var(--bg);
  padding: 8px 10px;
  border-radius: var(--radius);
  font-size: 11px;
  font-weight: 400;
  line-height: 1.45;
  width: 260px;
  max-width: calc(100vw - 32px);
  text-align: left;
  box-shadow: 0 4px 12px rgba(0,0,0,0.15);
  z-index: 100;
  white-space: normal;
  pointer-events: none;
}
.svb-tip::after {
  content: "";
  position: absolute;
  top: 100%;
  right: 12px;
  border: 5px solid transparent;
  border-top-color: var(--fg);
}
.tier-cards {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: 12px;
  margin-bottom: 24px;
}
.tier-card {
  background: var(--bg-alt);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 14px 16px;
  box-shadow: var(--shadow);
}
.tier-card .tier-title {
  font-weight: 600;
  font-size: 14px;
  margin-bottom: 2px;
  display: flex;
  align-items: center;
  gap: 8px;
}
.tier-card .tier-level {
  display: inline-block;
  background: var(--accent);
  color: #fff;
  font-size: 11px;
  font-weight: 700;
  padding: 1px 6px;
  border-radius: 10px;
}
.tier-card .tier-desc { font-size: 12px; color: var(--fg-muted); margin-bottom: 8px; }
.progress-row {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 4px;
  font-size: 12px;
}
.progress-row .prog-label { min-width: 80px; color: var(--fg-muted); }
.progress-bar {
  flex: 1;
  height: 8px;
  background: var(--bar-bg);
  border-radius: 4px;
  overflow: hidden;
}
.progress-bar .fill {
  height: 100%;
  border-radius: 4px;
  transition: width 0.3s;
}
.progress-bar .fill-golden { background: var(--golden-only); }
.progress-bar .fill-gen { background: var(--confirmed); }
.progress-row .prog-count { min-width: 30px; text-align: right; font-weight: 600; }
.controls {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
  margin-bottom: 16px;
  align-items: center;
}
.controls label { font-size: 13px; color: var(--fg-muted); }
.controls select {
  padding: 4px 8px;
  border: 1px solid var(--border);
  border-radius: var(--radius);
  background: var(--bg);
  color: var(--fg);
  font-size: 13px;
}
table {
  width: 100%;
  border-collapse: collapse;
  font-size: 14px;
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  overflow: hidden;
}
thead th {
  background: var(--bg-alt);
  border-bottom: 2px solid var(--border);
  padding: 8px 12px;
  text-align: left;
  font-weight: 600;
  font-size: 12px;
  text-transform: uppercase;
  letter-spacing: 0.3px;
  color: var(--fg-muted);
  cursor: pointer;
  user-select: none;
  white-space: nowrap;
}
thead th:hover { color: var(--fg); }
thead th .sort-arrow { font-size: 10px; margin-left: 4px; }
tbody tr { border-bottom: 1px solid var(--border); }
tbody tr:hover { background: var(--bg-alt); }
tbody td { padding: 6px 12px; vertical-align: top; }
td.op-name { font-weight: 600; }
td.api-col { font-family: "SFMono-Regular", Consolas, monospace; font-size: 13px; color: var(--fg-muted); }
.badge {
  display: inline-block;
  padding: 2px 8px;
  border-radius: 12px;
  font-size: 12px;
  font-weight: 500;
  cursor: default;
  white-space: nowrap;
}
.badge-confirmed { background: var(--confirmed-bg); color: var(--confirmed); }
.badge-golden_only { background: var(--golden-only-bg); color: var(--golden-only); }
.badge-pending { background: var(--pending-bg); color: var(--pending); }
.badge-claimed { background: var(--claimed-bg); color: var(--claimed); }
.badge-untested { background: var(--untested-bg); color: var(--untested); }
.badge-blocked { background: var(--blocked-bg); color: var(--blocked); }
.badge.clickable { cursor: pointer; text-decoration: underline; text-decoration-style: dotted; }
.rep-tags {
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
  margin-top: 4px;
}
.rep-tag {
  display: inline-block;
  font-size: 11px;
  padding: 1px 6px;
  border-radius: 10px;
  background: var(--bg-alt);
  border: 1px solid var(--border);
  color: var(--fg-muted);
}
.detail-panel {
  display: none;
  background: var(--bg-alt);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 12px 16px;
  margin: 4px 0 8px;
  font-size: 13px;
  line-height: 1.6;
}
.detail-panel.open { display: block; }
.detail-panel dt { font-weight: 600; display: inline; }
.detail-panel dt::after { content: ": "; }
.detail-panel dd { display: inline; margin: 0; }
.detail-panel dd::after { content: ""; display: block; }
.detail-panel .prompt-text {
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 8px 10px;
  font-size: 12px;
  margin: 4px 0;
  font-style: italic;
  color: var(--fg-muted);
}
td.prompt-col {
  max-width: 280px;
  font-size: 12px;
  color: var(--fg-muted);
}
.prompt-trunc {
  display: inline;
  font-style: italic;
}
.prompt-actions {
  display: inline-flex;
  gap: 3px;
  margin-left: 4px;
  vertical-align: middle;
}
.icon-btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 22px;
  height: 22px;
  border-radius: 4px;
  border: 1px solid var(--border);
  background: var(--bg);
  cursor: pointer;
  font-size: 11px;
  color: var(--fg-muted);
  transition: background 0.15s, color 0.15s;
  padding: 0;
  line-height: 1;
}
.icon-btn:hover { background: var(--bg-alt); color: var(--fg); }
.icon-btn.copied { color: var(--confirmed); border-color: var(--confirmed); }
.try-overlay {
  display: none;
  position: fixed;
  inset: 0;
  background: rgba(0,0,0,0.5);
  z-index: 1000;
  align-items: center;
  justify-content: center;
}
.try-overlay.open { display: flex; }
.try-modal {
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 24px;
  max-width: 720px;
  width: 92%;
  box-shadow: 0 8px 30px rgba(0,0,0,0.15);
}
.try-modal h3 { font-size: 16px; margin-bottom: 12px; }
.try-modal .prompt-full {
  background: var(--bg-alt);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 10px 12px;
  font-size: 13px;
  font-style: italic;
  color: var(--fg-muted);
  margin-bottom: 16px;
  line-height: 1.6;
}
.try-modal .cmd-block {
  background: #1e1e1e;
  color: #d4d4d4;
  border-radius: var(--radius);
  padding: 12px 48px 12px 16px;
  font-family: "SFMono-Regular", Consolas, monospace;
  font-size: 13px;
  line-height: 1.5;
  overflow-x: auto;
  margin-bottom: 16px;
  position: relative;
  white-space: pre-wrap;
  word-break: break-all;
}
.try-modal .cmd-copy {
  position: absolute;
  top: 8px;
  right: 8px;
  padding: 3px 8px;
  border: 1px solid #555;
  border-radius: 4px;
  background: #2d2d2d;
  color: #ccc;
  cursor: pointer;
  font-size: 11px;
}
.try-modal .cmd-copy:hover { background: #3d3d3d; }
.try-modal .close-btn {
  padding: 6px 16px;
  border: 1px solid var(--border);
  border-radius: var(--radius);
  background: var(--bg-alt);
  color: var(--fg);
  cursor: pointer;
  font-size: 13px;
}
.try-modal .close-btn:hover { background: var(--border); }
.sparkline {
  display: inline-flex;
  gap: 2px;
  align-items: center;
  margin-left: 6px;
  vertical-align: middle;
}
.spark-dot {
  width: 7px;
  height: 7px;
  border-radius: 2px;
  display: inline-block;
}
.spark-pass { background: var(--confirmed); }
.spark-fail { background: var(--blocked); }
.spark-skip { background: var(--untested); }
.spark-regression { background: #d29922; }
.tier-header td {
  background: var(--bg-alt);
  font-weight: 600;
  font-size: 13px;
  padding: 10px 12px;
  color: var(--fg-muted);
  border-bottom: 2px solid var(--border);
}
footer {
  margin-top: 32px;
  padding-top: 16px;
  border-top: 1px solid var(--border);
  font-size: 12px;
  color: var(--fg-muted);
}
footer a { color: var(--fg-muted); }
</style>
</head>
<body>

<h1>pyasc-skill-stack Capabilities</h1>
<p class="subtitle">Auto-generated from <code>capabilities.yaml</code> (v3) and <code>evidence/*.json</code></p>

<div class="runtime-banner" id="runtime-banner"></div>

<div class="skills-value-banner" id="skills-value-banner"></div>

<div class="tier-cards" id="tier-cards"></div>

<div class="controls">
  <label>Tier:
    <select id="filter-tier">
      <option value="all">All</option>
    </select>
  </label>
  <label>Status:
    <select id="filter-status">
      <option value="all">All</option>
      <option value="confirmed">Confirmed</option>
      <option value="golden_only">Golden only</option>
      <option value="pending">Pending</option>
      <option value="claimed">Claimed</option>
      <option value="untested">Untested</option>
      <option value="blocked">Blocked</option>
    </select>
  </label>
  <label>Dimension:
    <select id="filter-dimension">
      <option value="any">Any (golden or gen)</option>
      <option value="golden">Golden only</option>
      <option value="generative">Generative only</option>
    </select>
  </label>
</div>

<table>
  <thead>
    <tr id="thead-row"></tr>
  </thead>
  <tbody id="tbody"></tbody>
</table>

<div class="try-overlay" id="try-overlay" onclick="closeTryModal(event)">
  <div class="try-modal" onclick="event.stopPropagation()">
    <h3>Try this prompt</h3>
    <div class="prompt-full" id="try-prompt-text"></div>
    <p style="font-size:13px;color:var(--fg-muted);margin-bottom:8px;">Run in your terminal:</p>
    <div class="cmd-block">
      <button class="cmd-copy" onclick="copyCmd()">Copy</button>
      <code id="try-cmd-text"></code>
    </div>
    <button class="close-btn" onclick="closeTryModal()">Close</button>
  </div>
</div>

<footer>
  Generated <span id="gen-time"></span> &mdash;
  <a href="https://github.com/aloschilov/pyasc-skill-stack">pyasc-skill-stack</a>
</footer>

<script>
const DATA = __DATA_PLACEHOLDER__;

const TIER_ORDER = Object.entries(DATA.tiers)
  .sort((a, b) => a[1].level - b[1].level)
  .map(e => e[0]);

function init() {
  document.getElementById("gen-time").textContent = DATA.generated_at;
  renderRuntimeBanner();
  renderSkillsValueBanner();
  renderTierCards();
  populateTierFilter();
  renderTable();
  document.getElementById("filter-tier").addEventListener("change", renderTable);
  document.getElementById("filter-status").addEventListener("change", renderTable);
  document.getElementById("filter-dimension").addEventListener("change", renderTable);
}

function fmtPct(v) {
  if (v == null) return "\u2014";
  return Math.round(v * 100) + "%";
}

function fmtDelta(v, unit, opts) {
  if (v == null) return "\u2014";
  unit = unit || "";
  opts = opts || {};
  if (v === 0) return "0" + unit;
  var sign = v > 0 ? "+" : (v < 0 ? "\u2212" : "");
  var abs = Math.abs(v);
  var body;
  if (opts.usd) {
    if (abs >= 1) body = "$" + abs.toFixed(2);
    else if (abs >= 0.01) body = "$" + abs.toFixed(3);
    else body = "<$0.01";
  } else if (abs >= 1000000) {
    body = (abs / 1000000).toFixed(2) + "M";
  } else if (abs >= 10000) {
    body = (abs / 1000).toFixed(1) + "k";
  } else if (abs >= 1) {
    body = abs.toFixed(1);
  } else {
    body = abs.toFixed(2);
  }
  return sign + body + unit;
}

// Convention: for cost-like metrics (tokens, cost, time), skills-on minus
// skills-off; negative = skills used less and is GOOD.
function deltaClassLowerBetter(v) {
  if (v == null || v === 0) return "svb-delta-neutral";
  return v < 0 ? "svb-delta-good" : "svb-delta-bad";
}

function passDeltaPp(on, off) {
  if (on == null || off == null) return null;
  return Math.round((on - off) * 100);
}

function svbHelp(text) {
  var esc = text
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
  return '<span class="svb-help" tabindex="0" aria-label="' + esc
    + '"><span class="svb-tip">' + esc + '</span></span>';
}

var SVB_TIPS = {
  profile: "An OpenCode model profile defined under "
    + "docker/opencode-profiles/. The profile pins one LLM (cloud or "
    + "local) and identical OpenCode settings (timeout, max attempts, "
    + "tools). For each capability cell, the same profile is run twice: "
    + "once with the pyasc skills mounted (skills-on) and once without "
    + "(skills-off). Everything else is held constant; the only "
    + "intervention is the skill modules.",
  verdict: "Headline pass-rate movement attributable to mounting the "
    + "pyasc skills into the OpenCode project. Percentage points (pp) = "
    + "clean off pass-rate minus on pass-rate, expressed as an absolute "
    + "change \u2014 not relative. Only the clean off-baseline is used "
    + "(off-legs classified as infra/config failures are excluded). "
    + "When no clean off-runs exist for a profile, only the skills-on "
    + "pass-rate is reported.",
  passRate: "Fraction of capability cells where the OpenCode harness "
    + "produced a kernel that passed static + semantic + simulator "
    + "verification. Both legs use the same OpenCode harness, model, "
    + "prompt, timeout, max attempts, and evaluator; only the skills "
    + "mount differs. Raw bars include all paired cells, including ones "
    + "later excluded as not-comparable.",
  offValidity: "Per-pair comparability classification of each skills-off "
    + "evidence file. 'ok' = harness clearly executed (resolved model + "
    + "measurable LLM tokens). 'infra-fail' = nothing landed (no model, "
    + "zero tokens, no artifact, no kernel path) \u2014 likely an "
    + "instrumentation/configuration failure rather than a real "
    + "OpenCode-without-skills baseline. 'incomplete' = partial or "
    + "contradictory evidence (e.g. model resolved but no tokens "
    + "captured). Only 'ok' off-runs feed the headline delta.",
  unresolved: "Pairs where skills-on passed but the skills-off leg was "
    + "classified as infra/config failure. These are NOT counted as "
    + "unlocked capability \u2014 the no-skills counterfactual was "
    + "never measured for them. Reported as a separate axis so the "
    + "unlocked count is not silently inflated.",
  viability: "Strict unlock count: pairs where skills-on passed AND the "
    + "skills-off leg was a valid comparable baseline (off.validity = "
    + "'ok') AND skills-off did not pass. Denominator is the number of "
    + "cells with a clean off-baseline (cells_off_ok), not the total "
    + "compared count.",
  cellsCompared: "Cells with paired on/off evidence in this nightly. "
    + "Cells missing one side (e.g., job timed out) are excluded from "
    + "all deltas above.",
  tokens: "Average per-cell change in total LLM tokens (input + output "
    + "+ cache-read) when skills are on vs off, on the same OpenCode "
    + "harness and model. Skills typically consume more tokens because "
    + "they add context and retry until verification passes; off-legs "
    + "with no measurable tokens are not a clean baseline and inflate "
    + "the gap \u2014 the value is shown for budgeting only.",
  cost: "Average per-cell change in billed cost as reported by the "
    + "LLM provider. Cache pricing can make a large positive token "
    + "delta register near $0 in dollars. $0 means parity (or both "
    + "runs cached); negative means skills cost less.",
  walltime: "Average per-cell change in OpenCode wall-clock time. "
    + "Negative means skills finished faster on average despite using "
    + "more tokens, because they reach a passing kernel in fewer "
    + "attempts.",
  totals: "Cumulative totals across all paired cells for this profile. "
    + "Useful for budgeting: shows how much the on/off legs actually "
    + "consumed in absolute terms.",
  freshness: "Date of the most recent on-leg and off-leg evidence "
    + "feeding this card. The off-leg date reflects only its evidence "
    + "files \u2014 it does not promise the leg was a valid comparable "
    + "baseline; for that, see the off-leg validity row. "
    + "'Clean baseline staleness' is the age of the OLDEST off-leg "
    + "evidence that is currently classified as comparable ('ok'), so "
    + "you can see whether the headline delta is built on fresh "
    + "measurements or on data from a previous nightly.",
  partialRun: "The most recent nightly was partial: at least one "
    + "matrix leg of nightly-gate or local-stability-gate was cancelled "
    + "or failed before writing fresh evidence. The skills-value-report "
    + "job preserved the previously-committed per-cell evidence to "
    + "avoid mixing fresh and stale rows; only the summary itself was "
    + "updated. Re-running the nightly via 'Run workflow' \u2192 "
    + "tier=nightly will restore a full measurement.",
  failureModes: "Per-leg breakdown of why cells didn\u2019t pass. "
    + "Codes are heuristically derived from evidence v3 fields (no "
    + "schema bump): F7 static-verify, F8 simulator correctness, F9 "
    + "semantic-marker check, F10 agent never wrote a kernel.py, F12 "
    + "incomplete evidence, F13 infra/config fail, F0 didn\u2019t pass "
    + "but doesn\u2019t fit any of the above. Skills-on chips count "
    + "every paired cell whose on-leg didn\u2019t pass; skills-off "
    + "chips only count cells whose off-leg validity is \u2018ok\u2019, "
    + "i.e. comparable baselines. See "
    + "docs/evaluation-methodology.md \u00a7Failure taxonomy.",
  attemptsToPass: "1-based index of the first attempt that produced "
    + "a usable kernel, averaged over cells where the leg passed. A "
    + "lower number means the agent reached a passing kernel in fewer "
    + "tries on this profile. Skipped (\u2014) when the leg never "
    + "passed a cell. Skills-off is averaged only over clean-baseline "
    + "cells (off.validity = ok). This is the intervention-efficiency "
    + "signal that survives even when pass-rates are equal.",
  passRateCi: "Wilson 95% confidence interval on the pass-rate. For "
    + "small samples like 11/12 or 0/8, the bare percentage looks more "
    + "precise than the evidence supports; the CI shows what the data "
    + "actually rules out. A wide CI (e.g. 0\u201332%) means we don\u2019t "
    + "have enough samples to claim the underlying rate is 0%.",
};

// Display labels and a danger-level for each F-code. ``warn`` = orange,
// ``danger`` = red, default = neutral. Keep the labels short so they
// fit on a single chip; the full description lives in the tooltip.
var SVB_FCODE_LABELS = {
  F0_unknown:     {label: "F0 other",         tip: "Didn\u2019t pass but doesn\u2019t fit the known taxonomy (verification.mode missing, etc.). Worth investigating manually.", cls: "warn"},
  F7_static:      {label: "F7 static",        tip: "Kernel was written but static_verify failed.", cls: "warn"},
  F8_correctness: {label: "F8 correctness",   tip: "Simulator/runtime ran the kernel and the output was numerically wrong.", cls: "warn"},
  F9_semantic:    {label: "F9 semantic",      tip: "Kernel was written but the semantic-marker check failed (e.g. expected pyasc API not present).", cls: "warn"},
  F10_no_artifact:{label: "F10 no artifact",  tip: "Agent never wrote a kernel.py file. Most common pattern on smaller local models.", cls: "danger"},
  F12_incomplete: {label: "F12 incomplete",   tip: "Partial or contradictory evidence (e.g. model resolved but no tokens captured).", cls: "warn"},
  F13_infra_fail: {label: "F13 infra-fail",   tip: "Instrumentation/configuration failure: no model, no tokens, no artifact, no kernel path.", cls: "danger"},
};

function fchipsHtml(label, counts, includeNoneTip) {
  var keys = counts ? Object.keys(counts) : [];
  keys = keys.filter(function (k) { return (counts[k] || 0) > 0; });
  if (!keys.length) return "";
  keys.sort(function (a, b) {
    // Group by danger > warn > neutral, then by name for stability.
    var da = (SVB_FCODE_LABELS[a] || {}).cls || "";
    var db = (SVB_FCODE_LABELS[b] || {}).cls || "";
    var order = {danger: 0, warn: 1, "": 2};
    if (order[da] !== order[db]) return order[da] - order[db];
    return a.localeCompare(b);
  });
  var parts = keys.map(function (k) {
    var meta = SVB_FCODE_LABELS[k] || {label: k, tip: "", cls: ""};
    var cls = "svb-fchip" + (meta.cls ? " " + meta.cls : "");
    var tipEsc = (meta.tip || "")
      .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
    return '<span class="' + cls + '" tabindex="0" aria-label="'
      + meta.label + ': ' + counts[k] + '. ' + tipEsc + '">'
      + meta.label + ' \u00b7 ' + counts[k]
      + (tipEsc ? '<span class="svb-tip">' + tipEsc + '</span>' : '')
      + '</span>';
  });
  return '<span class="svb-fchips-label">' + label + '</span>'
    + parts.join("");
}

function fmtCiPct(ci) {
  if (!ci || ci.length !== 2) return "";
  var lo = Math.round(ci[0] * 100);
  var hi = Math.round(ci[1] * 100);
  return "CI " + lo + "\u2013" + hi + "%";
}

// Render an ISO-8601 timestamp (the collector emits `YYYY-MM-DDTHH:MM:SSZ`)
// as a short relative phrase like "today", "yesterday", or "N days ago".
// Returns "—" when the input is missing or unparseable so callers can
// embed the result directly into the HTML.
function fmtAgo(iso) {
  if (!iso) return "\u2014";
  var t = Date.parse(iso);
  if (isNaN(t)) return "\u2014";
  var days = Math.floor((Date.now() - t) / 86400000);
  if (days <= 0) return "today";
  if (days === 1) return "yesterday";
  if (days < 14) return days + " days ago";
  if (days < 60) return Math.round(days / 7) + " weeks ago";
  return Math.round(days / 30) + " months ago";
}

// "YYYY-MM-DD" excerpt of an ISO date for tooltip use; the dashboard
// stays locale-agnostic so we don't reformat with toLocaleDateString.
function fmtIsoDay(iso) {
  if (!iso) return "";
  return String(iso).slice(0, 10);
}

function renderSkillsValueBanner() {
  var sv = DATA.skills_value;
  var el = document.getElementById("skills-value-banner");
  if (!sv || !sv.by_profile) return;
  var profiles = Object.entries(sv.by_profile)
    .filter(function (e) { return (e[1].cells_compared || 0) > 0; });
  if (!profiles.length) return;
  el.classList.add("has-data");

  var partialRunHtml = '';
  if (sv.partial_run) {
    var legs = (sv.legs_status && sv.legs_status.legs) || [];
    var nonOkLegs = legs.filter(function (l) {
      return l.conclusion && l.conclusion !== "success" && l.conclusion !== "skipped";
    });
    var nonOkSummary = nonOkLegs.length
      ? nonOkLegs.map(function (l) {
          return l.name + " (" + l.conclusion + ")";
        }).join(", ")
      : "see legs_status for details";
    partialRunHtml = '<div class="svb-baseline-note" '
      + 'style="margin-top:0;margin-bottom:10px;">'
      + '<strong>Partial nightly</strong> \u2014 the most recent '
      + 'nightly run did not finish every matrix leg. To avoid '
      + 'mixing fresh and stale per-cell evidence, the auto-commit '
      + 'preserved the previously-measured rows; only the summary '
      + 'shown here was updated. Unfinished legs: ' + nonOkSummary
      + '. ' + svbHelp(SVB_TIPS.partialRun) + '</div>';
  }

  var html = partialRunHtml
    + '<div class="svb-title">OpenCode skills intervention '
    + '<span style="font-weight:400;color:var(--fg-muted);">'
    + '(OpenCode skills-on vs skills-off, latest nightly)</span></div>'
    + '<div class="svb-subtitle">'
    + 'Each card shows one OpenCode profile run twice on every '
    + 'capability cell \u2014 once with the pyasc skills mounted, once '
    + 'without. Same OpenCode harness, same resolved model, same '
    + 'prompt, same timeout, same max attempts, same evaluator; the '
    + 'only intervention is the skill modules. A per-pair validity '
    + 'classifier excludes off-legs that are instrumentation/'
    + 'configuration failures rather than real OpenCode-without-skills '
    + 'baselines (details in '
    + '<a href="https://github.com/aloschilov/pyasc-skill-stack/blob/'
    + 'main/docs/evaluation-methodology.md" target="_blank" '
    + 'rel="noopener" style="color:inherit;text-decoration:underline;'
    + 'text-decoration-style:dotted;">docs/evaluation-methodology.md'
    + '</a>). Hover the small '
    + '<em style="font-family:Georgia,serif;font-style:italic;">i</em> '
    + 'icons for each metric\u2019s definition. '
    + '<span class="svb-legend"><span class="svb-legend-swatch good">'
    + '</span> better with skills</span>'
    + '<span class="svb-legend"><span class="svb-legend-swatch bad">'
    + '</span> worse with skills</span>'
    + '<span class="svb-legend"><span class="svb-legend-swatch neutral">'
    + '</span> no change</span>'
    + '</div>'
    + '<div class="svb-grid">';

  profiles.sort(function (a, b) { return a[0].localeCompare(b[0]); });
  for (var i = 0; i < profiles.length; i++) {
    var name = profiles[i][0];
    var p = profiles[i][1];
    var compared = p.cells_compared || 0;
    var passOn = p.pass_on || 0;
    var passOff = p.pass_off || 0;
    var passOnPct = compared ? Math.round(100 * passOn / compared) : 0;
    var passOffPct = compared ? Math.round(100 * passOff / compared) : 0;

    // Validity-aware counters with safe defaults so older summary
    // files (schema_version "1") render without errors.
    var cellsOffOk = p.cells_off_ok != null ? p.cells_off_ok : compared;
    var cellsOffInfra = p.cells_off_infra_fail || 0;
    var cellsOffIncomplete = p.cells_off_incomplete || 0;
    var passRateOffClean = (p.pass_rate_off_clean !== undefined)
      ? p.pass_rate_off_clean
      : p.pass_rate_off;
    var viabilityClean = (p.viability_unlocked_clean !== undefined)
      ? p.viability_unlocked_clean
      : (p.viability_unlocked_count || 0);
    var unresolvedOffInfra = p.unresolved_due_to_off_infra || 0;
    var hasCleanBaseline = passRateOffClean != null && cellsOffOk > 0;

    var pp = hasCleanBaseline
      ? passDeltaPp(p.pass_rate_on, passRateOffClean)
      : null;
    var ppLabel = pp == null
      ? ""
      : (pp > 0 ? "+" + pp : (pp < 0 ? "\u2212" + Math.abs(pp) : "0")) + " pp";
    var verdictClass = (pp == null || pp === 0)
      ? "svb-delta-neutral"
      : (pp > 0 ? "svb-delta-good" : "svb-delta-bad");
    var verdictArrow = (pp == null || pp === 0)
      ? "\u2192"
      : (pp > 0 ? "\u2191" : "\u2193");
    var modelLine = p.model
      ? '<div class="svb-card-model">' + p.model + '</div>'
      : '';

    // Wilson 95% CI hints next to the percentages — the bare
    // "0% -> 92%" reads as precise even when the sample is 11/12.
    var ciOn = fmtCiPct(p.pass_rate_on_ci);
    var ciOffClean = fmtCiPct(p.pass_rate_off_clean_ci);
    var ciOnTip = ciOn ? svbHelp(SVB_TIPS.passRateCi) : '';

    var verdictHtml;
    if (hasCleanBaseline) {
      var ciChunk = '';
      if (ciOffClean || ciOn) {
        ciChunk = ' <span class="svb-ci-hint">('
          + (ciOffClean ? ciOffClean : '\u2014')
          + ' \u2192 '
          + (ciOn ? ciOn : '\u2014')
          + ')</span>';
      }
      verdictHtml = '<div class="svb-verdict ' + verdictClass + '">'
        + '<span class="svb-verdict-arrow">' + verdictArrow + '</span>'
        + fmtPct(passRateOffClean) + ' \u2192 ' + fmtPct(p.pass_rate_on)
        + ciChunk
        + ' pass-rate (' + ppLabel + ', clean baseline) '
        + svbHelp(SVB_TIPS.verdict)
        + ciOnTip
        + '</div>';
      var nonOk = cellsOffInfra + cellsOffIncomplete;
      if (nonOk > 0) {
        verdictHtml += '<div style="font-size:11px;color:var(--fg-muted);'
          + 'margin-top:3px;">Excluding ' + nonOk + ' off-run'
          + (nonOk === 1 ? '' : 's') + ' classified as '
          + (cellsOffInfra ? cellsOffInfra + ' infra/config failure'
              + (cellsOffInfra === 1 ? '' : 's') : '')
          + (cellsOffInfra && cellsOffIncomplete ? ' and ' : '')
          + (cellsOffIncomplete ? cellsOffIncomplete + ' incomplete' : '')
          + '. Raw skills-off pass-rate = ' + fmtPct(p.pass_rate_off) + '.'
          + '</div>';
      }
    } else {
      var ciChunkNoBaseline = ciOn
        ? ' <span class="svb-ci-hint">(' + ciOn + ')</span>'
        : '';
      verdictHtml = '<div class="svb-verdict svb-delta-neutral">'
        + '<span class="svb-verdict-arrow">\u26A0</span>'
        + 'Skills-on pass-rate: ' + fmtPct(p.pass_rate_on)
        + ciChunkNoBaseline
        + '. Clean skills-off baseline unavailable '
        + svbHelp(SVB_TIPS.verdict)
        + ciOnTip
        + '</div>'
        + '<div class="svb-baseline-note">'
        + '<strong>' + (cellsOffInfra + cellsOffIncomplete) + '/'
        + compared + '</strong> skills-off run'
        + (cellsOffInfra + cellsOffIncomplete === 1 ? '' : 's')
        + ' were not a comparable baseline'
        + ((cellsOffInfra || cellsOffIncomplete)
            ? ' (' + (cellsOffInfra
                ? cellsOffInfra + ' infra/config failure'
                  + (cellsOffInfra === 1 ? '' : 's')
                : '')
              + (cellsOffInfra && cellsOffIncomplete ? ', ' : '')
              + (cellsOffIncomplete ? cellsOffIncomplete + ' incomplete' : '')
              + ')'
            : '')
        + '. No Δ pp is reported. Raw skills-off pass-rate '
        + fmtPct(p.pass_rate_off) + ' is shown as diagnostic context '
        + 'only.</div>';
    }

    // Validity pills next to the off bar, only if non-zero.
    var validityPillsHtml = '';
    if (cellsOffInfra > 0 || cellsOffIncomplete > 0) {
      validityPillsHtml = '<span class="svb-validity-pills">';
      if (cellsOffInfra > 0) {
        validityPillsHtml += '<span class="svb-validity-pill infra" '
          + 'tabindex="0" aria-label="' + cellsOffInfra
          + ' off-run(s) classified as infra/config failures">'
          + 'infra-fail ' + cellsOffInfra
          + '<span class="svb-tip">' + cellsOffInfra
          + ' off-run' + (cellsOffInfra === 1 ? '' : 's')
          + ' missing model / tokens / artifact. Treated as '
          + 'instrumentation or configuration failure, not a clean '
          + 'OpenCode-without-skills baseline.</span></span>';
      }
      if (cellsOffIncomplete > 0) {
        validityPillsHtml += '<span class="svb-validity-pill incomplete" '
          + 'tabindex="0" aria-label="' + cellsOffIncomplete
          + ' off-run(s) classified as incomplete">'
          + 'incomplete ' + cellsOffIncomplete
          + '<span class="svb-tip">' + cellsOffIncomplete
          + ' off-run' + (cellsOffIncomplete === 1 ? '' : 's')
          + ' with partial or contradictory evidence (e.g. model '
          + 'resolved but no tokens captured). Not safe to count as a '
          + 'clean baseline.</span></span>';
      }
      validityPillsHtml += '</span>';
    }

    var tokensTotal = (p.tokens_on_sum != null && p.tokens_off_sum != null)
      ? (p.tokens_on_sum - p.tokens_off_sum)
      : null;
    var costTotal = (p.cost_on_sum != null && p.cost_off_sum != null)
      ? (p.cost_on_sum - p.cost_off_sum)
      : null;
    var elapsedTotal = (p.elapsed_on_sum != null && p.elapsed_off_sum != null)
      ? (p.elapsed_on_sum - p.elapsed_off_sum)
      : null;

    html += '<div class="svb-card">'
      + '<div class="svb-card-head">'
      + '<div class="svb-card-profile">'
      + name + ' ' + svbHelp(SVB_TIPS.profile) + '</div>'
      + modelLine
      + verdictHtml
      + '</div>'

      + '<div>'
      + '<div class="svb-section-label">Stability '
      + svbHelp(SVB_TIPS.passRate) + '</div>'
      + '<div class="svb-pass-row">'
      + '<span class="svb-pass-side">on</span>'
      + '<div class="svb-pass-bar"><div class="svb-pass-fill on" '
      + 'style="width:' + passOnPct + '%"></div></div>'
      + '<span>' + passOn + '/' + compared + ' (' + passOnPct + '%)</span>'
      + '</div>'
      + '<div class="svb-pass-row">'
      + '<span class="svb-pass-side">off</span>'
      + '<div class="svb-pass-bar"><div class="svb-pass-fill off" '
      + 'style="width:' + passOffPct + '%"></div></div>'
      + '<span>' + passOff + '/' + compared + ' (' + passOffPct + '%)'
      + validityPillsHtml + '</span>'
      + '</div>'
      // Failure-mode chip row: "why didn't this card pass?" Empty
      // string is appended when both counts are zero (no failures to
      // surface) so passing profiles stay visually clean.
      + (function () {
          var onChips = fchipsHtml('skills-on',
            p.failure_mode_counts_on);
          var offChips = fchipsHtml('skills-off',
            p.failure_mode_counts_off);
          if (!onChips && !offChips) return '';
          return '<div class="svb-fchips">'
            + onChips
            + (onChips && offChips
                ? '<span class="svb-fchips-sep"></span>' : '')
            + offChips
            + ' ' + svbHelp(SVB_TIPS.failureModes)
            + '</div>';
        })()
      + '<div class="svb-stat" style="margin-top:6px;">'
      + '<span class="svb-stat-label">Off-leg validity '
      + svbHelp(SVB_TIPS.offValidity) + '</span>'
      + '<span class="svb-val">'
      + cellsOffOk + ' ok'
      + (cellsOffInfra ? ' \u00b7 ' + cellsOffInfra + ' infra' : '')
      + (cellsOffIncomplete ? ' \u00b7 ' + cellsOffIncomplete + ' inc' : '')
      + '</span></div>'
      + '<div class="svb-stat">'
      + '<span class="svb-stat-label">Viability unlocked (clean) '
      + svbHelp(SVB_TIPS.viability) + '</span>'
      + '<span class="svb-val">'
      + viabilityClean + '/' + cellsOffOk
      + '</span></div>'
      + '<div class="svb-stat">'
      + '<span class="svb-stat-label">Unresolved (off infra-failed) '
      + svbHelp(SVB_TIPS.unresolved) + '</span>'
      + '<span class="svb-val">'
      + unresolvedOffInfra + '/' + (cellsOffInfra || 0)
      + '</span></div>'
      + '<div class="svb-stat">'
      + '<span class="svb-stat-label">Cells compared '
      + svbHelp(SVB_TIPS.cellsCompared) + '</span>'
      + '<span class="svb-val">' + compared
      + '/' + (p.cells_total || compared) + '</span></div>'
      + '</div>'

      + '<div>'
      + '<div class="svb-section-label">Resources per cell (avg) '
      + svbHelp(SVB_TIPS.totals) + '</div>'
      + '<div class="svb-stat">'
      + '<span class="svb-stat-label">Tokens '
      + svbHelp(SVB_TIPS.tokens) + '</span>'
      + '<span class="' + deltaClassLowerBetter(p.tokens_delta_avg) + '">'
      + fmtDelta(p.tokens_delta_avg) + '</span></div>'
      // Cost row: suppressed when both legs report $0 (local profiles
      // never bill, and the cloud profile's cache pricing also
      // collapses to $0 in this dataset). Hiding the row when both
      // values are zero keeps the card free of dead-weight stats.
      + (((p.cost_on_sum || 0) === 0 && (p.cost_off_sum || 0) === 0)
          ? ''
          : ('<div class="svb-stat">'
              + '<span class="svb-stat-label">Cost '
              + svbHelp(SVB_TIPS.cost) + '</span>'
              + '<span class="' + deltaClassLowerBetter(p.cost_delta_avg_usd) + '">'
              + fmtDelta(p.cost_delta_avg_usd, "", {usd: true}) + '</span></div>'))
      + '<div class="svb-stat">'
      + '<span class="svb-stat-label">Wall-time '
      + svbHelp(SVB_TIPS.walltime) + '</span>'
      + '<span class="' + deltaClassLowerBetter(p.elapsed_delta_avg_s) + '">'
      + fmtDelta(p.elapsed_delta_avg_s, "s") + '</span></div>'
      // Attempts-to-pass row: intervention-efficiency signal. Hidden
      // when both sides are null (the profile never passes a cell, so
      // there's nothing to average).
      + (function () {
          var meanOn = p.attempts_to_pass_on_mean;
          var meanOffClean = p.attempts_to_pass_off_clean_mean;
          if (meanOn == null && meanOffClean == null) return '';
          var diff = (meanOn != null && meanOffClean != null)
            ? (meanOn - meanOffClean) : null;
          var diffCls = deltaClassLowerBetter(diff);
          var partsHtml = '';
          if (meanOn != null) {
            partsHtml += '<span style="color:var(--fg);">on '
              + meanOn.toFixed(1) + '</span>';
          } else {
            partsHtml += '<span>on \u2014</span>';
          }
          partsHtml += ' \u00b7 ';
          if (meanOffClean != null) {
            partsHtml += '<span style="color:var(--fg);">off '
              + meanOffClean.toFixed(1) + '</span>';
          } else {
            partsHtml += '<span>off \u2014</span>';
          }
          if (diff != null) {
            var sign = diff > 0 ? '+' : (diff < 0 ? '\u2212' : '');
            partsHtml += ' <span class="' + diffCls + '">('
              + sign + Math.abs(diff).toFixed(1) + ')</span>';
          }
          return '<div class="svb-stat">'
            + '<span class="svb-stat-label">Attempts to pass '
            + svbHelp(SVB_TIPS.attemptsToPass) + '</span>'
            + '<span class="svb-val">' + partsHtml + '</span></div>';
        })()
      + '<div class="svb-stat" style="border-top:1px dashed var(--border);'
      + 'margin-top:4px;padding-top:4px;">'
      + '<span class="svb-stat-label" style="color:var(--fg-muted);'
      + 'font-size:11px;">Totals (\u0394)</span>'
      + '<span style="font-size:11px;color:var(--fg-muted);'
      + 'font-variant-numeric:tabular-nums;">'
      + fmtDelta(tokensTotal) + ' tok \u00b7 '
      + fmtDelta(costTotal, "", {usd: true}) + ' \u00b7 '
      + fmtDelta(elapsedTotal, "s")
      + '</span></div>'
      + '</div>';

    // Freshness footer: shows the date of the most recent on-leg
    // and off-leg evidence feeding this card, plus the staleness of
    // the oldest *clean* off-baseline (so a recent infra-failed
    // off-leg doesn't make the headline look fresh when the actual
    // comparable baseline is older).
    var onAgo = fmtAgo(p.on_last_run_at);
    var offAgo = fmtAgo(p.off_last_run_at);
    var onIso = fmtIsoDay(p.on_last_run_at);
    var offIso = fmtIsoDay(p.off_last_run_at);
    var stalenessDays = p.off_max_staleness_days;
    var stalenessHtml = '';
    if (stalenessDays != null && stalenessDays > 7) {
      stalenessHtml = '<span style="color:var(--pending);"> '
        + '\u00b7 oldest clean off-baseline: ' + stalenessDays + 'd</span>';
    } else if (stalenessDays != null) {
      stalenessHtml = ' \u00b7 oldest clean off-baseline: '
        + stalenessDays + 'd';
    }
    html += '<div class="svb-card-freshness" '
      + 'style="font-size:11px;color:var(--fg-muted);'
      + 'border-top:1px solid var(--border);'
      + 'margin-top:8px;padding-top:6px;'
      + 'font-variant-numeric:tabular-nums;">'
      + 'Last measured: on '
      + (onIso ? '<span title="' + onIso + '">' + onAgo + '</span>' : onAgo)
      + ' \u00b7 off '
      + (offIso ? '<span title="' + offIso + '">' + offAgo + '</span>' : offAgo)
      + stalenessHtml + ' '
      + svbHelp(SVB_TIPS.freshness)
      + '</div>'
      + '</div>';
  }
  html += '</div>';
  el.innerHTML = html;
}

function renderRuntimeBanner() {
  const s = DATA.runtime_summary;
  if (!s) return;
  const el = document.getElementById("runtime-banner");
  el.innerHTML = '<span class="rb-label">Runtime Verification:</span>'
    + '<span class="rb-item"><span class="rb-dot rb-dot-pass"></span> Verified: ' + s.runtime_pass + '/' + s.total + '</span>'
    + '<span class="rb-item"><span class="rb-dot rb-dot-static"></span> Static only: ' + s.static_only + '/' + s.total + '</span>'
    + '<span class="rb-item"><span class="rb-dot rb-dot-skip"></span> Skipped: ' + s.skipped + '/' + s.total + '</span>'
    + '<span class="rb-item"><span class="rb-dot rb-dot-none"></span> No evidence: ' + s.no_evidence + '/' + s.total + '</span>';
}

function renderTierCards() {
  const el = document.getElementById("tier-cards");
  let html = "";
  for (const tName of TIER_ORDER) {
    const t = DATA.tiers[tName];
    const goldenPct = t.total_cells ? Math.round(100 * t.golden_confirmed / t.total_cells) : 0;
    const genPct = t.total_cells ? Math.round(100 * t.generative_confirmed / t.total_cells) : 0;
    html += `<div class="tier-card">
      <div class="tier-title"><span class="tier-level">${t.level}</span> ${tName.replace(/_/g, " ")}</div>
      <div class="tier-desc">${t.description}</div>
      <div class="progress-row">
        <span class="prog-label">Golden</span>
        <div class="progress-bar"><div class="fill fill-golden" style="width:${goldenPct}%"></div></div>
        <span class="prog-count">${t.golden_confirmed}/${t.total_cells}</span>
      </div>
      <div class="progress-row">
        <span class="prog-label">Generative</span>
        <div class="progress-bar"><div class="fill fill-gen" style="width:${genPct}%"></div></div>
        <span class="prog-count">${t.generative_confirmed}/${t.total_cells}</span>
      </div>
    </div>`;
  }
  el.innerHTML = html;
}

function populateTierFilter() {
  const sel = document.getElementById("filter-tier");
  TIER_ORDER.forEach(t => {
    const opt = document.createElement("option");
    opt.value = t;
    opt.textContent = `Tier ${DATA.tiers[t].level}: ${t.replace(/_/g, " ")}`;
    sel.appendChild(opt);
  });
}

function makeBadge(status, evidence, kind, prompt) {
  const label = status.replace(/_/g, " ");
  const hasEvidence = !!evidence;
  const hasPrompt = !!prompt;
  const isClickable = hasEvidence || hasPrompt;
  const cls = isClickable ? "badge badge-" + status + " clickable" : "badge badge-" + status;
  let dataAttr = "";
  if (hasEvidence) dataAttr += ` data-detail='${JSON.stringify(evidence).replace(/'/g, "&#39;")}'`;
  if (hasPrompt) dataAttr += ` data-prompt='${prompt.replace(/'/g, "&#39;")}'`;
  dataAttr += ` data-kind="${kind}"`;
  let rtIcon = "";
  if (hasEvidence && status === "confirmed") {
    if (evidence.runtime_verified) {
      rtIcon = ' <span title="Runtime verified on simulator" style="font-size:11px;color:var(--confirmed);">&#9745;</span>';
    } else {
      rtIcon = ' <span title="Static only (runtime skipped)" style="font-size:11px;color:var(--pending);">&#9744;</span>';
    }
  }
  let trendIcon = "";
  if (hasEvidence && evidence.trend === "regression") {
    trendIcon = ' <span title="Regression from previous run" style="font-size:13px;color:var(--blocked);">&#9660;</span>';
  } else if (hasEvidence && evidence.trend === "improvement") {
    trendIcon = ' <span title="Improved from previous run" style="font-size:13px;color:var(--confirmed);">&#9650;</span>';
  }
  return `<span class="${cls}"${dataAttr} onclick="toggleDetail(this)">${label}</span>${rtIcon}${trendIcon}`;
}

function toggleDetail(el) {
  let panel = el.parentElement.querySelector(".detail-panel");
  if (panel) {
    panel.classList.toggle("open");
    return;
  }
  const detailStr = el.getAttribute("data-detail");
  const promptStr = el.getAttribute("data-prompt");
  const kind = el.getAttribute("data-kind");
  if (!detailStr && !promptStr) return;

  let html = "<dl>";
  if (promptStr) {
    html += `<dt>Prompt</dt><dd><div class="prompt-text">${promptStr}</div></dd>`;
  }
  const rowData = DATA.rows.find(r => {
    const dp = el.getAttribute("data-prompt");
    return dp && r.prompt === dp;
  });
  if (rowData && rowData.prompt_variants && Object.keys(rowData.prompt_variants).length) {
    html += `<dt>Variants</dt><dd>`;
    for (const [vname, vprompt] of Object.entries(rowData.prompt_variants)) {
      html += `<div style="margin:4px 0;"><strong>${vname}:</strong> <em style="color:var(--fg-muted)">${vprompt}</em></div>`;
    }
    html += `</dd>`;
  }
  if (detailStr) {
    const d = JSON.parse(detailStr);
    if (d.date) html += `<dt>Date</dt><dd>${d.date}</dd>`;
    if (d.score !== undefined) html += `<dt>Score</dt><dd>${d.score}/16</dd>`;
    if (d.verification_mode) {
      const rtLabel = d.runtime_verified ? "&#9745; runtime pass" : (d.verification_status === "skip" ? "&#9744; runtime skipped" : d.verification_status);
      html += `<dt>Verification</dt><dd>${d.verification_mode} &mdash; ${rtLabel}</dd>`;
    }
    if (d.shapes && d.shapes.length) html += `<dt>Shapes</dt><dd>${JSON.stringify(d.shapes)}</dd>`;
    if (d.kernel_path) html += `<dt>Kernel</dt><dd><code>${d.kernel_path}</code></dd>`;
    if (kind === "generative") {
      if (d.agent_platform) html += `<dt>Agent</dt><dd>${d.agent_platform}${d.agent_completed ? " (completed)" : " (incomplete)"}</dd>`;
      if (d.artifacts && d.artifacts.length) html += `<dt>Artifacts</dt><dd>${d.artifacts.join(", ")}</dd>`;
      if (d.semantic_check && d.semantic_check.passed !== undefined) {
        html += `<dt>Semantic</dt><dd>${d.semantic_check.passed ? "&#9745; pass" : "&#9744; fail"} &mdash; ${d.semantic_check.detail || ""}</dd>`;
      }
      if (d.failure_reason) html += `<dt>Failure</dt><dd style="color:var(--blocked);">${d.failure_reason}</dd>`;
      if (d.trend === "regression") html += `<dt>Trend</dt><dd style="color:var(--blocked);">&#9660; Regression from previous run</dd>`;
      else if (d.trend === "improvement") html += `<dt>Trend</dt><dd style="color:var(--confirmed);">&#9650; Improved from previous run</dd>`;
      if (d.ci_run_url) html += `<dt>CI run</dt><dd><a href="${d.ci_run_url}" target="_blank" rel="noopener">View run &amp; artifacts &#8599;</a></dd>`;
      if (d.history && d.history.length) {
        const rate = d.history.filter(h => h.overall_pass).length;
        html += `<dt>History</dt><dd>${rate}/${d.history.length} recent runs passed</dd>`;
      }
    }
    if (d.notes) html += `<dt>Notes</dt><dd>${d.notes}</dd>`;
  }
  html += "</dl>";
  panel = document.createElement("div");
  panel.className = "detail-panel open";
  panel.innerHTML = html;
  el.parentElement.appendChild(panel);
}

function copyToClipboard(text, btnEl) {
  navigator.clipboard.writeText(text).then(() => {
    if (btnEl) { btnEl.classList.add("copied"); btnEl.textContent = "\u2713"; setTimeout(() => { btnEl.classList.remove("copied"); btnEl.textContent = "\ud83d\udccb"; }, 1200); }
  });
}

function showTryModal(prompt) {
  document.getElementById("try-prompt-text").textContent = prompt;
  const cmd = 'opencode run "' + prompt.replace(/"/g, '\\"') + '"';
  document.getElementById("try-cmd-text").textContent = cmd;
  document.getElementById("try-overlay").classList.add("open");
}

function closeTryModal(ev) {
  if (ev && ev.target !== document.getElementById("try-overlay")) return;
  document.getElementById("try-overlay").classList.remove("open");
}

function copyCmd() {
  const cmd = document.getElementById("try-cmd-text").textContent;
  navigator.clipboard.writeText(cmd).then(() => {
    const btn = document.querySelector(".try-modal .cmd-copy");
    btn.textContent = "Copied!";
    setTimeout(() => { btn.textContent = "Copy"; }, 1200);
  });
}

function renderSparkline(history) {
  if (!history || !history.length) return "";
  const recent = history.slice(-10);
  const dots = recent.map((h, i) => {
    let cls;
    const isRegression = i > 0 && recent[i-1].overall_pass && !h.overall_pass && !h.skipped;
    if (isRegression) {
      cls = "spark-regression";
    } else if (h.overall_pass) {
      cls = "spark-pass";
    } else if (h.skipped) {
      cls = "spark-skip";
    } else {
      cls = "spark-fail";
    }
    const label = isRegression ? "regression" : (h.overall_pass ? "pass" : "fail");
    const title = h.date + ": " + label;
    return '<span class="spark-dot ' + cls + '" title="' + title + '"></span>';
  }).join("");
  return '<span class="sparkline">' + dots + '</span>';
}

function makePromptCell(prompt) {
  if (!prompt) return '<td class="prompt-col"><span style="color:var(--fg-muted);">&mdash;</span></td>';
  const trunc = prompt.length > 55 ? prompt.substring(0, 55) + "\u2026" : prompt;
  const escaped = prompt.replace(/'/g, "&#39;").replace(/"/g, "&quot;");
  return '<td class="prompt-col">'
    + '<span class="prompt-trunc">' + trunc + '</span>'
    + '<span class="prompt-actions">'
    + '<button class="icon-btn" title="Copy prompt" onclick="event.stopPropagation();copyToClipboard(\'' + escaped + '\', this)">\ud83d\udccb</button>'
    + '<button class="icon-btn" title="Try this prompt" onclick="event.stopPropagation();showTryModal(\'' + escaped + '\')">\u25b6</button>'
    + '</span></td>';
}

let sortCol = null;
let sortAsc = true;

function sortBy(col) {
  if (sortCol === col) { sortAsc = !sortAsc; }
  else { sortCol = col; sortAsc = true; }
  renderTable();
}

function renderTable() {
  const filterTier = document.getElementById("filter-tier").value;
  const filterStatus = document.getElementById("filter-status").value;
  const filterDim = document.getElementById("filter-dimension").value;

  let rows = DATA.rows.filter(r => {
    if (filterTier !== "all" && r.tier !== filterTier) return false;
    if (filterStatus !== "all") {
      if (filterDim === "golden") return r.golden_status === filterStatus;
      if (filterDim === "generative") return r.generative_status === filterStatus;
      return r.golden_status === filterStatus || r.generative_status === filterStatus;
    }
    return true;
  });

  if (sortCol) {
    rows.sort((a, b) => {
      let va = a[sortCol] || "";
      let vb = b[sortCol] || "";
      if (typeof va === "string") va = va.toLowerCase();
      if (typeof vb === "string") vb = vb.toLowerCase();
      if (va < vb) return sortAsc ? -1 : 1;
      if (va > vb) return sortAsc ? 1 : -1;
      return 0;
    });
  }

  const cols = [
    { key: "op", label: "Operation" },
    { key: "tier", label: "Tier" },
    { key: "dtype", label: "dtype" },
    { key: "asc2_api", label: "asc2 API" },
    { key: "prompt", label: "Prompt" },
    { key: "golden_status", label: "Golden" },
    { key: "generative_status", label: "Generative" },
  ];

  const thead = document.getElementById("thead-row");
  thead.innerHTML = cols.map(c => {
    const arrow = sortCol === c.key ? (sortAsc ? " &#9650;" : " &#9660;") : "";
    return `<th onclick="sortBy('${c.key}')">${c.label}<span class="sort-arrow">${arrow}</span></th>`;
  }).join("");

  const tbody = document.getElementById("tbody");
  let html = "";
  let lastTier = null;

  for (const r of rows) {
    if (r.tier !== lastTier) {
      lastTier = r.tier;
      const t = DATA.tiers[lastTier] || {};
      html += `<tr class="tier-header"><td colspan="${cols.length}">Tier ${t.level || "?"}: ${lastTier.replace(/_/g, " ")} &mdash; ${t.description || ""}</td></tr>`;
    }
    const repTags = (r.representative_of && r.representative_of.length)
      ? `<div class="rep-tags">${r.representative_of.map(x => `<span class="rep-tag">${x}</span>`).join("")}</div>`
      : "";
    html += "<tr>";
    html += `<td class="op-name">${r.op}${repTags}</td>`;
    html += `<td>${r.tier.replace(/_/g, " ")}</td>`;
    html += `<td>${r.dtype}</td>`;
    html += `<td class="api-col">${r.asc2_api}</td>`;
    html += makePromptCell(r.prompt);
    html += `<td>${makeBadge(r.golden_status, r.golden_evidence, "golden", r.prompt)}</td>`;
    const genHistory = r.generative_evidence ? r.generative_evidence.history : null;
    html += `<td>${makeBadge(r.generative_status, r.generative_evidence, "generative", r.prompt)}${renderSparkline(genHistory)}</td>`;
    html += "</tr>";
  }

  tbody.innerHTML = html;
}

init();
</script>
</body>
</html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate capabilities dashboard HTML.")
    parser.add_argument("--output-dir", default="_site", help="Output directory (default: _site)")
    args = parser.parse_args()

    if not CAPABILITIES_FILE.exists():
        print(f"ERROR: {CAPABILITIES_FILE} not found", file=sys.stderr)
        sys.exit(1)

    cap = _load_yaml(CAPABILITIES_FILE)
    data = build_data(cap)

    data_json = json.dumps(data, indent=None, ensure_ascii=False)
    html = HTML_TEMPLATE.replace("__DATA_PLACEHOLDER__", data_json)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "index.html").write_text(html, encoding="utf-8")
    (out_dir / ".nojekyll").write_text("", encoding="utf-8")

    print(f"Dashboard written to {out_dir / 'index.html'}")
    print(f"  {data['total_cells']} cells, {len(data['rows'])} rows")

    for t_name in sorted(data["tiers"], key=lambda k: data["tiers"][k]["level"]):
        t = data["tiers"][t_name]
        print(f"  Tier {t['level']} ({t_name}): golden {t['golden_confirmed']}/{t['total_cells']}, "
              f"gen {t['generative_confirmed']}/{t['total_cells']}")


if __name__ == "__main__":
    main()
