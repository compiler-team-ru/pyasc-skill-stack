#!/usr/bin/env python3
"""Sync capabilities.yaml generative_status from evidence files.

Reads each generative evidence JSON and updates the corresponding cell in
capabilities.yaml to 'confirmed' (runtime pass) or 'pending' (otherwise).
Skips cells with generative_status == 'blocked'.

Usage:
    python sync_capabilities.py [--capabilities capabilities.yaml] [--evidence-dir evidence/]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    print("ERROR: pyyaml required. pip install pyyaml", file=sys.stderr)
    sys.exit(1)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CAPS = REPO_ROOT / "capabilities.yaml"
DEFAULT_EVIDENCE_DIR = REPO_ROOT / "evidence"


def _evidence_pass(ev: dict) -> bool:
    """Replicate the overall_pass logic from collect_generative_evidence.py."""
    verification = ev.get("verification", {})
    runtime_ok = (
        verification.get("status") == "pass"
        if verification.get("mode") != "static_only"
        else True
    )
    return (
        ev.get("static_verify") == "pass"
        and ev.get("score", {}).get("accepted", False)
        and ev.get("semantic_check", {}).get("passed", False)
        and bool(ev.get("kernel_path"))
        and runtime_ok
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync capabilities.yaml from evidence.")
    parser.add_argument("--capabilities", type=Path, default=DEFAULT_CAPS)
    parser.add_argument("--evidence-dir", type=Path, default=DEFAULT_EVIDENCE_DIR)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    with open(args.capabilities) as f:
        caps = yaml.safe_load(f)

    evidence_map: dict[str, bool] = {}
    if args.evidence_dir.exists():
        # Phase 0: P6 evidence (skills on + guided) is authoritative for
        # generative_status updates; other protocols are reporting-only.
        # Read the legacy ``<op>-<dtype>-generative.json`` alias first
        # so older runs continue to drive the status, then fall back to
        # the per-protocol P6 file
        # (``<op>-<dtype>-generative-<profile>-p6.json``) when a Phase 0
        # nightly hasn't yet rewritten the legacy short name.
        for ev_path in sorted(args.evidence_dir.glob("*-generative.json")):
            try:
                with open(ev_path) as f:
                    ev = json.load(f)
                key = f"{ev.get('operation', '')}:{ev.get('dtype', '')}"
                evidence_map[key] = _evidence_pass(ev)
            except (json.JSONDecodeError, OSError):
                continue
        for ev_path in sorted(
            args.evidence_dir.glob("*-generative-*-p6.json")
        ):
            try:
                with open(ev_path) as f:
                    ev = json.load(f)
                key = f"{ev.get('operation', '')}:{ev.get('dtype', '')}"
                # Only override if we haven't already recorded the
                # legacy short-name evidence for this op/dtype — the
                # legacy file is the historical authority during the
                # rollout. Future cleanups can flip the precedence
                # once every cell has a P6 file in the evidence dir.
                evidence_map.setdefault(key, _evidence_pass(ev))
            except (json.JSONDecodeError, OSError):
                continue

    changes = 0
    for op in caps.get("operations", []):
        op_name = op["name"]
        for cell in op.get("cells", []):
            dtype = cell.get("dtype", "")
            key = f"{op_name}:{dtype}"
            current = cell.get("generative_status", "pending")
            if current == "blocked":
                continue
            if key in evidence_map:
                new_status = "confirmed" if evidence_map[key] else "pending"
                if new_status != current:
                    print(f"  {key}: {current} -> {new_status}")
                    cell["generative_status"] = new_status
                    changes += 1

    if changes == 0:
        print("capabilities.yaml already in sync with evidence.")
        return

    if args.dry_run:
        print(f"Dry run: {changes} change(s) would be made.")
        return

    with open(args.capabilities, "w") as f:
        yaml.dump(caps, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    print(f"Updated {changes} cell(s) in capabilities.yaml")


if __name__ == "__main__":
    main()
