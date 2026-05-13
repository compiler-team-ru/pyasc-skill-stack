#!/usr/bin/env python3
"""Aggregate generative evidence across (profile, skills_mode) pairs.

Reads every ``evidence/<op>-<dtype>-generative*.json`` and groups runs by
``(operation, dtype, model_profile)``. For each group it computes the
three deltas the skills-stack value report needs:

    quality_delta    = pass(on) - pass(off)        (-1, 0, or +1)
    tokens_delta     = tokens_total(on) - tokens_total(off)
    elapsed_delta_s  = elapsed_total_s(on) - elapsed_total_s(off)
    attempts_delta   = len(attempts(on)) - len(attempts(off))
    viability_unlocked = bool(not pass(off) and pass(on))

The legacy ``<op>-<dtype>-generative.json`` file is treated as the
(cloud-default, on) cell for that op/dtype, matching the filename
convention used by collect_generative_evidence.py.

Outputs:
  * ``evidence/skills-value-summary.json`` — structured summary
  * Markdown table on stdout — consumed by ``$GITHUB_STEP_SUMMARY``

Exit codes: 0 always (this is a reporting tool, not a gate).

Usage:
    python compare_skills_value.py [--evidence-dir DIR] [--output PATH]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
EVIDENCE_DIR = REPO_ROOT / "evidence"


_FILENAME_PATTERN = re.compile(
    r"^(?P<op>[a-z_]+)-(?P<dtype>f16|f32)-generative"
    r"(?:-(?P<profile>[a-z0-9.\-]+)-(?P<mode>on|off))?"
    r"(?:-(?P<suffix>[a-zA-Z0-9.\-]+))?"
    r"\.json$"
)


def _parse_filename(name: str) -> dict | None:
    """Return ``(op, dtype, profile, mode)`` for an evidence filename.

    Returns ``None`` for non-comparison files (e.g. ``*-golden.json``).
    The legacy ``<op>-<dtype>-generative.json`` filename maps to
    ``(cloud-default, on)`` by convention.

    The pattern is split so that:
      * trailing ``-minimal`` / ``-guided`` suffixes (existing
        ``--output-suffix`` outputs) are ignored and don't show up as
        spurious comparison cells.
      * profile names include dots and dashes (``local-llama-3.1-8b``).
    """
    m = _FILENAME_PATTERN.match(name)
    if not m:
        return None
    profile = m.group("profile") or "cloud-default"
    mode = m.group("mode") or "on"
    suffix = m.group("suffix") or ""
    if suffix and suffix != "":
        # Skip prompt-variant or ad-hoc archive files — they aren't a
        # canonical comparison cell.
        return None
    dtype_long = "float16" if m.group("dtype") == "f16" else "float32"
    return {
        "op": m.group("op"),
        "dtype": dtype_long,
        "profile": profile,
        "mode": mode,
    }


def _is_overall_pass(ev: dict) -> bool:
    """Reproduce the dashboard's pass logic."""
    verification = ev.get("verification", {}) or {}
    runtime_ok = (
        verification.get("status") == "pass"
        if verification.get("mode") != "static_only"
        else True
    )
    return bool(
        ev.get("static_verify") == "pass"
        and ev.get("score", {}).get("accepted", False)
        and ev.get("semantic_check", {}).get("passed", False)
        and ev.get("kernel_path")
        and runtime_ok
    )


def _extract_metrics(ev: dict) -> dict:
    """Pull the fields we care about out of an evidence document."""
    tokens = ev.get("tokens", {}) or {}
    return {
        "pass": _is_overall_pass(ev),
        "tokens_total": int(tokens.get("total", 0) or 0),
        "tokens_input": int(tokens.get("input", 0) or 0),
        "tokens_output": int(tokens.get("output", 0) or 0),
        "tokens_cache_read": int(tokens.get("cache_read", 0) or 0),
        "cost_usd": float(tokens.get("cost_usd", 0.0) or 0.0),
        "elapsed_s": float(ev.get("elapsed_total_s", 0) or 0),
        "attempts": len(ev.get("attempts", []) or []) or 1,
        "model": ev.get("model"),
        "score": ev.get("score", {}).get("value", 0),
        "date": ev.get("date", ""),
    }


def load_evidence(evidence_dir: Path) -> dict[tuple[str, str, str, str], dict]:
    """Build a ``{(op, dtype, profile, mode): metrics}`` table."""
    table: dict[tuple[str, str, str, str], dict] = {}
    for path in sorted(evidence_dir.glob("*-generative*.json")):
        parsed = _parse_filename(path.name)
        if parsed is None:
            continue
        try:
            with open(path) as f:
                ev = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        if ev.get("kind") != "generative":
            continue
        key = (parsed["op"], parsed["dtype"], parsed["profile"], parsed["mode"])
        # Prefer the on-document fields when present (newer schema_version=3),
        # otherwise fall back to filename inference for legacy files.
        profile = ev.get("model_profile", parsed["profile"])
        mode = ev.get("skills_mode", parsed["mode"])
        actual_key = (parsed["op"], parsed["dtype"], profile, mode)
        table[actual_key] = _extract_metrics(ev)
    return table


def compute_cell_deltas(
    table: dict[tuple[str, str, str, str], dict],
) -> list[dict]:
    """Compute deltas for every (op, dtype, profile) that has both modes."""
    rows: list[dict] = []
    seen_groups: set[tuple[str, str, str]] = set()
    for (op, dtype, profile, _mode) in table:
        seen_groups.add((op, dtype, profile))

    for (op, dtype, profile) in sorted(seen_groups):
        on = table.get((op, dtype, profile, "on"))
        off = table.get((op, dtype, profile, "off"))
        row: dict = {
            "op": op,
            "dtype": dtype,
            "profile": profile,
            "on": on,
            "off": off,
        }
        if on and off:
            row["quality_delta"] = int(bool(on["pass"])) - int(bool(off["pass"]))
            row["tokens_delta"] = on["tokens_total"] - off["tokens_total"]
            row["cost_delta_usd"] = round(on["cost_usd"] - off["cost_usd"], 6)
            row["elapsed_delta_s"] = round(on["elapsed_s"] - off["elapsed_s"], 2)
            row["attempts_delta"] = on["attempts"] - off["attempts"]
            row["viability_unlocked"] = bool(on["pass"] and not off["pass"])
        else:
            row["quality_delta"] = None
            row["tokens_delta"] = None
            row["cost_delta_usd"] = None
            row["elapsed_delta_s"] = None
            row["attempts_delta"] = None
            row["viability_unlocked"] = False
        rows.append(row)
    return rows


def aggregate_by_profile(rows: list[dict]) -> dict[str, dict]:
    """Per-profile aggregate of the three skills-stack-value axes."""
    by_profile: dict[str, dict] = {}
    for row in rows:
        prof = row["profile"]
        agg = by_profile.setdefault(prof, {
            "cells_total": 0,
            "cells_compared": 0,
            "pass_on": 0,
            "pass_off": 0,
            "tokens_on_sum": 0,
            "tokens_off_sum": 0,
            "cost_on_sum": 0.0,
            "cost_off_sum": 0.0,
            "elapsed_on_sum": 0.0,
            "elapsed_off_sum": 0.0,
            "viability_unlocked_count": 0,
            "model": None,
        })
        agg["cells_total"] += 1
        if row["on"] and row["off"]:
            agg["cells_compared"] += 1
            agg["pass_on"] += int(bool(row["on"]["pass"]))
            agg["pass_off"] += int(bool(row["off"]["pass"]))
            agg["tokens_on_sum"] += row["on"]["tokens_total"]
            agg["tokens_off_sum"] += row["off"]["tokens_total"]
            agg["cost_on_sum"] += row["on"]["cost_usd"]
            agg["cost_off_sum"] += row["off"]["cost_usd"]
            agg["elapsed_on_sum"] += row["on"]["elapsed_s"]
            agg["elapsed_off_sum"] += row["off"]["elapsed_s"]
            if row["viability_unlocked"]:
                agg["viability_unlocked_count"] += 1
            if agg["model"] is None and row["on"].get("model"):
                agg["model"] = row["on"]["model"]
    for prof, agg in by_profile.items():
        n = agg["cells_compared"] or 1
        agg["pass_rate_on"] = round(agg["pass_on"] / n, 3) if agg["cells_compared"] else None
        agg["pass_rate_off"] = round(agg["pass_off"] / n, 3) if agg["cells_compared"] else None
        agg["tokens_delta_avg"] = (
            (agg["tokens_on_sum"] - agg["tokens_off_sum"]) / n
            if agg["cells_compared"] else None
        )
        agg["cost_delta_avg_usd"] = (
            round((agg["cost_on_sum"] - agg["cost_off_sum"]) / n, 6)
            if agg["cells_compared"] else None
        )
        agg["elapsed_delta_avg_s"] = (
            round((agg["elapsed_on_sum"] - agg["elapsed_off_sum"]) / n, 2)
            if agg["cells_compared"] else None
        )
        agg["cost_on_sum"] = round(agg["cost_on_sum"], 6)
        agg["cost_off_sum"] = round(agg["cost_off_sum"], 6)
        agg["elapsed_on_sum"] = round(agg["elapsed_on_sum"], 2)
        agg["elapsed_off_sum"] = round(agg["elapsed_off_sum"], 2)
    return by_profile


def _fmt_int(v) -> str:
    if v is None:
        return "—"
    return f"{int(v):+,d}" if isinstance(v, (int, float)) and v != int(v) is False else (f"{v:+,}" if isinstance(v, int) else str(v))


def _fmt_delta(v, unit: str = "") -> str:
    if v is None:
        return "—"
    sign = "+" if v > 0 else ""
    if isinstance(v, float):
        return f"{sign}{v:.1f}{unit}"
    return f"{sign}{v:,}{unit}"


def _fmt_pct(v) -> str:
    if v is None:
        return "—"
    return f"{int(round(v * 100))}%"


def render_markdown(rows: list[dict], by_profile: dict[str, dict]) -> str:
    """Markdown report (per-profile summary + per-cell detail table)."""
    lines: list[str] = []
    lines.append("# Skills-stack value report")
    lines.append("")
    lines.append(
        f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%SZ')} "
        f"from {len(rows)} comparison cell(s)."
    )
    lines.append("")
    lines.append("## Per-profile summary")
    lines.append("")
    lines.append(
        "| Profile | Cells compared | Pass-rate on | Pass-rate off | "
        "Tokens Δ (avg) | Cost Δ (avg, USD) | Elapsed Δ (avg) | "
        "Viability unlocked |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    for prof in sorted(by_profile):
        a = by_profile[prof]
        lines.append(
            f"| `{prof}` | {a['cells_compared']}/{a['cells_total']} | "
            f"{_fmt_pct(a['pass_rate_on'])} | {_fmt_pct(a['pass_rate_off'])} | "
            f"{_fmt_delta(a['tokens_delta_avg'])} | "
            f"{_fmt_delta(a['cost_delta_avg_usd'])} | "
            f"{_fmt_delta(a['elapsed_delta_avg_s'], 's')} | "
            f"{a['viability_unlocked_count']}/{a['cells_compared']} |"
        )
    lines.append("")
    lines.append("## Per-cell deltas (skills-on minus skills-off)")
    lines.append("")
    lines.append(
        "| Op | dtype | Profile | Quality Δ | Tokens Δ | Elapsed Δ | "
        "Attempts Δ | Viability unlocked |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    for row in rows:
        if not (row["on"] and row["off"]):
            continue
        lines.append(
            f"| {row['op']} | {row['dtype']} | `{row['profile']}` | "
            f"{_fmt_delta(row['quality_delta'])} | "
            f"{_fmt_delta(row['tokens_delta'])} | "
            f"{_fmt_delta(row['elapsed_delta_s'], 's')} | "
            f"{_fmt_delta(row['attempts_delta'])} | "
            f"{'yes' if row['viability_unlocked'] else 'no'} |"
        )
    incomplete = [
        r for r in rows if not (r["on"] and r["off"])
    ]
    if incomplete:
        lines.append("")
        lines.append("### Cells with missing pair (one of on/off absent)")
        lines.append("")
        for r in incomplete:
            have = []
            if r["on"]:
                have.append("on")
            if r["off"]:
                have.append("off")
            lines.append(
                f"- {r['op']}/{r['dtype']}/`{r['profile']}` — "
                f"available: {', '.join(have) or 'none'}"
            )
    return "\n".join(lines) + "\n"


def build_summary(rows: list[dict], by_profile: dict[str, dict]) -> dict:
    return {
        "schema_version": "1",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "cells": rows,
        "by_profile": by_profile,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate generative evidence into a skills-value report.",
    )
    parser.add_argument(
        "--evidence-dir", default=str(EVIDENCE_DIR),
        help=f"Directory of generative evidence files (default: {EVIDENCE_DIR})",
    )
    parser.add_argument(
        "--output", default=str(EVIDENCE_DIR / "skills-value-summary.json"),
        help="Path to write the JSON summary",
    )
    parser.add_argument(
        "--markdown", default=None,
        help="Optional path to write the markdown report (default: stdout only)",
    )
    args = parser.parse_args()

    evidence_dir = Path(args.evidence_dir)
    if not evidence_dir.exists():
        print(f"ERROR: evidence dir not found: {evidence_dir}", file=sys.stderr)
        return 0

    table = load_evidence(evidence_dir)
    rows = compute_cell_deltas(table)
    by_profile = aggregate_by_profile(rows)

    summary = build_summary(rows, by_profile)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")

    md = render_markdown(rows, by_profile)
    print(md)
    if args.markdown:
        Path(args.markdown).write_text(md)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
