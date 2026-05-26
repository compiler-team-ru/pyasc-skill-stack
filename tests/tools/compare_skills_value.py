#!/usr/bin/env python3
"""Aggregate generative evidence into an OpenCode skills-intervention report.

The CI matrix runs each (operation, dtype, model_profile) cell twice:
once with the pyasc skills mounted into the OpenCode project and once
without. Both legs use the same OpenCode harness, the same resolved
model/profile, the same prompt, the same timeout, the same max
attempts, and the same evaluator. This script reads every
``evidence/<op>-<dtype>-generative*.json``, groups them by
``(operation, dtype, model_profile)``, and reports the OpenCode
skills-on vs skills-off comparison.

For each group it computes the per-cell deltas:

    quality_delta      = pass(on) - pass(off)        (-1, 0, or +1)
    tokens_delta       = tokens_total(on) - tokens_total(off)
    elapsed_delta_s    = elapsed_total_s(on) - elapsed_total_s(off)
    attempts_delta     = len(attempts(on)) - len(attempts(off))
    viability_unlocked = bool(on.pass and not off.pass)

It also runs a per-evidence validity classifier
(``_classify_validity``) so off-legs that are instrumentation /
configuration failures (no resolved model, zero tokens, no artifact)
are not silently counted as a clean OpenCode-without-skills baseline.
The classifier feeds three validity-aware aggregates per profile:

    pass_rate_off_clean         = pass_off_clean / cells_off_ok
                                  (None when cells_off_ok == 0)
    viability_unlocked_clean    = pairs where on.pass AND
                                  off.validity == "ok" AND not off.pass
    unresolved_due_to_off_infra = pairs where on.pass AND
                                  off.validity == "infra_fail"

See ``docs/evaluation-methodology.md`` for the full contract.

The legacy ``<op>-<dtype>-generative.json`` file is treated as the
(cloud-default, on) cell for that op/dtype, matching the filename
convention used by collect_generative_evidence.py.

Outputs:
  * ``evidence/skills-value-summary.json`` — structured summary
  * Markdown report on stdout — consumed by ``$GITHUB_STEP_SUMMARY``

Exit codes: 0 always (this is a reporting tool, not a gate).

Usage:
    python compare_skills_value.py [--evidence-dir DIR] [--output PATH]
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
EVIDENCE_DIR = REPO_ROOT / "evidence"


_FILENAME_PATTERN = re.compile(
    r"^(?P<op>[a-z_]+)-(?P<dtype>f16|f32)-generative"
    r"(?:-(?P<profile>[a-z0-9.\-]+)-(?P<leg>on|off|p[0-9]+))?"
    r"(?:-(?P<suffix>[a-zA-Z0-9.\-]+))?"
    r"\.json$"
)


# Phase 0 fallback mapping. When evidence carries no explicit
# ``protocol.id`` (legacy or local-stability-gate output), the
# aggregator treats today's ``on`` files as P6 and today's ``off`` files
# as P3, so a Phase-0 nightly run alongside a pre-Phase-0 nightly run
# is comparable.
_LEGACY_MODE_TO_PROTOCOL = {"on": "P6", "off": "P3"}
_PROTOCOL_TO_LEGACY_MODE = {"P2": "off", "P3": "off", "P4": "off", "P6": "on"}


def _parse_filename(name: str) -> dict | None:
    """Return parsed components for an evidence filename.

    Returns ``None`` for non-comparison files (e.g. ``*-golden.json``).
    The legacy ``<op>-<dtype>-generative.json`` filename maps to
    ``(cloud-default, on, P6)`` by convention.

    The pattern is split so that:
      * trailing ``-minimal`` / ``-guided`` suffixes (existing
        ``--output-suffix`` outputs) are ignored and don't show up as
        spurious comparison cells.
      * profile names include dots and dashes (``local-llama-3.1-8b``).
      * the leg can be ``on``/``off`` (pre-Phase-0) or
        ``p2``/``p3``/``p4``/``p6`` (Phase 0 per-protocol files).
    """
    m = _FILENAME_PATTERN.match(name)
    if not m:
        return None
    profile = m.group("profile") or "cloud-default"
    leg = m.group("leg") or "on"
    suffix = m.group("suffix") or ""
    if suffix and suffix != "":
        # Skip prompt-variant or ad-hoc archive files — they aren't a
        # canonical comparison cell.
        return None
    dtype_long = "float16" if m.group("dtype") == "f16" else "float32"
    if leg.startswith("p"):
        protocol_id = leg.upper()
        mode = _PROTOCOL_TO_LEGACY_MODE.get(protocol_id, "off")
    else:
        mode = leg
        protocol_id = _LEGACY_MODE_TO_PROTOCOL.get(leg)
    return {
        "op": m.group("op"),
        "dtype": dtype_long,
        "profile": profile,
        "mode": mode,
        "protocol_id": protocol_id,
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


def _classify_validity(ev: dict) -> str:
    """Classify a single evidence file's comparability for the intervention.

    The intended skills-on/off comparison requires that each leg's
    evidence reflects an actual OpenCode harness invocation. This
    classifier looks at four independent signals already present in
    schema v3:

      * a resolved LLM model id (``model``)
      * non-zero LLM token usage (``tokens.total``)
      * at least one artifact path captured (``agent.artifacts_found``)
      * a recorded kernel path (``kernel_path``)

    Returns one of:

      * ``ok`` — strong evidence the harness executed (a resolved
        model **and** measurable tokens). Suitable to compare against
        the other leg.
      * ``infra_fail`` — strong instrumentation/configuration-failure
        signature: none of the four signals fired. Not a comparable
        OpenCode-without-skills baseline; excluded from headline
        pass-rate.
      * ``incomplete`` — partial or contradictory evidence. Reported
        separately; not folded into the clean baseline.

    See ``docs/evaluation-methodology.md`` for the rationale and the
    headline rendering rule.
    """
    model = ev.get("model") or ""
    if isinstance(model, str):
        has_model = bool(model.strip())
    else:
        has_model = bool(model)
    tokens = ev.get("tokens") or {}
    has_tokens = int(tokens.get("total", 0) or 0) > 0
    artifacts = (ev.get("agent") or {}).get("artifacts_found") or []
    has_artifacts = bool(artifacts)
    kernel_path = ev.get("kernel_path") or ""
    has_kernel = bool(kernel_path.strip()) if isinstance(kernel_path, str) else bool(kernel_path)

    if not (has_model or has_tokens or has_artifacts or has_kernel):
        return "infra_fail"
    if has_model and has_tokens:
        return "ok"
    return "incomplete"


def _classify_failure_mode(ev: dict) -> str | None:
    """Heuristic failure-mode classifier over schema v3 evidence.

    Returns ``None`` if the cell passed (``_is_overall_pass`` is true),
    otherwise one of the F-codes documented in
    ``docs/evaluation-methodology.md`` (the in-flight derivation from
    existing fields, not the eventual schema-v4 ``failure_category``
    enum):

      * ``F13_infra_fail`` — same as ``_classify_validity == "infra_fail"``.
      * ``F12_incomplete`` — same as ``_classify_validity == "incomplete"``.
      * ``F10_no_artifact`` — the agent never wrote a ``kernel.py``
        (``agent.artifacts_found`` lacks a kernel entry).
      * ``F7_static`` — kernel exists but ``static_verify == "fail"``.
      * ``F9_semantic`` — kernel exists, static passes, but the
        semantic-marker check failed (``semantic_check.passed`` is
        ``False``).
      * ``F8_correctness`` — kernel exists, static + semantic pass, but
        the simulator/runtime evaluator rejected the output
        (``verification.status == "fail"`` and ``verification.mode`` in
        ``{"simulator", "runtime"}``).
      * ``F0_unknown`` — anything that didn't pass but doesn't fit
        the above signature. Worth surfacing on the dashboard so we
        notice that a new failure shape has appeared.

    The classifier checks F13/F12 first because those exclude the cell
    from comparability and would otherwise be misread as a normal
    failure mode. After that it tracks the natural pipeline order
    (artifact -> static -> semantic -> runtime).
    """
    if _is_overall_pass(ev):
        return None
    validity = _classify_validity(ev)
    if validity == "infra_fail":
        return "F13_infra_fail"
    if validity == "incomplete":
        return "F12_incomplete"
    agent = ev.get("agent") or {}
    artifacts = agent.get("artifacts_found") or []
    has_kernel = any(
        isinstance(a, str) and (a == "kernel.py" or a.endswith("/kernel.py"))
        for a in artifacts
    )
    if not has_kernel:
        return "F10_no_artifact"
    if (ev.get("static_verify") or "").lower() == "fail":
        return "F7_static"
    sem = ev.get("semantic_check") or {}
    if sem.get("passed") is False:
        return "F9_semantic"
    ver = ev.get("verification") or {}
    status = (ver.get("status") or "").lower()
    mode = (ver.get("mode") or "").lower()
    if status == "fail" and mode in {"simulator", "runtime"}:
        return "F8_correctness"
    return "F0_unknown"


def _attempts_to_pass(ev: dict) -> int | None:
    """1-based index of the first attempt that produced a usable kernel.

    Reads ``ev.attempts`` (a list of per-attempt records emitted by
    ``collect_generative_evidence.py``) and returns the 1-based index of
    the first attempt where the agent wrote a kernel (``kernel_found``
    truthy) and the attempt was not recorded as a failure
    (``outcome != "fail"``). Returns ``None`` when no such attempt
    exists or when ``attempts`` is absent — that is, the metric is the
    *number of attempts the agent needed before producing a kernel that
    cleared the per-attempt checks*. Cells that never produced any
    usable attempt contribute ``None`` and are skipped by the mean /
    median aggregators.
    """
    attempts = ev.get("attempts") or []
    for i, a in enumerate(attempts, start=1):
        if not isinstance(a, dict):
            continue
        if not a.get("kernel_found"):
            continue
        outcome = (a.get("outcome") or "").lower()
        if outcome == "fail":
            continue
        return i
    return None


def _wilson_ci(passes: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    """Two-sided Wilson score interval for a Bernoulli pass-rate.

    Returns ``(low, high)`` clamped to ``[0, 1]``, or ``None`` when
    ``n <= 0``. Default ``z == 1.96`` corresponds to a 95% CI. We use
    Wilson rather than the normal approximation because pass-rates of
    ``0/n`` and ``n/n`` are common in this dataset and normal-approx
    intervals collapse to a degenerate point at those extremes.
    """
    if n <= 0:
        return None
    p = passes / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = (z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def _extract_metrics(ev: dict, protocol_id: str | None = None) -> dict:
    """Pull the fields we care about out of an evidence document."""
    tokens = ev.get("tokens", {}) or {}
    return {
        "pass": _is_overall_pass(ev),
        "validity": _classify_validity(ev),
        # Failure-mode and attempts-to-pass are additive over schema v3
        # (derived from existing fields) and feed the per-card
        # explainability chips and the intervention-efficiency stat
        # row. See _classify_failure_mode / _attempts_to_pass.
        "failure_mode": _classify_failure_mode(ev),
        "attempts_to_pass": _attempts_to_pass(ev),
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
        "protocol_id": protocol_id,
    }


def load_evidence(
    evidence_dir: Path,
) -> tuple[
    dict[tuple[str, str, str, str], dict],
    dict[tuple[str, str, str, str], dict],
]:
    """Build the evidence tables.

    Returns ``(table_by_mode, table_by_protocol)`` where:

    * ``table_by_mode`` is keyed by ``(op, dtype, profile, mode)`` with
      ``mode in {"on", "off"}``. Used by the pre-Phase-0 paired
      aggregator. A Phase 0 ``P2`` or ``P4`` evidence file is NOT
      placed in this table — only P3 (slot=off) and P6 (slot=on) are,
      so the on/off pair-based aggregator remains comparable when a
      Phase 0 nightly partially populates the matrix.
    * ``table_by_protocol`` is keyed by ``(op, dtype, profile, protocol_id)``
      where ``protocol_id in {"P2", "P3", "P4", "P6"}`` (or any future
      ``P<n>`` once added). Powers the per-protocol aggregator.

    The evidence document's own ``protocol.id`` (Phase 0 schema) wins
    over what the filename implies; legacy files inherit the fallback
    mapping ``on -> P6``, ``off -> P3``.
    """
    table_by_mode: dict[tuple[str, str, str, str], dict] = {}
    table_by_protocol: dict[tuple[str, str, str, str], dict] = {}
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
        # Prefer the on-document fields when present (newer schema_version=3),
        # otherwise fall back to filename inference for legacy files.
        profile = ev.get("model_profile", parsed["profile"])
        protocol_obj = ev.get("protocol") or {}
        doc_protocol_id = (
            protocol_obj.get("id") if isinstance(protocol_obj, dict) else None
        )
        protocol_id = doc_protocol_id or parsed.get("protocol_id")
        # Resolve the legacy on/off slot for the paired aggregator. The
        # document's skills_mode wins when present; otherwise we derive
        # from the protocol id, otherwise the filename's leg field.
        doc_mode = ev.get("skills_mode")
        if isinstance(doc_mode, str) and doc_mode in {"on", "off"}:
            mode = doc_mode
        elif protocol_id and protocol_id in _PROTOCOL_TO_LEGACY_MODE:
            mode = _PROTOCOL_TO_LEGACY_MODE[protocol_id]
        else:
            mode = parsed["mode"]
        metrics = _extract_metrics(ev, protocol_id=protocol_id)
        # Per-protocol table always populated when we know the protocol.
        if protocol_id:
            table_by_protocol[
                (parsed["op"], parsed["dtype"], profile, protocol_id)
            ] = metrics
        # Legacy on/off table: only P3/P6 (or legacy on/off without a
        # protocol id) participate. P2 and P4 would collide with P3 on
        # ``slot=off`` so we omit them — they show up only in the
        # protocol-axis table.
        if protocol_id in (None, "P3", "P6"):
            table_by_mode[(parsed["op"], parsed["dtype"], profile, mode)] = metrics
    return table_by_mode, table_by_protocol


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
            # Per-cell freshness: pulled from the evidence ``date`` field
            # so the dashboard can show how recently each leg of a cell
            # was actually measured. None when that leg's evidence is
            # absent.
            "on_date": (on or {}).get("date") or None,
            "off_date": (off or {}).get("date") or None,
            # Per-cell failure mode for each leg, ``None`` when the leg
            # passed or when that leg's evidence is absent. Lets the
            # dashboard show "11 pass + 1 F8 correctness" rather than a
            # bare "11/12" headline.
            "failure_on": (on or {}).get("failure_mode") or None,
            "failure_off": (off or {}).get("failure_mode") or None,
        }
        if on and off:
            row["quality_delta"] = int(bool(on["pass"])) - int(bool(off["pass"]))
            row["tokens_delta"] = on["tokens_total"] - off["tokens_total"]
            row["cost_delta_usd"] = round(on["cost_usd"] - off["cost_usd"], 6)
            row["elapsed_delta_s"] = round(on["elapsed_s"] - off["elapsed_s"], 2)
            row["attempts_delta"] = on["attempts"] - off["attempts"]
            row["viability_unlocked"] = bool(on["pass"] and not off["pass"])
            row["viability_unlocked_clean"] = bool(
                on["pass"]
                and off.get("validity") == "ok"
                and not off["pass"]
            )
            row["unresolved_due_to_off_infra"] = bool(
                on["pass"] and off.get("validity") == "infra_fail"
            )
        else:
            row["quality_delta"] = None
            row["tokens_delta"] = None
            row["cost_delta_usd"] = None
            row["elapsed_delta_s"] = None
            row["attempts_delta"] = None
            row["viability_unlocked"] = False
            row["viability_unlocked_clean"] = False
            row["unresolved_due_to_off_infra"] = False
        rows.append(row)
    return rows


def _parse_iso_date(value):
    """Parse an ISO-8601 timestamp into a tz-aware datetime, or return None."""
    if not value or not isinstance(value, str):
        return None
    s = value.strip()
    if not s:
        return None
    # The collector emits ``YYYY-MM-DDTHH:MM:SSZ``; ``datetime.fromisoformat``
    # in 3.10 doesn't accept a trailing ``Z``, so normalise to ``+00:00``.
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return None


def aggregate_by_profile(rows: list[dict]) -> dict[str, dict]:
    """Per-profile aggregate of the OpenCode skills-intervention metrics.

    Adds validity-aware aggregates alongside the existing raw fields so
    consumers can choose between the raw delta view (``pass_rate_off``,
    ``viability_unlocked_count``) and the comparability-filtered view
    (``pass_rate_off_clean``, ``viability_unlocked_clean``,
    ``unresolved_due_to_off_infra``). The clean view follows the
    headline rendering rule documented in
    ``docs/evaluation-methodology.md``: ``pass_rate_off_clean`` is
    ``None`` (not ``0/0``) when no clean off-runs exist.
    """
    by_profile: dict[str, dict] = {}
    # Per-profile lists of evidence ``date`` strings so we can compute
    # freshness aggregates (latest on/off run, max staleness of the
    # clean off baseline) in the second pass below. Additive over the
    # schema v2 contract — see docs/evaluation-methodology.md.
    on_dates: dict[str, list[datetime]] = {}
    off_dates: dict[str, list[datetime]] = {}
    off_clean_dates: dict[str, list[datetime]] = {}
    # Per-profile collected ``attempts_to_pass`` integers. Skills-on is
    # collected from every cell whose on-leg passed; skills-off is
    # collected only from cells whose off-leg validity is ``ok`` and
    # whose off-leg passed (so the metric is averaged over comparable
    # baselines, not over infra-failed off legs that didn't pass for
    # instrumentation reasons).
    on_attempts_to_pass: dict[str, list[int]] = {}
    off_clean_attempts_to_pass: dict[str, list[int]] = {}
    # Per-profile failure-mode tallies for the explainability chips.
    # The on-leg is tallied for every paired cell whose on-leg did not
    # pass (so we surface the full failure shape including F13 when the
    # on-leg itself was an infra fail). The off-leg is tallied only
    # over clean off-baseline cells that did not pass, so the chips
    # describe failure modes that meaningfully contribute to the
    # comparable baseline.
    failure_counts_on: dict[str, dict[str, int]] = {}
    failure_counts_off: dict[str, dict[str, int]] = {}
    for row in rows:
        prof = row["profile"]
        agg = by_profile.setdefault(prof, {
            "cells_total": 0,
            "cells_compared": 0,
            "pass_on": 0,
            "pass_off": 0,
            "pass_off_clean": 0,
            "cells_off_ok": 0,
            "cells_off_infra_fail": 0,
            "cells_off_incomplete": 0,
            "tokens_on_sum": 0,
            "tokens_off_sum": 0,
            "cost_on_sum": 0.0,
            "cost_off_sum": 0.0,
            "elapsed_on_sum": 0.0,
            "elapsed_off_sum": 0.0,
            "viability_unlocked_count": 0,
            "viability_unlocked_clean": 0,
            "unresolved_due_to_off_infra": 0,
            "model": None,
        })
        if row.get("on_date"):
            dt = _parse_iso_date(row["on_date"])
            if dt is not None:
                on_dates.setdefault(prof, []).append(dt)
        if row.get("off_date"):
            dt = _parse_iso_date(row["off_date"])
            if dt is not None:
                off_dates.setdefault(prof, []).append(dt)
                # Track dates only for off-legs that are clean baselines
                # (validity == "ok") so ``off_max_staleness_days``
                # measures the comparable baseline, not the
                # placeholder fallbacks.
                off_val = (row.get("off") or {}).get("validity")
                if off_val == "ok":
                    off_clean_dates.setdefault(prof, []).append(dt)
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
            if row.get("viability_unlocked_clean"):
                agg["viability_unlocked_clean"] += 1
            if row.get("unresolved_due_to_off_infra"):
                agg["unresolved_due_to_off_infra"] += 1
            off_validity = row["off"].get("validity", "incomplete")
            if off_validity == "ok":
                agg["cells_off_ok"] += 1
                agg["pass_off_clean"] += int(bool(row["off"]["pass"]))
            elif off_validity == "infra_fail":
                agg["cells_off_infra_fail"] += 1
            else:
                agg["cells_off_incomplete"] += 1
            if agg["model"] is None and row["on"].get("model"):
                agg["model"] = row["on"]["model"]

            # Failure-mode tallies (additive). Only count when the leg
            # did not pass — _classify_failure_mode returns None on
            # pass. We still record F13/F12 on the on-leg because a
            # broken on-leg is a real outcome to surface; on the
            # off-leg we restrict to clean baselines so the chip row
            # describes the comparable failure mix.
            f_on = row["on"].get("failure_mode")
            if f_on:
                failure_counts_on.setdefault(prof, {})[f_on] = (
                    failure_counts_on.setdefault(prof, {}).get(f_on, 0) + 1
                )
            if off_validity == "ok":
                f_off = row["off"].get("failure_mode")
                if f_off:
                    failure_counts_off.setdefault(prof, {})[f_off] = (
                        failure_counts_off.setdefault(prof, {}).get(f_off, 0) + 1
                    )

            # Attempts-to-pass tallies. Only meaningful for cells the
            # leg actually passed; skipped otherwise so the mean is
            # "how many attempts when we *do* succeed", not diluted by
            # never-passed cells.
            if row["on"]["pass"]:
                atp_on = row["on"].get("attempts_to_pass")
                if isinstance(atp_on, int):
                    on_attempts_to_pass.setdefault(prof, []).append(atp_on)
            if off_validity == "ok" and row["off"]["pass"]:
                atp_off = row["off"].get("attempts_to_pass")
                if isinstance(atp_off, int):
                    off_clean_attempts_to_pass.setdefault(prof, []).append(atp_off)
    for prof, agg in by_profile.items():
        n = agg["cells_compared"] or 1
        agg["pass_rate_on"] = round(agg["pass_on"] / n, 3) if agg["cells_compared"] else None
        agg["pass_rate_off"] = round(agg["pass_off"] / n, 3) if agg["cells_compared"] else None
        # Clean off pass-rate: only meaningful when at least one off-run
        # was classified ``ok``. No max(1, ...) fallback — see the
        # methodology doc's "headline rendering rule".
        if agg["cells_off_ok"] > 0:
            agg["pass_rate_off_clean"] = round(
                agg["pass_off_clean"] / agg["cells_off_ok"], 3
            )
        else:
            agg["pass_rate_off_clean"] = None
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
        # Freshness aggregates: each profile's most recent on/off
        # measurement and the staleness (in whole days) of the OLDEST
        # clean off-baseline observation. All three are ISO-8601
        # strings (or an integer for ``off_max_staleness_days``); None
        # when no measurement of that kind exists yet.
        now = datetime.now(timezone.utc)
        on_seen = on_dates.get(prof, [])
        off_seen = off_dates.get(prof, [])
        off_clean_seen = off_clean_dates.get(prof, [])
        agg["on_last_run_at"] = (
            max(on_seen).strftime("%Y-%m-%dT%H:%M:%SZ") if on_seen else None
        )
        agg["off_last_run_at"] = (
            max(off_seen).strftime("%Y-%m-%dT%H:%M:%SZ") if off_seen else None
        )
        if off_clean_seen:
            staleness_days = (now - min(off_clean_seen)).days
            agg["off_max_staleness_days"] = staleness_days
        else:
            agg["off_max_staleness_days"] = None

        # Failure-mode breakdowns. Empty dict (not None) when nothing
        # tallied — keeps the dashboard's optional-chaining simple.
        agg["failure_mode_counts_on"] = dict(failure_counts_on.get(prof, {}))
        agg["failure_mode_counts_off"] = dict(failure_counts_off.get(prof, {}))

        # Attempts-to-pass aggregates (mean rounded to one decimal,
        # median as a plain int via standard high-median for even
        # samples). ``None`` when no pass-record contributed a value
        # so the dashboard can suppress the row entirely.
        on_atp = sorted(on_attempts_to_pass.get(prof, []))
        off_atp = sorted(off_clean_attempts_to_pass.get(prof, []))
        agg["attempts_to_pass_on_n"] = len(on_atp)
        agg["attempts_to_pass_off_clean_n"] = len(off_atp)
        agg["attempts_to_pass_on_mean"] = (
            round(sum(on_atp) / len(on_atp), 2) if on_atp else None
        )
        agg["attempts_to_pass_off_clean_mean"] = (
            round(sum(off_atp) / len(off_atp), 2) if off_atp else None
        )
        agg["attempts_to_pass_on_median"] = (
            int(on_atp[len(on_atp) // 2]) if on_atp else None
        )
        agg["attempts_to_pass_off_clean_median"] = (
            int(off_atp[len(off_atp) // 2]) if off_atp else None
        )

        # Wilson 95% CIs on the displayed pass-rates. Rounded to 3
        # decimals to match the surrounding ``round(..., 3)`` style
        # for ``pass_rate_*``. ``None`` when the denominator is zero
        # (e.g. no clean off baseline yet) so the dashboard can hide
        # the "(CI x-y%)" hint cleanly.
        on_ci = _wilson_ci(agg["pass_on"], agg["cells_compared"])
        agg["pass_rate_on_ci"] = (
            [round(on_ci[0], 3), round(on_ci[1], 3)] if on_ci else None
        )
        off_clean_ci = _wilson_ci(agg["pass_off_clean"], agg["cells_off_ok"])
        agg["pass_rate_off_clean_ci"] = (
            [round(off_clean_ci[0], 3), round(off_clean_ci[1], 3)]
            if off_clean_ci else None
        )
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


def _off_validity_cell(a: dict) -> str:
    """Render a compact 'ok / infra / inc' breakdown for the off-leg column."""
    parts = []
    parts.append(f"{a['cells_off_ok']} ok")
    if a["cells_off_infra_fail"]:
        parts.append(f"{a['cells_off_infra_fail']} infra")
    if a["cells_off_incomplete"]:
        parts.append(f"{a['cells_off_incomplete']} inc")
    return " / ".join(parts)


def render_markdown(rows: list[dict], by_profile: dict[str, dict]) -> str:
    """Markdown report — per-profile summary + per-cell detail table.

    Headline behavior follows ``docs/evaluation-methodology.md``: when a
    profile has no clean off-runs (``pass_rate_off_clean is None``),
    the per-profile table omits its Δ columns and a separate
    "Clean skills-off baseline unavailable" note is emitted instead.
    The raw ``pass_rate_off`` and per-cell raw deltas remain available
    for diagnostic context.
    """
    lines: list[str] = []
    lines.append("# OpenCode skills intervention (skills-on vs skills-off)")
    lines.append("")
    lines.append(
        f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%SZ')} "
        f"from {len(rows)} comparison cell(s). The CI matrix is an "
        f"intended paired OpenCode skills-on/off intervention (same "
        f"harness, model, prompt, budget, evaluator); a per-pair "
        f"validity classifier determines whether each off-leg is a "
        f"comparable baseline. See "
        f"`docs/evaluation-methodology.md` for the contract."
    )
    lines.append("")

    profiles_clean = [
        p for p in sorted(by_profile)
        if by_profile[p].get("pass_rate_off_clean") is not None
    ]
    profiles_unresolved = [
        p for p in sorted(by_profile)
        if by_profile[p].get("pass_rate_off_clean") is None
        and by_profile[p].get("cells_compared", 0) > 0
    ]

    if profiles_clean:
        lines.append("## Per-profile summary (clean baseline available)")
        lines.append("")
        lines.append(
            "| Profile | Cells compared | Pass-rate on | "
            "Pass-rate off (clean) | Off-leg validity | "
            "Tokens Δ (avg) | Cost Δ (avg, USD) | Elapsed Δ (avg) | "
            "Viability unlocked (clean) | Unresolved (off infra-failed) |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for prof in profiles_clean:
            a = by_profile[prof]
            lines.append(
                f"| `{prof}` | {a['cells_compared']}/{a['cells_total']} | "
                f"{_fmt_pct(a['pass_rate_on'])} | "
                f"{_fmt_pct(a['pass_rate_off_clean'])} | "
                f"{_off_validity_cell(a)} | "
                f"{_fmt_delta(a['tokens_delta_avg'])} | "
                f"{_fmt_delta(a['cost_delta_avg_usd'])} | "
                f"{_fmt_delta(a['elapsed_delta_avg_s'], 's')} | "
                f"{a['viability_unlocked_clean']}/{a['cells_off_ok']} | "
                f"{a['unresolved_due_to_off_infra']}/"
                f"{a['cells_off_infra_fail']} |"
            )
        lines.append("")

    if profiles_unresolved:
        lines.append("## Per-profile summary (no clean skills-off baseline)")
        lines.append("")
        for prof in profiles_unresolved:
            a = by_profile[prof]
            non_ok = a["cells_off_infra_fail"] + a["cells_off_incomplete"]
            lines.append(
                f"- `{prof}` — skills-on pass rate "
                f"{_fmt_pct(a['pass_rate_on'])} "
                f"({a['pass_on']}/{a['cells_compared']}). "
                f"Clean skills-off baseline unavailable: "
                f"{a['cells_off_infra_fail']} off-run(s) classified as "
                f"infra/config failures, "
                f"{a['cells_off_incomplete']} as incomplete "
                f"(total non-comparable: {non_ok}/{a['cells_compared']}). "
                f"Raw skills-off pass rate "
                f"{_fmt_pct(a['pass_rate_off'])} is reported as "
                f"diagnostic context only."
            )
        lines.append("")

    lines.append("## Per-cell deltas (skills-on minus skills-off)")
    lines.append("")
    lines.append(
        "| Op | dtype | Profile | Off validity | Quality Δ | "
        "Tokens Δ | Elapsed Δ | Attempts Δ | Viability unlocked (clean) |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for row in rows:
        if not (row["on"] and row["off"]):
            continue
        off_validity = row["off"].get("validity", "incomplete")
        if row.get("viability_unlocked_clean"):
            unlocked = "yes"
        elif row.get("unresolved_due_to_off_infra"):
            unlocked = "unresolved (off infra-failed)"
        elif row["on"]["pass"] and off_validity == "incomplete":
            unlocked = "unresolved (off incomplete)"
        else:
            unlocked = "no"
        lines.append(
            f"| {row['op']} | {row['dtype']} | `{row['profile']}` | "
            f"{off_validity} | "
            f"{_fmt_delta(row['quality_delta'])} | "
            f"{_fmt_delta(row['tokens_delta'])} | "
            f"{_fmt_delta(row['elapsed_delta_s'], 's')} | "
            f"{_fmt_delta(row['attempts_delta'])} | "
            f"{unlocked} |"
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


_PROTOCOL_DELTAS = ("P3-P2", "P4-P3", "P6-P4", "P5-P2")


def _pct_change(a: float | None, b: float | None) -> float | None:
    """``round((a - b) / b * 100, 1)``; ``None`` when ``b in {None, 0}``."""
    if a is None or b in (None, 0):
        return None
    return round((a - b) / b * 100.0, 1)


def aggregate_by_protocol(
    table_by_protocol: dict[tuple[str, str, str, str], dict],
) -> dict[str, dict]:
    """Per-(profile, protocol_id) aggregate of pass-rate / tokens /
    attempts, plus the Phase 0 delta table (``deltas_pp``).

    The output shape matches the contract in
    ``docs/evaluation-methodology.md`` §"Protocol-axis CI mapping
    (Phase 0)":

    ```python
    {
      profile: {
        "by_protocol": {
          "P2": {pass_rate, attempts_to_pass_mean, tokens_mean,
                 n_cells, n_clean},
          ...
        },
        "deltas_pp": {
          "P3-P2": {pass_pp, tokens_pct, attempts_delta},
          ...
          "P5-P2": None,  # protocol not yet run
        },
      }
    }
    ```
    """
    grouped: dict[tuple[str, str], list[dict]] = {}
    for (op, dtype, profile, protocol_id), metric in table_by_protocol.items():
        grouped.setdefault((profile, protocol_id), []).append(metric)
    profiles = sorted({profile for (profile, _pid) in grouped})

    out: dict[str, dict] = {}
    for profile in profiles:
        by_protocol: dict[str, dict] = {}
        for pid in ("P2", "P3", "P4", "P5", "P6"):
            metrics = grouped.get((profile, pid))
            if not metrics:
                continue
            n_cells = len(metrics)
            n_clean = sum(1 for m in metrics if m.get("validity") == "ok")
            pass_count = sum(1 for m in metrics if m.get("pass"))
            tokens_vals = [int(m.get("tokens_total", 0) or 0) for m in metrics]
            tokens_mean = (
                round(sum(tokens_vals) / len(tokens_vals), 1)
                if tokens_vals else None
            )
            attempts_vals = [
                m["attempts_to_pass"] for m in metrics
                if isinstance(m.get("attempts_to_pass"), int)
            ]
            attempts_mean = (
                round(sum(attempts_vals) / len(attempts_vals), 2)
                if attempts_vals else None
            )
            by_protocol[pid] = {
                "pass_rate": round(pass_count / n_cells, 3) if n_cells else None,
                "attempts_to_pass_mean": attempts_mean,
                "tokens_mean": tokens_mean,
                "n_cells": n_cells,
                "n_clean": n_clean,
            }

        deltas_pp: dict[str, dict | None] = {}
        for spec in _PROTOCOL_DELTAS:
            a_id, b_id = spec.split("-")
            a = by_protocol.get(a_id)
            b = by_protocol.get(b_id)
            if not (a and b):
                deltas_pp[spec] = None
                continue
            pass_pp = (
                round((a["pass_rate"] - b["pass_rate"]) * 100, 1)
                if a["pass_rate"] is not None and b["pass_rate"] is not None
                else None
            )
            tokens_pct = _pct_change(a["tokens_mean"], b["tokens_mean"])
            attempts_delta = (
                round(a["attempts_to_pass_mean"] - b["attempts_to_pass_mean"], 2)
                if a["attempts_to_pass_mean"] is not None
                and b["attempts_to_pass_mean"] is not None
                else None
            )
            deltas_pp[spec] = {
                "pass_pp": pass_pp,
                "tokens_pct": tokens_pct,
                "attempts_delta": attempts_delta,
            }
        out[profile] = {"by_protocol": by_protocol, "deltas_pp": deltas_pp}
    return out


def build_summary(
    rows: list[dict],
    by_profile: dict[str, dict],
    *,
    partial_run: bool = False,
    legs_status: dict | None = None,
) -> dict:
    """Assemble the top-level summary document.

    ``partial_run`` and ``legs_status`` are additive v2 fields written
    by the CI ``skills-value-report`` job when one or more matrix legs
    of the most recent nightly were cancelled or failed. They let the
    dashboard render a "partial run" banner without changing the rest
    of the v2 contract; see ``docs/evaluation-methodology.md``.
    """
    out: dict = {
        # schema_version "2" adds the validity-aware aggregates and
        # per-cell ``viability_unlocked_clean`` /
        # ``unresolved_due_to_off_infra`` flags. All schema_version "1"
        # fields are preserved alongside the new ones.
        "schema_version": "2",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "partial_run": bool(partial_run),
        "legs_status": legs_status or None,
        "cells": rows,
        "by_profile": by_profile,
    }
    return out


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
    parser.add_argument(
        "--partial-run", action="store_true",
        help=(
            "Mark the produced summary as a partial nightly. Set by the "
            "skills-value-report CI job when one or more matrix legs of "
            "the most recent nightly were cancelled or failed before "
            "writing fresh evidence."
        ),
    )
    parser.add_argument(
        "--legs-status-file", default=None,
        help=(
            "Optional JSON file describing the conclusion of each matrix "
            "leg in the most recent nightly. Embedded under "
            "`legs_status` in the summary so the dashboard can show "
            "exactly what was/wasn't measured."
        ),
    )
    args = parser.parse_args()

    evidence_dir = Path(args.evidence_dir)
    if not evidence_dir.exists():
        print(f"ERROR: evidence dir not found: {evidence_dir}", file=sys.stderr)
        return 0

    legs_status: dict | None = None
    if args.legs_status_file:
        legs_path = Path(args.legs_status_file)
        if legs_path.exists():
            try:
                legs_status = json.loads(legs_path.read_text())
            except json.JSONDecodeError as e:
                print(
                    f"WARNING: could not parse --legs-status-file {legs_path}: {e}",
                    file=sys.stderr,
                )
        else:
            print(
                f"WARNING: --legs-status-file not found: {legs_path}",
                file=sys.stderr,
            )

    table_by_mode, table_by_protocol = load_evidence(evidence_dir)
    rows = compute_cell_deltas(table_by_mode)
    by_profile = aggregate_by_profile(rows)

    # Phase 0 protocol-axis decomposition. Merge into each profile's
    # entry so older readers can ignore the new keys safely. Profiles
    # with no per-protocol evidence emit empty by_protocol /
    # all-null deltas_pp dicts (additive contract).
    protocol_view = aggregate_by_protocol(table_by_protocol)
    for profile, agg in by_profile.items():
        view = protocol_view.get(profile) or {
            "by_protocol": {},
            "deltas_pp": {spec: None for spec in _PROTOCOL_DELTAS},
        }
        agg["by_protocol"] = view["by_protocol"]
        agg["deltas_pp"] = view["deltas_pp"]
    # Profiles that exist only in the protocol-axis view (no on/off
    # pair, e.g. a future P2/P4-only profile) still get rendered.
    for profile, view in protocol_view.items():
        if profile not in by_profile:
            by_profile[profile] = {
                "cells_total": 0,
                "cells_compared": 0,
                "by_protocol": view["by_protocol"],
                "deltas_pp": view["deltas_pp"],
            }

    summary = build_summary(
        rows, by_profile,
        partial_run=args.partial_run,
        legs_status=legs_status,
    )

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
