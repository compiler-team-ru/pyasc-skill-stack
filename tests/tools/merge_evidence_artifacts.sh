#!/usr/bin/env bash
# =============================================================================
# merge_evidence_artifacts.sh
# =============================================================================
# Per-leg merge of nightly evidence artifacts. Used by the
# `skills-value-report` job in `.github/workflows/ci.yml` to fold the
# fresh per-leg evidence into the working `evidence/` directory while
# preventing stale checkout copies of cross-leg filenames from
# overwriting freshly-written rows.
#
# Each artifact directory is named `evidence-<leg>` where `<leg>` is
# either `cloud-P{2,3,4,6}` (Phase 0 `nightly-gate` matrix), legacy
# `cloud-on` / `cloud-off` (pre-Phase 0 cloud `nightly-gate` matrix —
# still accepted so a partial-rollback works), or `<profile>-<mode>`
# (for the `local-stability-gate` matrix). We `cp -f` only the
# filenames that leg legitimately writes, so even if an upload step
# regresses to "the whole `evidence/` directory", the merge still
# cannot cross-clobber another leg's fresh evidence with a stale
# checkout copy.
#
# Filename conventions (from `collect_generative_evidence.py`):
#   cloud-default + Pn  : <op>-<dtype>-generative-cloud-default-p{2,3,4,6}.json
#   cloud-default + on  : <op>-<dtype>-generative.json     (legacy, pre-Phase 0)
#   cloud-default + off : <op>-<dtype>-generative-cloud-default-off.json
#   <profile>     + <m> : <op>-<dtype>-generative-<profile>-<m>.json
#
# Usage:
#   merge_evidence_artifacts.sh <artifacts-root> <evidence-out>
#
# Exits 0 on success; prints warnings (but does not fail) for
# unrecognised leg names so a future matrix addition doesn't break
# the report job before its own merge case is added.
# =============================================================================

set -euo pipefail

ARTIFACTS_ROOT="${1:-artifacts}"
EVIDENCE_OUT="${2:-evidence}"

mkdir -p "$EVIDENCE_OUT"
shopt -s nullglob

for d in "$ARTIFACTS_ROOT"/evidence-*; do
  [ -d "$d" ] || continue
  leg=${d##*/evidence-}
  echo "Merging from $d (leg=$leg)"
  case "$leg" in
    cloud-P[0-9]*)
      # Phase 0 protocol-axis legs: evidence-cloud-P2, ..., evidence-cloud-P6.
      pid=${leg##cloud-}
      pid_lower=$(echo "$pid" | tr '[:upper:]' '[:lower:]')
      cp -f "$d"/*-generative-cloud-default-"$pid_lower".json "$EVIDENCE_OUT"/ 2>/dev/null || true
      ;;
    cloud-on)
      cp -f "$d"/*-generative.json "$EVIDENCE_OUT"/ 2>/dev/null || true
      ;;
    cloud-off)
      cp -f "$d"/*-generative-cloud-default-off.json "$EVIDENCE_OUT"/ 2>/dev/null || true
      ;;
    perf)
      # Report-only perf-gate: combined perf-vs-AscendC records + the
      # aggregated perf-summary that feeds the dashboard perf panel.
      mkdir -p "$EVIDENCE_OUT/perf-vs-ascendc"
      cp -f "$d"/perf-vs-ascendc/*.json "$EVIDENCE_OUT"/perf-vs-ascendc/ 2>/dev/null || true
      cp -f "$d"/perf-summary.json "$EVIDENCE_OUT"/ 2>/dev/null || true
      ;;
    *-on|*-off)
      mode=${leg##*-}
      profile=${leg%-"$mode"}
      cp -f "$d"/*-generative-"$profile"-"$mode".json "$EVIDENCE_OUT"/ 2>/dev/null || true
      ;;
    *)
      echo "::warning::Unrecognised evidence artifact leg '$leg' \u2014 skipping"
      ;;
  esac
done

count=$(ls -1 "$EVIDENCE_OUT" 2>/dev/null | wc -l)
echo "Evidence directory now contains $count files."
