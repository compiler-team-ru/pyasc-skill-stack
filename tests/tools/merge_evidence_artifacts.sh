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
# either `cloud-on` / `cloud-off` (for the cloud `nightly-gate`
# matrix) or `<profile>-<mode>` (for the `local-stability-gate`
# matrix). We `cp -f` only the filenames that leg legitimately writes,
# so even if an upload step regresses to "the whole `evidence/`
# directory", the merge still cannot cross-clobber another leg's
# fresh evidence with a stale checkout copy.
#
# Filename conventions (from `collect_generative_evidence.py`):
#   cloud-default + on  : <op>-<dtype>-generative.json     (legacy)
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
    cloud-on)
      cp -f "$d"/*-generative.json "$EVIDENCE_OUT"/ 2>/dev/null || true
      ;;
    cloud-off)
      cp -f "$d"/*-generative-cloud-default-off.json "$EVIDENCE_OUT"/ 2>/dev/null || true
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
