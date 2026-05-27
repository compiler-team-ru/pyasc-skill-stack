#!/usr/bin/env python3
"""Compare v2-eval Stage 3.3 evidence against quarantined matmul-WIP-tree
results. Emits a markdown table + JSON delta for inclusion in
`docs/skill-value-q1-findings.md`.

The comparison is per cell × protocol on the 48 Stage B files:

  evidence/<op>-<dtype>-generative-cloud-default-p<N>.json        (v2-eval, this revision)
  evidence/legacy-cann-mirror-wip/stage33/<op>-<dtype>-generative-cloud-default-p<N>.json
                                                                  (matmul-WIP-tree, quarantined)

For each cell × protocol it reports:
  * v2-eval: overall_pass + verification.status
  * wip:     overall_pass + verification.status
  * flip:   "persist"    -> both fail simulator (drift confirmed)
            "resolved"   -> wip drifts, v2-eval passes
            "regressed"  -> wip passes, v2-eval fails
            "stable_pass" -> both pass
            "shared_fail" -> both fail in the same way
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
EVIDENCE = REPO_ROOT / "evidence"
LEGACY = EVIDENCE / "legacy-cann-mirror-wip" / "stage33"

CELLS = [
    ("abs", "float16"), ("abs", "float32"),
    ("add", "float16"),
    ("reduce_sum", "float16"), ("reduce_sum", "float32"),
    ("gelu", "float16"), ("gelu", "float32"),
    ("leaky_relu", "float16"),
    ("softmax", "float16"),
    ("matmul", "float16"),
    ("rms_norm", "float16"), ("rms_norm", "float32"),
]
PROTOCOLS = ("P2", "P3", "P4", "P6")


def load(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def overall_pass(rec: dict) -> bool:
    if not rec:
        return False
    static = rec.get("static_verify") == "pass"
    scored = (rec.get("score") or {}).get("accepted", False)
    semantic = (rec.get("semantic_check") or {}).get("passed", False)
    kernel = bool(rec.get("kernel_path"))
    mode = (rec.get("verification") or {}).get("mode", "")
    sim_status = (rec.get("verification") or {}).get("status", "")
    runtime_ok = (mode != "simulator") or sim_status == "pass"
    return static and scored and semantic and kernel and runtime_ok


def static_ok(rec: dict) -> bool:
    return bool(rec) and rec.get("static_verify") == "pass"


def sim_status(rec: dict) -> str:
    if not rec:
        return "—"
    v = rec.get("verification") or {}
    if v.get("mode") != "simulator":
        return f"static-only({v.get('status', '?')})"
    return v.get("status", "?")


def classify(v2_rec: dict | None, wip_rec: dict | None) -> str:
    v2_pass = overall_pass(v2_rec or {})
    wip_pass = overall_pass(wip_rec or {})
    if v2_pass and wip_pass:
        return "stable_pass"
    if v2_pass and not wip_pass:
        return "resolved"
    if not v2_pass and wip_pass:
        return "regressed"
    # both fail — distinguish static-only drift from shared total fail
    v2_static = static_ok(v2_rec or {})
    wip_static = static_ok(wip_rec or {})
    if v2_static and wip_static:
        return "persist_drift"
    if not v2_static and not wip_static:
        return "shared_total_fail"
    return "shared_partial_fail"


def main() -> int:
    rows: list[dict] = []
    summary = {
        "stable_pass": 0, "resolved": 0, "regressed": 0,
        "persist_drift": 0, "shared_total_fail": 0,
        "shared_partial_fail": 0, "missing": 0,
    }
    for op, dtype in CELLS:
        dtype_short = dtype.replace("float", "f")
        for proto in PROTOCOLS:
            base = f"{op}-{dtype_short}-generative-cloud-default-{proto.lower()}.json"
            v2_rec = load(EVIDENCE / base)
            wip_rec = load(LEGACY / base)
            if v2_rec is None:
                cls = "missing"
            else:
                cls = classify(v2_rec, wip_rec)
            summary[cls] = summary.get(cls, 0) + 1
            rows.append({
                "cell": f"{op}/{dtype}",
                "protocol": proto,
                "v2_pass": overall_pass(v2_rec or {}),
                "v2_static": static_ok(v2_rec or {}),
                "v2_sim": sim_status(v2_rec or {}),
                "wip_pass": overall_pass(wip_rec or {}),
                "wip_static": static_ok(wip_rec or {}),
                "wip_sim": sim_status(wip_rec or {}),
                "classification": cls,
            })

    print("# v2-eval vs matmul-WIP-tree Stage 3.3 delta\n")
    print(
        "Comparison of the 48 Stage 3.3 cells run against "
        "`compiler-team/pyasc#v2 @ 7b85554a` "
        "(`evidence/<cell>-cloud-default-p<N>.json`) vs the quarantined "
        "`cann/pyasc#wip-matmul-sync-and-reduce-fuse @ 345e13c` "
        "(`evidence/legacy-cann-mirror-wip/stage33/...`).\n"
    )
    print("Classification summary:\n")
    for cls in (
        "stable_pass", "resolved", "regressed",
        "persist_drift", "shared_total_fail",
        "shared_partial_fail", "missing",
    ):
        if summary.get(cls):
            print(f"- **{cls}**: {summary[cls]}")
    print()
    print("Per-cell × protocol:\n")
    print("| Cell | Protocol | v2-eval (pass / static / sim) "
          "| wip (pass / static / sim) | flip |")
    print("|---|---|---|---|---|")
    for r in rows:
        v2 = f"{'✓' if r['v2_pass'] else '✗'} / "
        v2 += f"{'✓' if r['v2_static'] else '✗'} / {r['v2_sim']}"
        wip = f"{'✓' if r['wip_pass'] else '✗'} / "
        wip += f"{'✓' if r['wip_static'] else '✗'} / {r['wip_sim']}"
        print(f"| `{r['cell']}` | {r['protocol']} | {v2} | {wip} | "
              f"{r['classification']} |")

    out_json = REPO_ROOT / "docs" / "v2-vs-wip-delta.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(
        {"rows": rows, "summary": summary}, indent=2,
    ))
    print(f"\n(JSON: {out_json.relative_to(REPO_ROOT)})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
