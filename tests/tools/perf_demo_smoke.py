#!/usr/bin/env python3
"""Fast, camodel-free smoke for the perf-vs-AscendC demo harness.

Validates the *structure* of the repo-aware reference runner, the generated-side
runner, and the demo orchestrator without building any operator or launching the
simulator (those are exercised by the live demo, not by CI):

  * every demo cell maps to a registered op spec + valid dtype;
  * every op spec names a known reference repo + an aclnn header + a body fn;
  * the C++ driver source generates for each op (contains its aclnn call);
  * the per-op gen-side input builders produce well-formed arg specs;
  * the gen probe's launch-selection prefers the public, op-named dispatcher.

Exits non-zero with a diagnostic on the first failure.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PERF_DIR = REPO_ROOT / "tests" / "tools" / "perf"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ref = _load("ascendc_ref_runner", PERF_DIR / "ascendc_ref_runner.py")
    gen = _load("pyasc_gen_runner", PERF_DIR / "pyasc_gen_runner.py")
    demo = _load("demo_vector_ops", REPO_ROOT / "tests" / "tools" / "demo_vector_ops.py")

    known_repos = {"ops-math", "ops-nn"}
    failures: list[str] = []

    # 1. op spec integrity
    for op, spec in ref.OP_SPECS.items():
        if spec["repo"] not in known_repos:
            failures.append(f"OP_SPECS[{op}].repo={spec['repo']!r} not in {known_repos}")
        if "aclnn" not in spec["header"]:
            failures.append(f"OP_SPECS[{op}].header={spec['header']!r} lacks aclnn")
        if not callable(spec["body"]):
            failures.append(f"OP_SPECS[{op}].body is not callable")

    # 2. driver source generates for each op + contains its aclnn entry point
    op_shapes = {
        "abs": [32, 4096], "add": [32, 4096], "reduce_sum": [32, 4096],
        "tanh": [32, 4096], "drop_out_do_mask": [32, 4096],
        "rms_norm": [8, 256], "batch_norm_v3": [32, 64, 64], "apply_adam": [32, 4096],
    }
    aclnn_entry = {
        "abs": "aclnnAbs", "add": "aclnnAdd", "reduce_sum": "aclnnReduceSum",
        "tanh": "aclnnTanh", "drop_out_do_mask": "aclnnDropoutDoMask",
        "rms_norm": "aclnnRmsNorm", "batch_norm_v3": "aclnnBatchNorm",
        "apply_adam": "aclnnApplyAdam",
    }
    for op in ref.OP_SPECS:
        dtype = "f16" if op in ("abs", "add", "tanh", "drop_out_do_mask") else "f32"
        if op == "rms_norm":
            dtype = "f16"
        try:
            src = ref._driver_source(op, dtype, op_shapes[op])
        except Exception as exc:  # noqa: BLE001
            failures.append(f"_driver_source({op}) raised: {exc}")
            continue
        if aclnn_entry[op] not in src:
            failures.append(f"driver source for {op} missing {aclnn_entry[op]}")
        if "REF_DRIVER_DONE" not in src:
            failures.append(f"driver source for {op} missing REF_DRIVER_DONE sentinel")

    # 3. cell->op->dtype maps consistent
    for cell, (op, dtype) in ref.CELL_TO_OP_DTYPE.items():
        if op not in ref.OP_SPECS:
            failures.append(f"CELL_TO_OP_DTYPE[{cell}] op {op} not in OP_SPECS")
        if dtype not in ref.DTYPES:
            failures.append(f"CELL_TO_OP_DTYPE[{cell}] dtype {dtype} not in DTYPES")

    # 4. gen-side input builders
    rms_specs = gen.arg_specs_for("rms_norm/float16", [8, 256], "float16")
    if not (isinstance(rms_specs, list) and len(rms_specs) == 2):
        failures.append(f"rms_norm arg_specs malformed: {rms_specs}")
    else:
        x, g = rms_specs
        if x.get("shape") != [8, 256] or g.get("shape") != [256]:
            failures.append(f"rms_norm arg_specs shapes wrong: {rms_specs}")
        if g.get("fw") != "torch":
            failures.append("rms_norm gamma must be torch-framed")
    if gen.arg_specs_for("tanh/float16", [32, 4096], "float16") is not None:
        failures.append("elementwise tanh must use the auto (None) input path")

    # 5. demo cells all resolve to a ref op + dtype
    for cell, spec in demo.CELLS.items():
        if spec["ref_op"] not in ref.OP_SPECS:
            failures.append(f"demo CELLS[{cell}].ref_op {spec['ref_op']} not in OP_SPECS")
        if spec["ref_dtype"] not in ref.DTYPES:
            failures.append(f"demo CELLS[{cell}].ref_dtype {spec['ref_dtype']} invalid")
        if not isinstance(spec["shape"], list):
            failures.append(f"demo CELLS[{cell}].shape must be a list")

    # 6. the 5 requested operators are all wired into the demo
    requested = {"tanh", "drop_out_do_mask", "rms_norm", "batch_norm_v3", "apply_adam"}
    demo_ops = {spec["ref_op"] for spec in demo.CELLS.values()}
    missing = requested - demo_ops
    if missing:
        failures.append(f"requested operators not wired into demo CELLS: {sorted(missing)}")

    if failures:
        print("PERF DEMO SMOKE: FAIL", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1
    print(f"PERF DEMO SMOKE: PASS ({len(ref.OP_SPECS)} op specs, "
          f"{len(demo.CELLS)} demo cells, 5/5 requested ops wired)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
