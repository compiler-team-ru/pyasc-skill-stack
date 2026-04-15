#!/usr/bin/env python3
"""Automated code-review scoring for pyasc asc2 kernels.

Implements 16 checklist categories for asc2 kernels:
  - 10 structural checks (always applied)
  - 5 operation-aware checks (applied when --op is given)
  - 1 semantic API-usage check (applied when --op is given)

Each category is scored 0 or 1; the final score is out of 16.
The workflow acceptance threshold is >= 12.

Usage:
  python score_kernel.py <kernel.py> [--json] [--op abs] [--dtype float16]
"""

import ast
import json
import re
import sys
from pathlib import Path
from typing import NamedTuple


class Check(NamedTuple):
    name: str
    passed: bool
    detail: str = ""


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------

def _is_jit(node: ast.expr) -> bool:
    """Match @asc.jit, @asc2.jit, or their call forms."""
    if isinstance(node, ast.Attribute):
        if isinstance(node.value, ast.Name):
            if node.value.id in ("asc", "asc2") and node.attr == "jit":
                return True
    if isinstance(node, ast.Call):
        return _is_jit(node.func)
    return False


def _jit_funcs(tree: ast.Module) -> list[ast.FunctionDef]:
    return [
        n for n in ast.iter_child_nodes(tree)
        if isinstance(n, ast.FunctionDef) and any(_is_jit(d) for d in n.decorator_list)
    ]


def _all_calls(node: ast.AST) -> list[str]:
    """Return a flat list of callee names found under *node*."""
    names = []
    for n in ast.walk(node):
        if isinstance(n, ast.Call):
            c = n.func
            if isinstance(c, ast.Name):
                names.append(c.id)
            elif isinstance(c, ast.Attribute):
                names.append(c.attr)
    return names


def _count_numeric_tuples_in_source(source: str) -> int:
    """Count occurrences of shape-like tuples/lists such as (32, 4096) or [1, 128]."""
    return len(re.findall(r'[\(\[]\s*\d+\s*,\s*\d+\s*[\)\]]', source))


BANNED_BUILTINS = {"print", "input", "open", "eval", "exec", "compile"}
BANNED_STMTS = (ast.AsyncFunctionDef, ast.AsyncFor, ast.AsyncWith,
                ast.Yield, ast.YieldFrom, ast.Try,
                ast.Raise, ast.Global, ast.Nonlocal,
                ast.Import, ast.ImportFrom, ast.Lambda, ast.With)
try:
    BANNED_STMTS = (*BANNED_STMTS, ast.TryStar)
except AttributeError:
    pass

OP_SEMANTIC_MARKERS: dict[str, list[str]] = {
    "abs": ["asc2.abs"],
    "exp": ["asc2.exp"],
    "log": ["asc2.log"],
    "sqrt": ["asc2.sqrt"],
    "relu": ["asc2.relu"],
    "erf": ["asc2.erf"],
    "add": ["x + y", "x+y", "+ y", "+y"],
    "sub": ["x - y", "x-y", "- y", "-y"],
    "mul": ["x * y", "x*y", "* y", "*y"],
    "div": ["x / y", "x/y", "/ y", "/y"],
    "reduce_sum": ["asc2.reduce_sum", ".sum("],
    "reduce_max": ["asc2.reduce_max", ".max("],
    "reduce_min": [".min("],
    "gelu": ["asc2.erf"],
    "leaky_relu": ["asc2.where"],
    "softmax": ["asc2.softmax", "asc2.exp", "softmax"],
    "matmul": ["asc2.matmul", "@ "],
}

OP_ARITY = {
    "abs": 1, "exp": 1, "log": 1, "sqrt": 1, "relu": 1, "erf": 1,
    "sin": 1, "cos": 1, "neg": 1, "ceil": 1, "floor": 1, "rsqrt": 1, "tanh": 1,
    "add": 2, "sub": 2, "mul": 2, "div": 2,
    "reduce_sum": 1, "reduce_max": 1, "reduce_min": 1, "reduce_prod": 1,
    "gelu": 1, "leaky_relu": 1, "softmax": 1,
    "matmul": 3,
}

DTYPE_STRINGS = {
    "float16": ["float16", "f16", "fp16"],
    "float32": ["float32", "f32", "fp32"],
    "bfloat16": ["bfloat16", "bf16"],
    "int32": ["int32", "i32"],
    "int8": ["int8", "i8"],
}


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score(path: str, op: str | None = None, dtype: str | None = None) -> list[Check]:
    source = Path(path).read_text()
    tree = ast.parse(source)
    checks: list[Check] = []

    jit = _jit_funcs(tree)

    # 1. @asc2.jit or @asc.jit decorator present
    checks.append(Check("jit_decorator", len(jit) > 0,
                         f"{len(jit)} JIT function(s)" if jit else "missing"))

    # 2. Kernel does not return a value
    ret_ok = True
    for f in jit:
        for n in ast.walk(f):
            if isinstance(n, ast.Return) and n.value is not None:
                ret_ok = False
    checks.append(Check("no_return_value", ret_ok,
                         "OK" if ret_ok else "kernel returns a value"))

    # 3. No banned constructs
    ban_issues = []
    for f in jit:
        for n in ast.walk(f):
            if isinstance(n, BANNED_STMTS):
                ban_issues.append(type(n).__name__)
            if isinstance(n, ast.Call):
                c = n.func
                if isinstance(c, ast.Name) and c.id in BANNED_BUILTINS:
                    ban_issues.append(c.id)
            if isinstance(n, (ast.Break, ast.Continue)):
                ban_issues.append(type(n).__name__)
    checks.append(Check("no_banned_constructs", len(ban_issues) == 0,
                         "OK" if not ban_issues else f"found: {', '.join(ban_issues[:5])}"))

    # 4. asc2.load / asc2.store present in JIT functions
    calls_in_jit = []
    for f in jit:
        calls_in_jit.extend(_all_calls(f))
    has_load_store = "load" in calls_in_jit and "store" in calls_in_jit
    checks.append(Check("asc2_load_store", has_load_store,
                         "load+store" if has_load_store else "missing asc2.load/asc2.store"))

    # 5. asc2.tensor present in JIT functions
    has_tensor = "tensor" in calls_in_jit
    checks.append(Check("asc2_tensor", has_tensor,
                         "present" if has_tensor else "missing asc2.tensor"))

    # 6. asc2.range or range used for loops in JIT
    has_range = "range" in calls_in_jit
    checks.append(Check("loop_range", has_range,
                         "range/asc2.range present" if has_range else "no range call in kernel"))

    # 7. Verification call (allclose or assert_allclose in file)
    has_verify = "allclose" in source
    checks.append(Check("verification", has_verify,
                         "allclose present" if has_verify else "no allclose call"))

    # 8. Launch pattern: kernel[core_num](...) — asc2 style (no stream)
    has_launch = bool(re.search(r'\w+\[\s*\w+\s*\]\(', source) or
                      re.search(r'\w+\[\s*\d+\s*\]\(', source))
    checks.append(Check("launch_pattern", has_launch,
                         "kernel launch found" if has_launch else "no launch pattern"))

    # 9. GlobalAddress used in kernel parameter types
    has_ga = "GlobalAddress" in source
    checks.append(Check("global_address", has_ga,
                         "GlobalAddress found" if has_ga else "missing GlobalAddress type"))

    # 10. File is syntactically valid Python (already parsed above)
    checks.append(Check("valid_python", True, "parsed without errors"))

    # --- Operation-aware checks (11-15) ---

    # 11. Input arity: correct number of GlobalAddress parameters in JIT function
    if op and op in OP_ARITY and jit:
        expected_arity = OP_ARITY[op]
        first_jit = jit[0]
        ga_params = 0
        for arg in first_jit.args.args:
            ann = arg.annotation
            if ann:
                ann_str = ast.dump(ann)
                if "GlobalAddress" in ann_str:
                    ga_params += 1
        if ga_params == 0:
            ga_params = len(first_jit.args.args)
        passed = ga_params >= expected_arity
        checks.append(Check("input_arity", passed,
                             f"expected>={expected_arity} params, found {ga_params}"))
    else:
        checks.append(Check("input_arity", True, "skipped (no --op or op not in arity table)"))

    # 12. Output dtype annotation: verification section references expected dtype
    if dtype and dtype in DTYPE_STRINGS:
        needles = DTYPE_STRINGS[dtype]
        found_dtype = any(n in source for n in needles)
        checks.append(Check("output_dtype", found_dtype,
                             f"found dtype ref" if found_dtype
                             else f"none of {needles} found in source"))
    else:
        checks.append(Check("output_dtype", True, "skipped (no --dtype)"))

    # 13. Tiling consistency: ceildiv or integer-division tiling math present
    has_tiling = bool(
        "ceildiv" in source
        or re.search(r'\(\s*\w+\s*\+\s*\w+\s*-\s*1\s*\)\s*//', source)
        or re.search(r'\(\s*\w+\s*\+\s*\w+\s*-\s*1\s*\)\s*/', source)
        or "tile" in source.lower()
        or "BLOCK" in source
        or "block_size" in source
        or "block_num" in source
    )
    checks.append(Check("tiling_consistency", has_tiling,
                         "tiling logic found" if has_tiling else "no tiling math found"))

    # 14. Multi-shape verification: allclose called for multiple test sizes
    allclose_count = len(re.findall(r'allclose', source))
    has_shape_loop = bool(re.search(r'for\s+\w+\s+in\s+.*shape', source, re.IGNORECASE)
                          or re.search(r'for\s+\w+\s+in\s+\[.*\(', source)
                          or re.search(r'for\s+.*\bin\b.*\[.*,.*\]', source))
    multi_shape = allclose_count >= 2 or has_shape_loop
    checks.append(Check("multi_shape_verification", multi_shape,
                         f"allclose x{allclose_count}, shape loop={'yes' if has_shape_loop else 'no'}"
                         if multi_shape else "single allclose, no shape loop"))

    # 15. No hardcoded shapes inside JIT function body
    no_hardcoded = True
    if jit:
        for f in jit:
            jit_source = ast.get_source_segment(source, f)
            if jit_source:
                tuples_in_jit = _count_numeric_tuples_in_source(jit_source)
                if tuples_in_jit > 0:
                    no_hardcoded = False
    checks.append(Check("no_hardcoded_shapes", no_hardcoded,
                         "shapes parameterized" if no_hardcoded
                         else "numeric shape tuples found inside JIT function"))

    # 16. Semantic API usage: expected API calls for the operation are present
    if op and op in OP_SEMANTIC_MARKERS:
        markers = OP_SEMANTIC_MARKERS[op]
        found = [m for m in markers if m in source]
        sem_pass = len(found) > 0
        checks.append(Check("semantic_api_usage", sem_pass,
                             f"found {found}" if sem_pass
                             else f"none of {markers} found in source"))
    else:
        checks.append(Check("semantic_api_usage", True,
                             "skipped (no --op or op not in semantic table)"))

    return checks


MAX_SCORE = 16
THRESHOLD = 12


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <kernel.py> [--json] [--op OP] [--dtype DTYPE]",
              file=sys.stderr)
        sys.exit(2)

    path = sys.argv[1]
    use_json = "--json" in sys.argv

    op_val: str | None = None
    dtype_val: str | None = None
    for i, arg in enumerate(sys.argv):
        if arg == "--op" and i + 1 < len(sys.argv):
            op_val = sys.argv[i + 1]
        if arg == "--dtype" and i + 1 < len(sys.argv):
            dtype_val = sys.argv[i + 1]

    try:
        checks = score(path, op=op_val, dtype=dtype_val)
    except SyntaxError as exc:
        if use_json:
            print(json.dumps({"file": path, "score": 0.0, "max_score": MAX_SCORE, "error": str(exc)}))
        else:
            print(f"FAIL: SyntaxError: {exc}")
        sys.exit(1)
    except FileNotFoundError:
        if use_json:
            print(json.dumps({"file": path, "score": 0.0, "max_score": MAX_SCORE, "error": "file not found"}))
        else:
            print(f"FAIL: file not found: {path}")
        sys.exit(1)

    total = sum(1 for c in checks if c.passed)

    if use_json:
        data = {
            "file": path,
            "score": float(total),
            "max_score": MAX_SCORE,
            "threshold": float(THRESHOLD),
            "accepted": total >= THRESHOLD,
            "checks": {c.name: {"passed": c.passed, "detail": c.detail} for c in checks},
        }
        print(json.dumps(data, indent=2))
    else:
        for c in checks:
            tag = "PASS" if c.passed else "FAIL"
            print(f"  [{tag}] {c.name}: {c.detail}")
        print(f"\n  Score: {total}/{MAX_SCORE} (threshold: {THRESHOLD})")
        if total >= THRESHOLD:
            print("  ACCEPTED")
        else:
            print("  REJECTED")

    sys.exit(0 if total >= THRESHOLD else 1)


if __name__ == "__main__":
    main()
