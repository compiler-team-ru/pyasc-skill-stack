#!/usr/bin/env python3
"""Automated code-review scoring for pyasc asc2 kernels.

Implements 16 checklist categories for asc2 kernels:
  - 10 structural checks (always applied)
  - 5 operation-aware checks (applied when --op is given)
  - 1 semantic API-usage check (applied when --op is given)

Each category is scored 0 or 1; the final score is out of 16.
The workflow acceptance threshold is >= 12.

In addition to the 16 checks, two **gating** checks introduced in Phase 10
force `accepted: false` (regardless of total score) and emit
``failure_category`` in the JSON output:

  - ``F_legacy_api_import``: kernel imports or references a banned legacy
    framework (``ascendcl``, ``tik``, ``tik2``, ``tbe``, ``TPosition``,
    ``.npu()``). See ``skills/pyasc-api-patterns/SKILL.md`` -> "Required
    host imports & forbidden APIs".
  - ``F_missing_asc2_import``: kernel source calls ``asc2.<something>`` but
    does not ``import asc2``. Would surface as ``NameError`` at simulator
    launch and burn the 150 s sim budget.

Per-cell opt-out: pass ``--allow-legacy-apis`` to skip the
``F_legacy_api_import`` gate (used only by cells that explicitly set
``allow_legacy_apis: true`` in capabilities.yaml; none today).

Usage:
  python score_kernel.py <kernel.py> [--json] [--op abs] [--dtype float16]
                                     [--allow-legacy-apis]
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


# Phase 10 host-import policy. Top-level module names (and submodule prefixes)
# that constitute a legacy v1 / TBE / TIK leak. See
# ``skills/pyasc-api-patterns/SKILL.md`` -> "Required host imports & forbidden
# APIs" for the policy rationale and the four Phase 9 failure modes that
# motivated the ban.
LEGACY_API_MODULES = ("ascendcl", "tik", "tik2", "tbe")
# Banned source-level symbol references (regex-matched against the raw source).
# ``TPosition`` and ``.npu()`` are v1 dispatch / placement markers; the asc2
# stack handles these automatically.
LEGACY_API_SYMBOL_PATTERNS = (
    r"\bTPosition\b",
    r"\.npu\(",
)


def _imported_top_modules(tree: ast.Module) -> set[str]:
    """Return the set of top-level module names imported in *tree*.

    For ``import a.b.c`` -> ``{'a'}``. For ``from a.b import c`` -> ``{'a'}``.
    """
    mods: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".", 1)[0]
                if top:
                    mods.add(top)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                top = node.module.split(".", 1)[0]
                if top:
                    mods.add(top)
    return mods


def _legacy_api_findings(tree: ast.Module, source: str) -> list[str]:
    """Return the list of banned legacy-API references in this kernel.

    Empty list means the kernel is policy-clean.
    """
    findings: list[str] = []
    imported = _imported_top_modules(tree)
    for mod in LEGACY_API_MODULES:
        if mod in imported:
            findings.append(f"import {mod}")
    for pat in LEGACY_API_SYMBOL_PATTERNS:
        if re.search(pat, source):
            findings.append(pat.replace(r"\b", "").replace(r"\(", "(").replace(r"\.", "."))
    return findings


def _has_asc2_call_without_import(tree: ast.Module, source: str) -> bool:
    """True if source calls ``asc2.<something>`` without ``import asc2``.

    The call detection is lenient (string search) to cover both ``asc2.foo()``
    and aliased imports — but if ``asc2`` is in the import set we accept it
    regardless.
    """
    imported = _imported_top_modules(tree)
    if "asc2" in imported:
        return False
    # Look for actual asc2.<name> usage; mere ``asc2`` mention in a docstring
    # is not enough.
    return bool(re.search(r"\basc2\.\w", source))

from semantic_markers import OP_SEMANTIC_MARKERS

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

class ScoreResult(NamedTuple):
    checks: list[Check]
    failure_category: str | None
    gating_detail: dict


def score(path: str, op: str | None = None, dtype: str | None = None,
          allow_legacy_apis: bool = False) -> ScoreResult:
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

    # 7. Verification call (allclose, assert_allclose, or torch.testing.assert_close)
    has_verify = "allclose" in source or "assert_close" in source
    checks.append(Check("verification", has_verify,
                         "verification call present" if has_verify else "no allclose/assert_close call"))

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

    # 14. Multi-shape verification: verification call for multiple test sizes
    verify_count = len(re.findall(r'allclose|assert_close', source))
    has_shape_loop = bool(re.search(r'for\s+\w+\s+in\s+.*shape', source, re.IGNORECASE)
                          or re.search(r'for\s+\w+\s+in\s+\[.*\(', source)
                          or re.search(r'for\s+.*\bin\b.*\[.*,.*\]', source))
    multi_shape = verify_count >= 2 or has_shape_loop
    checks.append(Check("multi_shape_verification", multi_shape,
                         f"verify x{verify_count}, shape loop={'yes' if has_shape_loop else 'no'}"
                         if multi_shape else "single verify call, no shape loop"))

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

    # --- Phase 10 host-import policy (gating; not part of the 16-point score)

    failure_category: str | None = None
    gating_detail: dict = {
        "legacy_api_findings": [],
        "missing_asc2_import": False,
        "allow_legacy_apis": allow_legacy_apis,
    }

    legacy_findings = _legacy_api_findings(tree, source)
    if legacy_findings and not allow_legacy_apis:
        gating_detail["legacy_api_findings"] = legacy_findings
        if failure_category is None:
            failure_category = "F_legacy_api_import"

    if _has_asc2_call_without_import(tree, source):
        gating_detail["missing_asc2_import"] = True
        if failure_category is None:
            failure_category = "F_missing_asc2_import"

    return ScoreResult(checks=checks,
                       failure_category=failure_category,
                       gating_detail=gating_detail)


MAX_SCORE = 16
THRESHOLD = 12


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <kernel.py> [--json] [--op OP] [--dtype DTYPE] "
              f"[--allow-legacy-apis]",
              file=sys.stderr)
        sys.exit(2)

    path = sys.argv[1]
    use_json = "--json" in sys.argv
    allow_legacy_apis = "--allow-legacy-apis" in sys.argv

    op_val: str | None = None
    dtype_val: str | None = None
    for i, arg in enumerate(sys.argv):
        if arg == "--op" and i + 1 < len(sys.argv):
            op_val = sys.argv[i + 1]
        if arg == "--dtype" and i + 1 < len(sys.argv):
            dtype_val = sys.argv[i + 1]

    try:
        result = score(path, op=op_val, dtype=dtype_val,
                       allow_legacy_apis=allow_legacy_apis)
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

    checks = result.checks
    total = sum(1 for c in checks if c.passed)
    gated = result.failure_category is not None
    accepted = (total >= THRESHOLD) and not gated

    if use_json:
        data = {
            "file": path,
            "score": float(total),
            "max_score": MAX_SCORE,
            "threshold": float(THRESHOLD),
            "accepted": accepted,
            "failure_category": result.failure_category,
            "gating_detail": result.gating_detail,
            "checks": {c.name: {"passed": c.passed, "detail": c.detail} for c in checks},
        }
        print(json.dumps(data, indent=2))
    else:
        for c in checks:
            tag = "PASS" if c.passed else "FAIL"
            print(f"  [{tag}] {c.name}: {c.detail}")
        if gated:
            print(f"\n  [GATE FAIL] {result.failure_category}: "
                  f"legacy_findings={result.gating_detail['legacy_api_findings']} "
                  f"missing_asc2_import={result.gating_detail['missing_asc2_import']}")
        print(f"\n  Score: {total}/{MAX_SCORE} (threshold: {THRESHOLD})")
        if accepted:
            print("  ACCEPTED")
        else:
            reason = "score below threshold"
            if gated:
                reason = f"score below threshold or {result.failure_category}" \
                    if total < THRESHOLD else result.failure_category
            print(f"  REJECTED ({reason})")

    sys.exit(0 if accepted else 1)


if __name__ == "__main__":
    main()
