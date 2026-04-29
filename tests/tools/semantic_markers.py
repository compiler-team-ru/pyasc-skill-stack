"""Canonical OP_SEMANTIC_MARKERS — single source of truth.

Imported by both score_kernel.py and collect_generative_evidence.py to avoid
marker drift between the two semantic checks.
"""

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
    "gelu": ["asc2.erf", "erf(", "gelu", "0.5 * x", "0.5*x"],
    "leaky_relu": ["asc2.where"],
    "softmax": ["asc2.softmax", "asc2.exp", "softmax"],
    "matmul": ["asc2.matmul", "@ "],
    "rms_norm": ["asc2.sqrt", "sum_sq"],
}
