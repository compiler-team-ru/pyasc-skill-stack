---
name: pyasc-build-run-verify
description: pyasc asc2 kernel build, run, and verification skill. Provides JIT compilation diagnostics, runtime execution guidance, and output verification patterns. Trigger — after kernel implementation, when running pyasc kernels, debugging JIT errors, or verifying kernel output correctness.
---

# pyasc Build, Run, and Verify (asc2)

## Overview

pyasc uses JIT (Just-In-Time) compilation: Python -> ASC-IR -> Ascend C -> Bisheng compiler -> NPU binary. The asc2 API simplifies this by handling synchronization and memory management automatically.

## Workflow

```
Kernel implementation complete
    |
    +-- JIT compilation (automatic on first call)
    |       |
    |       +-- Success -> Run kernel
    |       |
    |       +-- Failure -> Check diagnostics
    |
    +-- Run kernel
    |       |
    |       +-- Model backend (simulator, always available)
    |       |
    |       +-- NPU backend (requires hardware)
    |
    +-- Verify output
            |
            +-- np.testing.assert_allclose / torch.allclose
```

## Running a pyasc asc2 kernel

### Basic execution

> **IMPORTANT**: Use `python3.10` (not `python` or `python3`). The pyasc packages are installed under python3.10.

```bash
# Set up simulator environment (required for Model backend)
export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/tools/simulator/Ascend910B1/lib:$LD_LIBRARY_PATH

# Run with Model backend (simulator) — specify platform explicitly
python3.10 kernel.py -r Model -v Ascend910B1

# Run with NPU backend (requires hardware)
python3.10 kernel.py -r NPU -v Ascend910B1
```

> The `-v Ascend910B1` flag is required. Do NOT use `-v Ascend910B` (missing version suffix).

### Running via pytest

```bash
pytest kernel.py --backend Model --platform Ascend910B1
```

This requires a `conftest.py` with `backend` and `platform` fixtures (see the kernel template).

### Script tool

```bash
bash scripts/run_kernel.sh {kernel_path} [backend] [platform]
```

## JIT Diagnostics

### Environment variables for debugging

| Variable | Purpose | Example |
|----------|---------|---------|
| `PYASC_DUMP_PATH` | Save generated ASC-IR and Ascend C files | `export PYASC_DUMP_PATH=/tmp/pyasc_dump` |
| `PYASC_HOME` | JIT cache root directory | `export PYASC_HOME=$HOME` |
| `PYASC_CACHE_DIR` | Specific cache directory | `export PYASC_CACHE_DIR=$HOME/.pyasc/cache` |

### Compile options for debugging

| Option | Purpose | Usage |
|--------|---------|-------|
| `always_compile=True` | Force recompilation (bypass cache) | `@asc2.jit(always_compile=True)` — **standard for development** |
| `opt_level=0` | Disable optimizations for debugging | `@asc2.jit(opt_level=0)` |

Note: `insert_sync=True` and `run_asc2_passes=True` are defaults for `@asc2.jit` — do not disable them unless debugging.

### Common JIT errors

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `SyntaxError` in AST visitor | Unsupported Python syntax | Check `pyasc-syntax-constraints` |
| Type error in IR builder | Wrong parameter type | Check type constraints for kernel params |
| Bisheng compilation error | Invalid generated Ascend C | Check `PYASC_DUMP_PATH` output for generated code |
| `ImportError: asc` or `asc2` | pyasc not installed | Run `pip install pyasc` or build from source |
| `RuntimeError` on launch | Wrong core count | Verify `CORE_NUM` value |
| Used `range()` instead of `asc2.range()` | Wrong loop construct inside kernel | Replace with `asc2.range()` |

## Verification Patterns

### numpy verification (REQUIRED for asc2)

> **CRITICAL**: Always use **numpy** for host-side verification and data preparation.
> Do NOT use `scipy`, `torch`, or any other library — they are not available in
> the simulator Docker environment and will cause runtime verification failures.
> For reference implementations (e.g., softmax, gelu), write a pure numpy version.

```python
import numpy as np
result = kernel_launch(x)
expected = np.abs(x)  # or whatever the operation should produce
np.testing.assert_allclose(result, expected, atol=1e-3, rtol=1e-3)
```

For operations that need `erf`, use `math.erf` with `np.vectorize`:

```python
import math
_verf = np.vectorize(math.erf)
expected = 0.5 * x * (1.0 + _verf(x / np.sqrt(2.0)))
```

For softmax, implement a pure numpy reference:

```python
def softmax_numpy(x):
    shifted = x - x.max(axis=-1, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / exp_x.sum(axis=-1, keepdims=True)
```

### Verification script

```bash
python scripts/verify_output.py {kernel_path} [--backend Model] [--atol 1e-5]
```

## Backend selection

| Backend | When to use | Availability |
|---------|-------------|-------------|
| `Model` | Development, CI, no NPU hardware | Always (requires CANN simulator libs) |
| `NPU` | Final verification, performance testing | Requires Atlas A2/A3 hardware |

**If runtime execution is unavailable**: Perform static verification (syntax check, ASC-IR dump inspection) and state the limitation explicitly in the delivery.

## References

- [JIT Diagnostics Guide](references/jit-diagnostics.md)
- [Verification Patterns](references/verification-patterns.md)
