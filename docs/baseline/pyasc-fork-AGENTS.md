<!--
Vendored from the upstream pyasc-fork checkout.

  source:  /home/aloschilov/workspace/pyasc-fork/AGENTS.md
  sha256:  1544c058df24050b5edce6c7a69d0b00809e914b787fb1783e31885883d36cda
  vendored: 2026-05-22

This file is the **baseline** AGENTS.md mounted into OpenCode test
projects under the Phase 0 P4 protocol (skills off + baseline
AGENTS.md). Do not edit by hand — re-vendor from a pinned upstream
revision and update the sha256 above when refreshing. The skill-stack
AGENTS.md at teams/pyasc-kernel-dev-team/AGENTS.md is intentionally
distinct and is only mounted under P6 (skills on).
-->

# Agent Guidelines for PyAsc2

## Build, Lint, and Test Commands

### Building
- Install in development mode (recommended): `pip install -e .`
- Build wheel: `pip wheel . --no-deps -w dist/`
- Clean build artifacts: `rm -rf build/ dist/`
Other options are in `docs/installation/build-from-source.rst`.

### Linting and Formatting
- Run ruff linter: `ruff check python/`
- Run ruff with fixes: `ruff check --fix python/`
- Format with yapf (it is better to specify filenames): `yapf -rip python/`
- Format with yapf (diff only, recommended): `git diff | yapf-diff -i`
- Format C++: `clang-format -i <filename>`

### Testing
- Run all Python tests: `pytest python/test/`
- Run asc2 kernel tests: `pytest python/test/kernels/asc2/`
- Run asc2 unit tests: `pytest python/test/unit/language/tile/`
- Run specific test file: `pytest python/test/kernels/asc2/test_vadd.py`
- Run specific test function: `pytest python/test/kernels/asc2/test_vadd.py::test_vadd`
- Run tests in parallel: `pytest -n auto python/test/`
- Run with coverage: `pytest --cov=asc2 python/test/`
- Run backend/MLIR tests: `lit -v test/`

## Code Style Guidelines

### Python Code
- Maximum line length: 120 characters
- Use yapf for formatting (PEP8-based with custom settings)
- Use isort for import organization (120 char line length)
- Configure yapf: `column_limit = 120`, `disable_split_list_with_comment = true`
- Use `from __future__ import annotations` for modern type hints
- Use type hints extensively: `from typing import Any, Optional, Union, List, Dict, Tuple`
- Use `@overload` decorator for function overloads
- Use `dataclass` for structured data with `@dataclass` decorator
- Naming: functions/variables `snake_case`, classes `PascalCase`, constants `UPPER_SNAKE_CASE`
- Private members: `_leading_underscore`
- Every file must include the copyright header (see existing files)
- Use docstrings for classes and public methods
- Use `__all__` to explicitly export public API
- Use exceptions: `RuntimeError`, `ValueError`, `NotImplementedError`

### C++ Code
- Header files: `.h` extension, Implementation files: `.cpp` extension
- Use traditional include guards (`#ifndef`, `#define`, `#endif`) - **Do not use** `#pragma once`
- Use clang-format with `.clang-format` configuration (LLVM-based, 120 char line length)
- Indentation: 4 spaces (no tabs)
- Braces: opening brace on new line for functions, same line for classes/structs/enums/namespaces/control statements
- Pointers/references: aligned left (e.g., `int* ptr`)
- Short functions: can be single-line (empty bodies)
- Short if/loops: not allowed on single line
- Naming: classes/structs/enums `PascalCase`, variables/functions `camelCase`, macros `UPPER_SNAKE_CASE`
- Test filenames: `kebab-case` (e.g., `my-test-case.mlir`)
- Include ordering: local project, empty line, MLIR/LLVM/Clang, empty line, std library (all alphabetical)
- After closing namespace, add comment: `// namespace <name>`
- **Do not use** `using namespace` in header files
- Place file-local functions/classes in anonymous namespace (not `static`)
- Template typename arguments: `PascalCase` (e.g., `typename AttrT`), non-type: `camelCase`
- Always use `typename` instead of `class`

### MLIR Code
- Sort operations, types, attributes, interfaces alphabetically in TableGen files
- Each MLIR pass in separate `.cpp` file under `Transforms/` directory
- Pass filename matches pass name without "Pass" suffix (e.g., `UnrollLoop.cpp`)
- List passes alphabetically in `Passes.td`, `CMakeLists.txt`, and `Passes.h`
- Tests in `test/` directory with `kebab-case` filenames
- IR tests: `test/Dialect/<dialect>/IR/`, Transform tests: `test/Dialect/<dialect>/Transforms/`
- Target/emission tests: `test/Dialect/<dialect>/Target/`, Tool tests: `test/Tools/`

### Testing Guidelines
- Use pytest for Python tests, lit for backend/MLIR tests
- Place kernel tests in `python/test/kernels/asc2/`, unit tests in `python/test/unit/language/tile/`
- Use descriptive test names, group related tests in test classes
- Use fixtures for common setup/teardown, `pytest.skip()` for conditional test execution

## PyAsc2 Programming Model

### Core Concepts
- **Tensor**: Global memory (HBM) descriptor with pointer and shape (dynamic shapes)
- **Tile**: On-chip memory (UB, L0A, L0B, L0C, L1) with fixed static shape (known at JIT time)
- Tiles use value semantics - each operation produces a new tile
- Memory hierarchy (Ascend NPU related): `TileLocation.UB` (default), `TileLocation.L0A`/`L0B`/`L0C`, `TileLocation.L1`

### PyAsc2 API Patterns
```python
# Tensor creation and loading
x_gm = asc2.tensor(x_ptr, [size])
tile = asc2.load(tensor, shape=[128], offsets=[base])  # explicit byte offsets
tile = asc2.load(tensor, shape=[128], tile_id=[idx])  # tile_id * shape = offset
scalar = asc2.load(tensor, offsets=[i])
asc2.store(tile, tensor, offsets=[base])

# Arithmetic: add, sub, mul, div, maximum, minimum, left_shift, right_shift
# Comparison: equal, not_equal, greater, greater_equal, less, less_equal
# Unary: abs, ceil, floor, negative, relu, relu, sqrt, rsqrt, exp, log, sin, cos, tanh, erf, softmax
# Operator overloads: +, -, *, /, >, ==, etc.
# Reductions: reduce_sum, reduce_max, reduce_min, reduce_prod (with axes, keep_dims)
# Methods: tile.sum(), tile.max(), etc.
# Shape: reshape, broadcast_to, expand_dims, squeeze
# Creation: full(shape, value), zeros(shape), full_like, zeros_like
# Advanced: matmul(a, b), where(mask, src0, src1)
# Atomics: atomic_add, atomic_max, atomic_min

# Programming model operations
i = asc2.block_idx()    # current NPU block index
n = asc2.block_num()    # total number of blocks
k = asc2.num_tiles(tensor, axis=0, shape=[128])  # tiles fitting along axis

# Loop control
for i in asc2.range(start, stop, step, unroll_factor=4, parallel=False):
    # unroll_factor: how many iterations to unroll
    # parallel=True: enable parallel load/store optimization

# Masking
with asc2.mask(count=8, other=0):
    # operations apply to first 8 elements
```

### JIT Compilation
```python
@asc2.jit # or with options: @asc.jit(always_compile=True, ...)
def kernel(x_ptr, y_ptr, out_ptr, size: int, TILE: asc.ConstExpr[int]):
    x_gm = asc2.tensor(x_ptr, [size])
    y_gm = asc2.tensor(y_ptr, [size])
    out_gm = asc2.tensor(out_ptr, [size])
    # ... kernel implementation

x = torch.rand_like(..., device="cpu") # Initialize tensors with numpy or torch
kernel[8](x, y, out, size, TILE=256)   # Launch with 8 cores
```

### JIT Compile Options
- `run_asc2_passes=True`: Enable AscTile + AscLower pipeline (enabled by default when using `@asc2.jit`)
- `static_alloc=False`: Static vs TPipe-managed UB allocation
- `reuse_ub=False`: Reuse freed UB regions
- `always_compile=False`: Bypass cache, recompile every call
- `opt_level=3`: Bisheng optimization level (1-3)
Other options are in `python/asc/runtime/compiler.py`.

### Architecture Notes
- `python/asc2/`: PyAsc2 frontend API and JIT decorator
- `python/asc/language/tile/`: Tile operations implementation (tensor.py, tile.py, binary_ops.py, unary_ops.py, reduction_ops.py, memory_ops.py, shape_ops.py, creation_ops.py, atomic_ops.py, indexing_ops.py, prog_model_ops.py, range.py)
- `lib/Dialect/AscTile/`: asctile MLIR dialect definition
- `lib/Dialect/AscTile/Transforms/`: AscTile optimization passes
- `lib/Dialect/AscLower/`: Lowering passes (asctile → ascendc)

### Compilation Pipeline
1. Python AST → asctile MLIR (FunctionVisitor)
2. AscTile passes (unrolling, loop transforms, math specialization)
3. AscLower passes (asctile → ascendc dialect)
4. ascendc passes (UB allocation, sync insertion, boilerplate)
5. CodeEmitter (ascendc/asc → Ascend C source)
6. Bisheng compiler (Ascend C → .o binary)

### Key MLIR Passes
- **UnrollLoop**: Unroll loops by factor in `asctile.unroll_factor` attribute
- **PromotePureOps**: Hoist pure ops out of loops
- **TransformMathOps**: Specialize math ops for tiles
- **LowerAscTile**: Main lowering to ascendc dialect
- **ReuseUBAllocation**: Reuse freed UB regions
- **ComputeMemoryConsumption**: Check UB limits (asc2 only)

### Hardware Targets
- Ascend 910B, Ascend 910_93; Ascend 910_95 (a.k.a. Ascend950PR_95)
- Automatic sync insertion per hardware variant
- BufID-based sync for 910_95 (InsertBufIdSync)

### Development Workflow
1. Run linters: `ruff check --fix python/`
2. Format code: `git diff | yapf-diff -i`
3. Format C++: `clang-format -i <files>`
4. Run tests: `pytest -n auto python/test/kernels/asc2/ python/test/unit/language/tile/`

### Debugging
- Use `print_ir_before_all=True` to print IR in-between passes to stderr
- Use `always_compile=True` to bypass cache
- Check MLIR with `lit` tests
