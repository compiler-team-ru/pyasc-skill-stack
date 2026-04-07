---
name: pyasc-api-patterns
description: pyasc API usage patterns and best practices. Provides correct usage for tensor operations, data transfer, sync primitives, JIT options, and type system. Trigger — when calling pyasc APIs, encountering parameter errors, or needing API usage guidance.
---

# pyasc API Best Practices

## API Category Index

| API Category | Key APIs | Documentation | Typical Scenarios |
|-------------|----------|---------------|-------------------|
| **Tensor operations** | `asc.add`, `asc.sub`, `asc.mul`, `asc.div` | [api-tensor-ops.md](references/api-tensor-ops.md) | Element-wise computation |
| **Data transfer** | `asc.data_copy` | [api-data-transfer.md](references/api-data-transfer.md) | GM <-> UB data movement |
| **Sync primitives** | `asc.set_flag`, `asc.wait_flag`, `asc.HardEvent` | [api-sync-pipeline.md](references/api-sync-pipeline.md) | Pipeline synchronization |
| **JIT options** | `@asc.jit(...)`, compile parameters | [api-jit-options.md](references/api-jit-options.md) | Compilation control |
| **Quick reference** | All APIs | [api-quickref.md](references/api-quickref.md) | Quick lookup |

## Core Types

### Tensor types

| Type | Purpose | Example |
|------|---------|---------|
| `asc.GlobalAddress` | Kernel parameter for global memory pointer | `def kernel(x: asc.GlobalAddress, ...)` |
| `asc.GlobalTensor` | Global memory tensor handle | `x_gm = asc.GlobalTensor()` |
| `asc.LocalTensor` | Local (UB) memory tensor | `x_local = asc.LocalTensor(dtype, pos, offset, size)` |

### Enum types

| Enum | Values | Purpose |
|------|--------|---------|
| `asc.TPosition` | `VECIN`, `VECOUT`, `VECCALC`, `A1`, `A2`, `B1`, `B2`, `CO1`, `CO2` | Tensor logical position |
| `asc.HardEvent` | `MTE2_V`, `V_MTE3`, `MTE3_MTE2`, ... | Pipeline sync events |

### Configuration types

| Type | Purpose | Example |
|------|---------|---------|
| `asc.runtime.config.Backend` | Execution backend | `Backend.NPU`, `Backend.Model` |
| `asc.runtime.config.Platform` | Target platform | Hardware-specific SoC |

## Common Patterns

### Kernel function pattern

```python
@asc.jit
def my_kernel(x: asc.GlobalAddress, y: asc.GlobalAddress, z: asc.GlobalAddress, n: int):
    offset = asc.get_block_idx() * n
    x_gm = asc.GlobalTensor()
    x_gm.set_global_buffer(x + offset, n)
    # ... computation ...
```

### Launch pattern

```python
import asc.lib.runtime as rt

my_kernel[core_num, rt.current_stream()](x_tensor, y_tensor, z_tensor, block_length)
```

### Data copy pattern (GM -> UB -> Compute -> UB -> GM)

```python
asc.data_copy(x_local[buf_offset:], x_gm[tile_offset:], tile_length)
asc.set_flag(asc.HardEvent.MTE2_V, buf_id)
asc.wait_flag(asc.HardEvent.MTE2_V, buf_id)

asc.add(z_local[buf_offset:], x_local[buf_offset:], y_local[buf_offset:], tile_length)

asc.set_flag(asc.HardEvent.V_MTE3, buf_id)
asc.wait_flag(asc.HardEvent.V_MTE3, buf_id)
asc.data_copy(z_gm[tile_offset:], z_local[buf_offset:], tile_length)
```

### Dynamic tiling for variable shapes (CRITICAL)

> **When the user specifies multiple shapes, some may be too small for the default
> `tile_num=8` / `core_num=8`.  You MUST use dynamic tiling — otherwise the
> simulator will produce incorrect results silently.**

The tile length (`block_length / tile_num / BUFFER_NUM`) **must** be >= 32 for
DMA to work correctly.  Use `_compute_tiling()` to pick safe values and pass
`tile_num` as an `asc.ConstExpr[int]` kernel parameter.  This ensures the JIT
generates a separate compiled kernel per `tile_num` value (pyasc caches by
`ConstExpr` values; ordinary globals are NOT included in the cache key).

```python
BUFFER_NUM = 2
MIN_TILE_LENGTH = 32

def _compute_tiling(total_length: int) -> tuple:
    """Pick (core_num, tile_num) that keep tile_length >= MIN_TILE_LENGTH."""
    for cores in (8, 4, 2, 1):
        if total_length % cores != 0:
            continue
        block = total_length // cores
        for tiles in (8, 4, 2, 1):
            if block % (tiles * BUFFER_NUM) != 0:
                continue
            if block // tiles // BUFFER_NUM >= MIN_TILE_LENGTH:
                return cores, tiles
    return 1, 1

@asc.jit
def my_kernel(x: asc.GlobalAddress, y: asc.GlobalAddress,
              block_length: int, tile_num: asc.ConstExpr[int]):
    tile_length = block_length // tile_num // BUFFER_NUM
    # ... use tile_num inside range() and tile_length for DMA/compute ...
    for i in range(tile_num * BUFFER_NUM):
        ...

def kernel_launch(x: torch.Tensor) -> torch.Tensor:
    y = torch.zeros_like(x)
    total_length = y.numel()
    core_num, tile_num = _compute_tiling(total_length)
    block_length = total_length // core_num
    my_kernel[core_num, rt.current_stream()](x, y, block_length, tile_num)
    return y
```

**Why `ConstExpr`?** pyasc's JIT file cache keys include `ConstExpr` parameter
values but NOT ordinary module globals.  If you use `global TILE_NUM` mutation,
a stale cache entry compiled with a different `TILE_NUM` may be silently reused,
producing incorrect results.

**Example**: shape `[1, 128]` has 128 elements.  With default `cores=8, tile_num=8`,
`tile_length = 128/8/8/2 = 1` — far too small, produces garbage.
`_compute_tiling(128)` returns `(2, 1)` → `tile_length = 64/1/2 = 32` — correct.

### Verification pattern

```python
import torch
z = kernel_launch(x, y)
assert torch.allclose(z, x + y), "Kernel output mismatch"
```

## API Restrictions

**Do not use inside `@asc.jit` functions**:
- `print()` — use `assert` with f-strings for debug messages
- Standard library imports — all imports must be outside JIT scope
- Dynamic Python features (exceptions, generators, etc.)

**Type constraints for kernel parameters**:
- Supported: `bool`, `int`, `float`, numpy scalars/ndarray, `torch.Tensor`, `asc.GlobalAddress`
- Not supported as runtime args: `str`, `tuple`, `list`, `dict` (use `asc.ConstExpr[T]` for compile-time)

## References

- [API Quick Reference](references/api-quickref.md)
- [Tensor Operations](references/api-tensor-ops.md)
- [Data Transfer](references/api-data-transfer.md)
- [Sync and Pipeline](references/api-sync-pipeline.md)
- [JIT Options](references/api-jit-options.md)
