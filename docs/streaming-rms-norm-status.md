# Streaming RMSNorm — current status against `e2e-rms-norm-streaming-en.md`

The user-visible objective is the streaming RMSNorm pattern described in
[`pyasc-fork/docs/e2e-rms-norm-streaming-en.md`][doc]: per-row two-pass
streaming with a scalar `sum_sq` accumulator and an arbitrary, runtime-
dynamic `num_cols`. This file records what works and what is currently
blocked in the pinned MR-85 pyasc build.

[doc]: ../../pyasc-fork/docs/e2e-rms-norm-streaming-en.md

## What ships today (golden kernels)

`golden/kernels/rms_norm_f16.py` and `golden/kernels/rms_norm_f32.py`
use the `asc2.rms_norm` builtin with the normalization dimension as a
compile-time `ConstExpr[int]`. Multi-core distribution covers the row
dim. Per-shape recompile is required, but every shape works end-to-end
on the simulator with `Ascend910B1` and the patched
`ghcr.io/aloschilov/pyasc-sim:py3.11` image. This satisfies the
"dynamic batch dimension" reading of the user prompt.

## What is blocked: streaming form with dynamic `num_cols`

The streaming kernel from the doc compiles, but produces incorrect
results in the simulator due to two MR-85 codegen limitations that are
each independently observable:

### 1. `asc2.mask(count=N)` does not constrain `asc2.store`

Reproducer (verified):

```python
@asc2.jit(always_compile=True)
def kernel(out_ptr, num_cols, tile_cols: asc.ConstExpr[int],
           valid: asc.ConstExpr[int]):
    out_gm = asc2.tensor(out_ptr, [1, num_cols])
    one_tile = asc2.full([1, tile_cols], 7.0, dtype=asc.float32)
    with asc2.mask(count=valid):
        asc2.store(one_tile, out_gm, offsets=[0, 0])

# num_cols=9, tile_cols=8, valid=3, initial out=99
# Expected: [7, 7, 7, 99, 99, 99, 99, 99, 99]
# Got:      [7, 7, 7, 7,  7,  7,  7,  7,  99]
```

The mask context is silently ignored; the full tile is written.

### 2. Scalar broadcast and `asc2.full` only fill one vector lane

Reproducers (each verified independently):

```python
# tile * scalar
x = asc2.load(x_gm, [1, 1024], offsets=[0, 0])
out = x * 2.0          # only out[0:64] equals 2 * x[0:64]; out[64:] is 0
asc2.store(out, ...)
```

```python
# asc2.full of a wide tile
full = asc2.full([1, 1024], 7.0, dtype=asc.float32)
out = x * full         # full[0:64] == 7, full[64:] == 0
```

`asc2.abs(x)` and `x * x` (tile * tile, same shape) both work correctly
across the full 1024-element tile. The breakage is specifically in the
"materialize-a-scalar-into-a-wide-tile" path that the streaming kernel
needs for `inv_rms_tile = asc2.full([1, tile_cols], inv_rms, ...)`.

64 floats × 4 bytes = 256 bytes, exactly one vector register on
Ascend910B1, so the symptom is "first 64 lanes correct, rest are
garbage / zero".

## Why the streaming kernel cannot dodge these issues today

A correct streaming RMSNorm needs at least one of:

- `asc2.mask(count=tail_cols)` around `asc2.store` for the tail tile,
  or
- a vector-wide `inv_rms_tile = asc2.full([1, tile_cols], inv_rms)`
  that we can multiply against `x * gamma`.

Both paths rely on the broken primitives above. Host-side zero-padding
removes the need for `asc2.mask`, but the wide-tile broadcast path is
still required for `inv_rms`, so the kernel still produces only the
first 64 lanes of each row correctly.

## Path forward

This is a tooling problem, not a skill-stack problem:

1. Fix the two codegen issues upstream in `pyasc` (ideally before the
   next image rebuild). Each has a sub-100-line standalone reproducer
   in this file.
2. Once `asc2.full` and scalar broadcast vectorize correctly, port the
   streaming kernel from the doc as `golden/kernels/rms_norm_streaming_f32.py`
   and add a third capability cell `rms_norm/streaming` whose semantic
   check looks for `sum_sq`, `asc2.range(...full_tiles...)`, and
   `pad_value=0.0`.
3. The `rms_norm/float16` and `rms_norm/float32` cells stay as the
   builtin form; they remain useful as a baseline and as a fallback
   when the model decides the row fits in UB.

## Investigation artefacts

The reproducers above were written and exercised against
`pyasc-sim:rmsnorm` (the patched MR-85 image). They were intentionally
deleted after diagnosis to keep the repo root clean — re-create them
from this document if needed when revisiting.

## Resolution: single-vector-lane streaming with host padding

The capability cells `rms_norm/float16` and `rms_norm/float32` no longer
use the `asc2.rms_norm` builtin or the wide-tile streaming pattern from
the doc. Both have been replaced with a **single-vector-lane streaming
kernel** that:

- pins `tile_cols = 64` (one Ascend910B1 SIMD lane = 256 bytes), which
  sidesteps both wide-tile bugs above (`asc2.mask` and the 64-lane
  `asc2.full` / scalar-broadcast issue);
- writes outputs as `[1, 64]` tile stores rather than scalar
  `SetValueOp`, sidestepping a third MR-85 multi-core bug discovered
  while exploring `pyasc-fork/docs/e2e-rms-norm-column-loop-en.md`:
  pure-scalar `asc2.store(plain_value, gm, offsets=[r,c])` is silently
  dropped on even-indexed blocks (rows from `block_idx ∈ {0, 2, 4, …}`
  come back as zero, deterministically reproduced with an 8-row probe);
- pads `x` and `gamma` to a multiple of `tile_cols` host-side, so the
  kernel sees a clean rectangle and the tail vanishes; `sum_sq` is
  divided by the REAL `num_cols`, so padded zeros do not bias the result;
- accepts both `num_rows` and `num_cols` as runtime `int`, giving the
  truly dynamic norm dim that the user-visible prompt asks for.

The pinned production goldens are
[`golden/kernels/rms_norm_f32.py`](../golden/kernels/rms_norm_f32.py)
(atol=rtol=1e-4) and
[`golden/kernels/rms_norm_f16.py`](../golden/kernels/rms_norm_f16.py)
(float32 accumulator, atol=rtol=5e-2), both verified at shape `(8, 1055)`
on `Ascend910B1` and rescalable to `(64, 100003)` by changing only the
launcher constants. The teaching pattern lives in
`skills/pyasc-api-patterns/SKILL.md` under "Normalization layers —
single-vector-lane streaming RMSNorm". See also
[`pyasc-fork/docs/e2e-rms-norm-column-loop-en.md`](../../pyasc-fork/docs/e2e-rms-norm-column-loop-en.md)
for the alternative pattern that motivated this resolution.

The two upstream bugs above (and the multi-core SetValueOp bug
described here) remain open; once they are fixed in a future pyasc
image, the kernel can be simplified back toward either the doc's wide-
tile streaming or its pure-scalar column-loop without changing the
capability matrix shape.
