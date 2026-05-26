# Glossary — pyasc / asc2 / capability cell metadata

This is the single source of truth for the terminology used by
[capabilities.yaml](../capabilities.yaml), the golden kernels under
[golden/kernels/](../golden/kernels/), and the evaluation methodology
in [docs/evaluation-methodology.md](evaluation-methodology.md).

It documents the standard pyasc / asc2 / CANN vocabulary (sections 1–5)
and the new Phase 1 capability-cell metadata enums (section 6). Every
term has a one-sentence definition, one example from a current golden
kernel, and a back-link to the cell that uses it.

The metadata fields documented here are reporting-only — they describe
what each cell *claims to prove*. They do not change the prompt the
agent sees and they do not change kernel behavior. The cell metadata
must mirror the golden kernel's docstring header (see
[golden/kernels/](../golden/kernels/)); the
[tests/tools/check_capabilities.py](../tests/tools/check_capabilities.py)
gate refuses drift between the two.

---

## §1 Shape vocabulary

- **shape** — the ordered tuple of axis lengths for a tensor; in
  capabilities.yaml a cell's `shapes` field is the list of test shapes
  the golden kernel is verified at. Example: `abs/f16` lists shapes
  `[[1, 128], [4, 2048], [32, 4096]]`. Used by every cell.
- **dtype** — the element type of a tensor, one of `float16` /
  `float32` in the current matrix. Example: `reduce_sum/f16` differs
  from `reduce_sum/f32` only in `dtype` and the corresponding
  `OUT_PAD` constant in [`golden/kernels/reduce_sum_f16.py`](../golden/kernels/reduce_sum_f16.py).
- **contiguous layout** — a layout where neighboring elements along
  the last axis are at adjacent memory addresses; every asc2 kernel
  in this repo assumes contiguous inputs.
- **axis** — a single dimension of a tensor's shape; `axis=-1` denotes
  the last axis (the convention used by every reducing op).
- **stride** — the element-count step between adjacent elements along
  one axis; not surfaced in this repo because all loads use the
  contiguous default.

## §2 Memory hierarchy (Ascend 950 / C310)

- **GM** — Global Memory, off-core DRAM. Kernel arguments
  (`asc.GlobalAddress`) and `asc2.tensor(ptr, shape)` views live in
  GM. Every kernel starts with `asc2.load(<gm>, ...)` and ends with
  `asc2.store(<tile>, <gm>, ...)`.
- **UB** — Unified Buffer, the on-core scratchpad for vector ops.
  Default destination of `asc2.load(gm, ...)` when no `location=` is
  given. The full-row RMSNorm path requires that one row tile fits in
  UB (see [`rms_norm_f16.py`](../golden/kernels/rms_norm_f16.py)
  `UB_BUDGET_BYTES = 64 * 1024`).
- **L1** — a smaller on-core buffer between UB and L0; not used
  explicitly by any current golden.
- **L0A / L0B** — the cube-unit input buffers for matmul. The cube
  reads `A` from L0A and `B` from L0B, so the matmul golden uses
  `asc2.load(..., location=asc2.TileLocation.L0A)` and
  `... = asc2.TileLocation.L0B` (see
  [`matmul_f16.py`](../golden/kernels/matmul_f16.py)).

## §3 Compute partitioning

- **tile** — a sub-shape of the input that fits in a buffer and is
  processed atomically by a vector op. Element-wise goldens use
  `TILE_SIZE = 128` (or 64 for the `gelu/f32` golden's tanh path);
  matmul uses `m_tile × n_tile` cube tiles.
- **block** / **core** — one AI Core on the device. The `[CORE_NUM]`
  launch syntax dispatches `CORE_NUM` blocks; `asc2.block_idx()` is
  the per-launch block index. Element-wise goldens use `CORE_NUM = 16`;
  RMSNorm uses `CORE_NUM = 8`.
- **block grid** — a 2-D `(m_block, n_block)` grid over the output
  matrix; used by [`matmul_f16.py`](../golden/kernels/matmul_f16.py).
- **host dispatcher** — a Python-side function on the host that picks
  one of several `@asc2.jit` kernels before launch. Example:
  `rms_norm_launch(x, gamma, eps)` in
  [`rms_norm_f16.py`](../golden/kernels/rms_norm_f16.py) routes to
  `_full_row_launch` or `_split_d_launch` based on
  `num_cols * dtype_bytes <= UB_BUDGET_BYTES and num_cols % 8 == 0`.
- **full-row path** — a kernel variant where one entire row fits in
  UB and the kernel handles it in a single `asc2.load` /
  `asc2.reduce_sum` / `asc2.store` chain. Used by
  `rms_norm_full_row_kernel`.
- **split-row / split-D path** — a kernel variant that streams along
  the last axis in `tile_cols` chunks because the row exceeds UB.
  Used by `rms_norm_split_d_kernel`; the host pre-pads the input to a
  multiple of `tile_cols`.

## §4 Reduction vocabulary

- **reduce axis** — the axis along which a reduction collapses the
  shape. `reduce_axis=-1` means "reduce along the last axis";
  `reduce_sum` and `rms_norm` both reduce the last axis; matmul
  treats K as the reduction axis.
- **accumulator dtype** — the dtype the partial sums are held in. All
  current reducing/composed goldens hold the accumulator in `float32`
  regardless of input dtype:
  [`reduce_sum_f16.py`](../golden/kernels/reduce_sum_f16.py)
  promotes via `x_f32 = x.to(asc.float32)` before
  `asc2.reduce_sum(x_f32 * x_f32)`; the matmul cube accumulator is
  always `float32`.
- **identity** / **neutral element** — the value that does not
  change the accumulator (`0` for sum, `1` for product, `-inf` for
  max, `+inf` for min). Stored as a string in capabilities.yaml so
  YAML does not coerce `0` to int or `-inf` to a NaN literal.
  `reduce_sum` cells have `identity: "0"`; `softmax` has no single
  identity (max-stage identity is `-inf`, sum-stage identity is `0`)
  and stores `identity: null`.
- **finalization** — the post-reduction step that converts the
  accumulator into the output value (e.g. `mean = sum / N`,
  `inv_rms = 1 / sqrt(mean_sq + eps)`). RMSNorm uses
  `inv_rms = 1.0 / asc2.sqrt(sum_sq / num_cols + epsilon)`.

## §5 Tail vocabulary

- **padding** — extra elements appended to the last axis to satisfy
  an alignment requirement. `reduce_sum_f32` requires a 32-byte
  output last-dim, so the kernel writes 8 elements per row
  (`OUT_PAD = 8`) and the host slices the first column back; the
  `f16` variant uses `OUT_PAD = 16` because each element is half the
  width. The cell metadata's `padding` field records the
  element-count `OUT_PAD`, not the byte count.
- **tail** — the partial tile at the end of the last axis when the
  shape is not a multiple of `tile_size`. The current matrix's
  elementwise cells avoid tails entirely by listing shapes that are
  multiples of `TILE_SIZE` (e.g. 128 or 64).
- **mask** — an `asc2.mask` predicate that disables some lanes of a
  vector op for partial tiles. Reserved for Phase 5; no current cell
  uses it.
- **real_shape** — the `real_shape=...` argument to `asc2.load` /
  `asc2.store` that performs a partial load smaller than the tile
  shape. Reserved for Phase 5; no current cell uses it.
- **host zero-padding** — a host-side step that pads the input to a
  multiple of the kernel's tile size with zeros, then slices the
  output back. Used by `rms_norm_split_d` for inputs whose
  `num_cols` is not a multiple of `tile_cols=64`.

## §6 Cell metadata enums

These are the Phase 1 capability-cell metadata fields. All are
additive on `schema_version: "3"`; no consumer outside
[check_capabilities.py](../tests/tools/check_capabilities.py) reads
them today.

### `shape_regime`

How the kernel handles input shape variability.

- **`fixed`** — both `num_rows` and `num_cols` (or their per-tier
  equivalent) are compile-time constants for every shape the cell
  is verified at. Every elementwise / composed / softmax / matmul
  / reduce_sum cell in today's matrix is `fixed`: the test shapes
  are pinned and the kernel does not branch on shape.
- **`runtime_size_only`** — the shape is a runtime `int` but the
  kernel does not branch on it (it streams along the axis in
  fixed tiles). Reserved for future dynamic-shape coverage; not
  used by any current cell on its own (RMSNorm split_d uses it
  internally but the overall cell is `dynamic` because of the
  dispatcher).
- **`dynamic`** — multiple shape regimes routed by a host
  dispatcher. Today's only `dynamic` cell is `rms_norm/{f16,f32}`
  whose `rms_norm_launch` picks `full_row` vs `split_d`.

### `tail_behavior`

How the kernel handles the partial tile at the end of the last axis.

- **`aligned_only`** — the kernel assumes shape is a multiple of
  `tile_size`; no tail logic. Inputs that violate the alignment are
  out of scope and verification rejects them. Used by every
  elementwise / composed / softmax / matmul cell.
- **`host_pad`** — the host wraps the result with
  `asc2.full([1, OUT_PAD], s, dtype=row.dtype)` for 32-byte
  alignment and slices the output back. Used by
  `reduce_sum/{f16,f32}`.
- **`mask`** — the kernel uses `asc2.mask` on the vector path for
  partial tiles. Reserved for Phase 5; no current cell uses it.
- **`real_shape`** — the kernel uses
  `asc2.load(..., real_shape=...)` for partial loads. Reserved for
  Phase 5; no current cell uses it.
- **`host_dispatcher`** — the host picks between a full-tile kernel
  and a tail-aware kernel before launch. Used by `rms_norm` whose
  `rms_norm_launch` routes to full_row vs split_d.
- **`unsupported`** — the cell explicitly does not handle
  non-aligned tails; the unsupported case is in
  `unsupported_regimes`.

### `partitioning`

How work is distributed across cores.

- **`row_per_core`** — each block processes one or more rows in a
  `for row in asc2.range(asc2.block_idx(), num_rows, asc2.block_num())`
  loop. Used by `reduce_sum/{f16,f32}`, `softmax/f16`, and both
  RMSNorm sub-kernels.
- **`tile_per_core`** — each block processes one or more 1-D tiles
  in a flat iteration. Used by `abs/{f16,f32}`, `add/f16`,
  `gelu/{f16,f32}`, and `leaky_relu/f16`.
- **`block_grid`** — a 2-D `(m_block, n_block)` grid; each block
  handles one `(m_tile, n_tile)` cube tile. Used by `matmul/f16`.
- **`host_dispatcher`** — partition decided by the host before
  launch; the kernel itself uses one of the above patterns. Used
  by `rms_norm/{f16,f32}`.

### `unsupported_regimes`

Free-form list of regime slugs that this cell explicitly does *not*
cover. Each slug below is `currently_unsupported_in_this_cell` —
removing the entry (after adding the support) is the path forward,
not adding a contradictory new slug. See Phase 1 risks in
[`.cursor/plans/phase_1_spec_hygiene.plan.md`](../.cursor/plans/phase_1_spec_hygiene.plan.md).

Standard slugs (alphabetic):

- **`abl1_full`** — matmul variant where both A and B are buffered
  in L1 before being staged to L0A/L0B. Future MatMul branch.
- **`al1_full`** — matmul variant where A is buffered in L1.
  Future MatMul branch.
- **`bl1_full`** — matmul variant where B is buffered in L1.
  Future MatMul branch.
- **`dynamic_num_cols`** — `num_cols` is a runtime int and the
  kernel branches on it.
- **`dynamic_num_cols_not_8_aligned_in_full_row`** — RMSNorm
  full_row dispatcher rejects `num_cols` that are not multiples of
  8 (falls through to split_d).
- **`dynamic_num_rows`** — `num_rows` is a runtime int.
- **`k_tiled`** — matmul variant that tiles along the K axis;
  current matmul golden does not tile K.
- **`long_rows_exceeding_UB`** — softmax variant for rows that do
  not fit in UB. Today's softmax cell uses the full-row path only.
- **`multi_axis_reduction`** — reducing along more than one axis
  in a single kernel call; today's reduce_sum is last-axis only.
- **`non_16_multiple_shapes`** — matmul shapes whose `m`/`k`/`n`
  are not multiples of 16 (cube alignment).
- **`non_last_axis`** — reducing along an axis other than the last.
- **`num_cols_below_split_d_tile_threshold`** — RMSNorm split_d
  expects `num_cols >= tile_cols (64)`; smaller inputs would
  produce zero tiles after padding logic.
- **`split_row`** — softmax variant where a single row exceeds UB
  and must be split. Today's softmax cell uses the full-row path
  only.

When the metadata schema gains a new cell that adds support for one
of these regimes, the corresponding slug is removed from the parent
cell's `unsupported_regimes` list — the new capability becomes a
separate cell with its own metadata.

### `identity` encoding

Stored as a string so PyYAML's safe-load does not coerce `0` to
`int(0)` or `-inf` to `float('-inf')`. The current consumer
([check_capabilities.py](../tests/tools/check_capabilities.py)) does
string-only comparison; downstream consumers that want the numeric
value must `float(...)` explicitly. Allowed values: `"0"`, `"1"`,
`"-inf"`, `"+inf"`, or `null` (no single identity, e.g. softmax).

### `accumulator_dtype`

The dtype the partial sums are held in. `float32` for every
reduction/composed/matmul golden in this repo (cube accumulator,
reduce_sum promotion, RMSNorm sum-of-squares). `null` for
elementwise cells that do no accumulation.

### `reduce_axis`

The axis along which a reduction collapses the shape. `-1` for
every reducing op in this repo (reduce_sum, softmax, rms_norm) and
for matmul's `K` axis. `null` for non-reducing ops (abs, add, gelu,
leaky_relu).

## §7 Operator coverage notes

### ReLU scope (Phase 1.1 decision)

`relu` is **not** a standalone capability cell. It is covered by
pattern under the `abs` operation, whose `representative_of` list
already includes `relu` (along with `exp`, `log`, `sqrt`, `erf`,
`sin`, `cos`, `neg`, `ceil`, `floor`, `rsqrt`, `tanh`). Adding a
standalone `relu` cell would test the same Tier-0
template-substitution pattern (`asc2.load → single asc2 op →
asc2.store`) with no new information, so the cost (~+8 % nightly
budget per added cell) is not justified.

The path forward, if a future generative failure shows that the
`representative_of` claim does not hold for one of these operators,
is to add a *new* cell for that specific operator with its own
prompt and golden. Until then, the chip on the Tier-0 dashboard
implicitly covers ReLU; the `coverage_note` field on the
[`abs`](../capabilities.yaml) op records this decision.
