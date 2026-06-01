# Phase 11 blocker — generated (pyasc) side: multi-input + reduction kernels

> **RESOLVED in Phase 11b (2026-06-01).** This blocker no longer holds. The
> notes below are kept for history. Current state:
>
> | cell               | ref_ticks | gen_ticks | ratio | gate |
> |--------------------|-----------|-----------|-------|------|
> | abs/float16        | 4349      | 4690      | 0.93  | PASS |
> | add/float16        | 4281      | 6304      | 0.68  | FAIL (honest perf miss) |
> | reduce_sum/float32 | 8328      | 5106      | 1.63  | PASS |
>
> **Two findings retired the blocker:**
> 1. **The host codegen segfault did not reproduce in Phase 11b.** On the same
>    built extension, a two-load probe ran 5/5 and add + reduce_sum gen runners
>    ran 6/6 — 11/11 clean codegen cycles. No `pyasc-v2-eval` source patch was
>    needed; the segfault seen in Phase 11 did not survive the environment
>    refresh. Refs and gen therefore share one host camodel (Docker fallback not
>    needed).
> 2. **The last `BLOCKED` cell (reduce_sum) was a demo-harness bug, not a
>    toolchain fault.** The live launch wrapper `reduce_sum_launch(x,
>    out_pad=OUT_PAD)` has a defaulted scalar param; the gen-runner probe passed
>    an ndarray for `out_pad`, so the kernel ran a no-op (15 ticks). Fixed in
>    `tests/tools/perf/pyasc_gen_runner.py` (probe now supplies only required,
>    non-defaulted positional params). See `docs/perf-vs-ascendc-demo.md`.
>
> All three kernels were live-regenerated (opencode 1.15.10 + dashscope/glm-5,
> `oracle_guided`, attempt-1 pass; archived under `regen-archive/`).

---

**Historical status (Phase 11):** abs/f16 gate PASSES end-to-end (ratio 0.93).
add/f16 and reduce_sum/f32 were blocked on the **generated side only** by a
pyasc-v2-eval codegen/runtime fault believed independent of the skill-stack
kernels.

## What works
| cell              | ref_ticks (canonical AscendC) | gen_ticks (pyasc) | ratio | gate |
|-------------------|-------------------------------|-------------------|-------|------|
| abs/float16       | 4346 (median of 3)            | 4689              | 0.93  | PASS |
| add/float16       | 4285 (median of 3)            | — (BLOCKED)       | —     | —    |
| reduce_sum/float32| 8330 (median of 3)            | — (BLOCKED)       | —     | —    |

The **canonical ops-math reference compiles and runs on camodel for all three
cells** (no fabrication; `ascendc_ref_runner.py`, evidence under
`evidence/perf/ascendc-ref/`). The blocker is purely the pyasc generated side.

## The fault (isolated)
Running any generated pyasc kernel that loads **two global tensors** (e.g.
`add`) or contains a **reduction `for`-loop** (`reduce_sum`) on the host
`pyasc-v2-eval` build crashes during codegen:

```
Fatal Python error: Segmentation fault
  File ".../asc/language/core/dtype.py", line 89 in to_ir
  File ".../asc/language/.../{block_idx|load}
  File ".../asc/codegen/function_visitor.py", line ... (visit_BinOp / compute_inout/visit_For)
  File ".../asc/runtime/jit.py", line 195 in _run_codegen
```

The crash is in the C++ MLIR builder reached via `dtype.to_ir() ->
global_builder.get_ir_builder()`.

### Minimal isolation (single-launch, fresh JIT cache each run)
| kernel body                              | result                         |
|------------------------------------------|--------------------------------|
| 1 load + `asc2.abs(x)` (unary)           | OK — 16 block_end, Total tick  |
| 1 load + `x + x` (binary, one tensor)    | OK — 16 block_end, Total tick  |
| **2 loads** (`x=load; y=load; store x`)  | **segfault in codegen**        |
| 2 loads + `x + y` (== add)               | segfault in codegen            |
| 2 loads + `asc2.add(x, y)` (call form)   | segfault in codegen            |
| reduce_sum (1 load + reduce for-loop)    | segfault in `compute_inout`    |

The trigger is **two global loads** (and, separately, the reduction-loop
pre-pass), NOT the arithmetic operator and NOT the skill-stack kernel source.

### Not a kernel-authoring bug — the eval's own test crashes identically
`pyasc-v2-eval/python/test/asc2/kernels/test_vadd.py` (the upstream two-input
vadd test) **also segfaults (exit 139)** on this host with the same
`to_ir`/`_run_codegen` stack. Both the golden (`golden/kernels/add_f16.py`,
TILE=128) and the oracle_guided kernel (TILE=2048) crash the same way.

### Docker (`pyasc-sim:latest`) is not a usable fallback here
The image ships a *different* asc API (`asc2.GlobalAddress` absent) and its
camodel runtime aborts with `terminate: std::out_of_range _Map_base::at` —
**even for abs**. So Docker cannot produce gen ticks for any cell currently.

### Why P6 didn't catch this
The P6 generative evidence
(`evidence/{abs,add,reduce_sum}-...-p6.json`) records
`verification.status = pass` but `shapes_verified = []` and carries **no
camodel/tick data** — P6 was a semantic/generation check, never a simulator
run. Phase 11 is the first time these kernels are launched on camodel, which is
why the latent two-input/reduction codegen fault surfaces now.

## Reproduce
```bash
cd /home/aloschilov/workspace/pyasc-v2-eval
source /usr/local/Ascend/cann-9.0.0/set_env.sh
export PYTHONPATH=$PWD/python:$PYTHONPATH
export LD_LIBRARY_PATH=/usr/local/Ascend/cann-9.0.0/tools/simulator/Ascend950PR_9599/lib:$LD_LIBRARY_PATH
cd python/test/asc2 && python3.11 -m pytest kernels/test_vadd.py   # -> Segmentation fault (139)
```

## Options (need a decision)
1. **Ship abs/f16 as the proven gate cell** (real 0.93), keep add/reduce_sum as
   *reference-captured + gen-blocked*, documented honestly. Lowest risk.
2. **Fix/rebuild the host pyasc-v2-eval** (the `to_ir`/`get_ir_builder` C++
   lifetime issue for multi-load + reduction codegen) and re-run 11.4. Highest
   fidelity, unknown effort (upstream C++).
3. **Repair the Docker sim image** (API + `std::out_of_range`) and run the gen
   side there. Medium effort, separate environment from the references.

No gen_ticks were fabricated for add/reduce_sum, per the plan's
canonical-only / surface-blockers decision.
