# Nightly post-mortem 2026-05-07: 5/12 generative regressions after C310 migration

This is the post-mortem for nightly run
[25475895458](https://github.com/aloschilov/pyasc-skill-stack/actions/runs/25475895458),
which ran 2026-05-07 04:20–09:02 UTC. It was the first scheduled nightly
after [`d7dfbcb`](https://github.com/aloschilov/pyasc-skill-stack/commit/d7dfbcb)
flipped the default platform from `Ascend910B1` to `Ascend950PR_9599`.

## Headline

- **Result**: 7/12 generative cells passed (above the 50% nightly gate
  threshold, so the gate stayed green and evidence was committed
  automatically by `sync_capabilities.py`).
- **Regression vs. previous nightly**: 5 cells flipped from `confirmed`
  to `pending`: `abs/float32`, `add/float16`, `gelu/float16`,
  `gelu/float32`, `leaky_relu/float16`.
- **Goldens**: unaffected. `merge-gate` (which exercises all 10 goldens
  with `TIMEOUT=1500s`) finished green in 21 minutes.
- **Root causes**: two distinct ones, summarised below.

## Failure breakdown

| Cell | Failure mode | Root cause |
|---|---|---|
| `abs/float32` | Simulator timeout (300 s budget) on attempts 1-3 | C310 camodel ~80 s/launch + 3 shapes × 16 cores |
| `add/float16` | Simulator timeout (300 s) on attempts 1-3 | Same as above; `[32, 4096]` shape is too big for budget |
| `gelu/float16` | Simulator timeout (600 s) on attempts 1-3 | Same as above + `asc2.erf` lowering cost |
| `gelu/float32` | Codegen `AttributeError: 'Tile' object has no attribute '__rmul__'` | opencode wrote `GELU_C * x_cubed` (scalar on left) |
| `leaky_relu/float16` | Simulator timeout (300 s) on attempts 1-3 | Same as the four-class above |

The other 7 cells passed:
- `abs/float16`, `reduce_sum/float16`, `reduce_sum/float32`, `softmax/float16`,
  `matmul/float16`, `rms_norm/float16`, `rms_norm/float32`.

## Root cause 1: nightly Docker timeout was sized for `Ascend910B1`

Before the platform migration, the camodel for 910B1 was fast enough that
elementwise cells completed inside `DOCKER_TO=300 s` and `gelu`/`softmax`/
`matmul` inside `600 s` ([ci.yml lines 165–169](../../.github/workflows/ci.yml)
prior to fix). The C310 camodel is markedly slower — our JIT-vs-tick probe
(see [ticks-calculation.md](ticks-calculation.md)) measured ~80 s host
wall per single-launch `abs(x)` over 8 cores at `[size=8192]`. The agent
kernels run three test shapes (`[1,128]`, `[4,2048]`, `[32,4096]`) with
`CORE_NUM=16`, `TILE_SIZE=128`, so the per-cell wallclock is 3 × shape ×
launch ≈ 5–10 minutes — **above the 300 s/600 s budget but well below
the merge-gate's `1500 s`**.

**Fix.** Bump the nightly's `DOCKER_TO` to `1500 s` for every op (matches
the merge-gate budget). Single uniform timeout, no per-op special cases.

```diff
-            DOCKER_TO=300
-            if [ "$op" = "softmax" ] || [ "$op" = "gelu" ] || [ "$op" = "matmul" ]; then
-              DOCKER_TO=600
-            fi
+            # Uniform 1500 s docker timeout (matches merge-gate post-C310 migration).
+            DOCKER_TO=1500
```

This is the only nightly-runtime change needed; the `--timeout 420 s`
opencode budget is unchanged because opencode itself is independent of
the simulator path.

## Root cause 2: `Tile.__rmul__` not enforced in the `gelu/float32` prompt

Looking at the agent kernel for `gelu/float32` ([archived in run
artifacts](https://github.com/aloschilov/pyasc-skill-stack/actions/runs/25475895458/artifacts)):

```python
inner = (x + GELU_C * x_cubed) * GELU_K
```

`GELU_C * x_cubed` puts the Python scalar on the left. `float.__mul__(Tile)`
returns `NotImplemented`, Python tries `Tile.__rmul__(0.044715)`, but the
asc2 `Tile` class does not implement `__rmul__` → codegen raises
`AttributeError: 'Tile' object has no attribute '__rmul__'`.

The `SKILL.md` "Common Mistakes" section already had a row for this in
the build-run-verify skill, but the `gelu/float32` prompt didn't reinforce
it. Our golden (`golden/kernels/gelu_f32.py:54-55`) does it correctly:
`(x + x * x * x * GELU_C) * GELU_K`, `x * (asc2.tanh(inner) + 1) * 0.5`.

**Fix.**
1. Promote the rule to a top-level "Common Mistakes" entry in
   `skills/pyasc-api-patterns/SKILL.md` with a code-pointer to
   `gelu_f32.py:54-55`.
2. Add an explicit `CRITICAL: tile-on-left` rule to the `gelu/float32`
   default and guided prompts in `capabilities.yaml`.

## What worked (so we don't break it)

- The 50% nightly-gate threshold + `sync_capabilities.py` self-healing
  produced an honest `pending` state for the 5 regressions and committed
  the evidence — no babysitting required.
- `merge-gate` correctly stayed green; goldens are not affected.
- `rms_norm/{f16,f32}` (the most complex cell, with two kernel variants
  per cell) passed cleanly on the first attempt — proving the host-side
  dispatcher and `torch.Tensor` contract survived the platform flip.

## Verification plan

1. Apply both fixes (timeout bump + prompt hardening).
2. Trigger nightly via `gh workflow run CI -f tier=nightly` so the next
   run is not 24h away.
3. Expect `>=11/12` to pass. If `gelu/float32` still fails, the prompt
   hardening was insufficient and we should pin a reference implementation
   into the skill discovery layer.
4. Update `capabilities.yaml` `generative_status` for the 5 cells back
   to `confirmed` is automatic via `sync_capabilities.py` once their
   evidence files report `runtime: pass`.

## Why this didn't catch in the migration's pre-flight

The pre-flight (W0 in the migration plan, see prior chat) only verified
the 7 currently-910B1 **goldens** on C310, not the **generative**
opencode-driven cells. Goldens use bespoke driver code (e.g. small test
shapes, `CORE_NUM=8` for `rms_norm`); generative cells use whatever the
agent emits, which on 910B1 happened to be `CORE_NUM=16`, `TILE_SIZE=128`,
`[32,4096]` — fine for the older simulator, too heavy for C310 at the
nightly's old timeouts.

**Future-proofing.** When changing simulator platform defaults, also run
one `collect_generative_evidence.py --op X --dtype Y --runtime
--docker-timeout 1500` on a representative elementwise cell as part of
the pre-flight — that would have caught this in seconds.
