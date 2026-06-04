# CI Gate Tiers

Three gate tiers ensure fast feedback on PRs while reserving expensive checks for merge and nightly runs.

## Tiers

| Tier | Trigger | Time budget | What runs |
|------|---------|-------------|-----------|
| **pr** | Every push / PR | < 30s | L1 unit tests + JIT verification of golden kernels |
| **merge** | Merge to main | < 5 min | PR tier + simulator execution of golden kernels |
| **nightly** | Scheduled (daily) | 15-30 min | Merge tier + L2 behavior + L3 agentic integration |

## Entry point

```bash
bash tests/ci-gate.sh --tier pr        # Fast PR gate
bash tests/ci-gate.sh --tier merge     # Merge gate (includes simulator)
bash tests/ci-gate.sh --tier nightly   # Full nightly run
```

## PR gate (`--tier pr`)

Runs in under 30 seconds. Suitable for pre-commit hooks and PR checks.

1. `run-tests.sh --fast` -- L1 structural and content validation (skills, agents, teams)
2. JIT verification of all golden kernels via `pytest_verify_kernel.py` -- confirms pyasc JIT compilation works without needing the simulator

No network, no simulator, no opencode required.

## Merge gate (`--tier merge`)

Runs in under 5 minutes. Requires CANN simulator environment.

1. Everything in PR gate
2. Simulator execution of all golden kernels via `run_and_verify.py --mode simulator` -- confirms numerical correctness with `np.testing.assert_allclose`

Requires: `source $HOME/Ascend/cann/set_env.sh` and `LD_LIBRARY_PATH` set. See [cann-setup.md](cann-setup.md).

## Nightly gate (`--tier nightly`)

Runs in 15-30 minutes. Requires opencode CLI and CANN simulator.

1. Everything in merge gate
2. `run-tests.sh --all` -- L2 behavior tests (agent trigger correctness, premature action detection) and L3 integration tests (full agent-in-the-loop kernel generation)

Requires: opencode CLI on PATH, CANN simulator environment.

## Perf gate (`perf-gate`, report-only)

A separate **nightly, non-blocking** GitHub Actions job (`continue-on-error:
true`) that measures perf-vs-AscendC for every demo cell and publishes the
result to the dashboard. It is **not** part of `ci-gate.sh`; it runs only on the
schedule / `workflow_dispatch tier=nightly`, alongside `nightly-gate`.

For each cell the harness ([tests/tools/demo_vector_ops.py](../tests/tools/demo_vector_ops.py)
`--all`) builds the canonical `ops-math`/`ops-nn` AscendC reference and the
generated pyasc kernel on the same `Ascend950PR_9599` camodel, then computes
`ratio = ref_ticks / gen_ticks`. The 0.70 gate is **reported, never enforced**,
so documented honest misses (`apply_adam` ~0.46, `batch_norm_v3` ~0.10) stay
green.

- **Image:** runs inside the docker_full perf image
  `ghcr.io/<owner>/pyasc-sim-perf:py3.11`, which extends `pyasc-sim` with the
  vendored `ops-math`/`ops-nn` reference repos, the `pyasc-v2-eval` tree, and
  the `dav_3510`/`Ascend950PR_9599` simulators. Built **manually** on the host
  that has the private clones via
  [docker/build-perf-image.sh](../docker/build-perf-image.sh)
  (`docker/build-perf-image.sh --push`); CI only `docker pull`s it.
- **Output:** `evidence/perf-vs-ascendc/*.json` + an aggregated
  `evidence/perf-summary.json` (via
  [tests/tools/perf/aggregate_perf.py](../tests/tools/perf/aggregate_perf.py)),
  uploaded as the `evidence-perf` artifact.
- **Commit + publish:** the single-writer `skills-value-report` job merges the
  `evidence-perf` artifact, re-aggregates, and commits `perf-summary.json` +
  `perf-vs-ascendc/*.json` to `main`. The `pages.yml` `evidence/**` trigger then
  redeploys the dashboard, whose perf panel renders `perf-summary.json` (falling
  back to each cell's curated `perf_ratio_demo` in `capabilities.yaml` when no
  measured summary is present, e.g. local/dev renders).
- **Compiler SIMD-fusion A/B (same job):** the `perf-gate` job also runs
  [tests/tools/demo_vf_fusion.py](../tests/tools/demo_vf_fusion.py) `--all`, which
  recompiles each generated kernel with `--cce-simd-vf-fusion` **off vs on** on
  the same camodel (verifying the "pyasc2 uses no micro-api; the compiler does
  fusion" positioning — see
  [docs/perf-vs-ascendc-demo.md](perf-vs-ascendc-demo.md)). It is also
  report-only.
  [tests/tools/perf/aggregate_vf_fusion.py](../tests/tools/perf/aggregate_vf_fusion.py)
  writes `evidence/vf-fusion-summary.json` (per-cell `ticks_off`/`ticks_on`/
  `fusion_speedup` + an `improved`/`neutral`/`regressed` verdict); both it and
  `evidence/vf-fusion/*.json` ride the same `evidence-perf` artifact and are
  committed by `skills-value-report`. The dashboard renders them in a **Compiler
  SIMD fusion** panel.

## Environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `PYASC_PYTHON` | `python3.10` | Python interpreter with pyasc |
| `ASCEND_HOME_PATH` | (from set_env.sh) | CANN toolkit root |
| `LD_LIBRARY_PATH` | (must include simulator) | Simulator libraries |
| `NODE_TLS_REJECT_UNAUTHORIZED` | `0` (for opencode) | Bypass TLS issues |
| `PYASC_PERF_IMAGE` | `ghcr.io/<owner>/pyasc-sim-perf:py3.11` | docker_full perf image (perf-gate) |
| `OPS_MATH_HOME` / `OPS_NN_HOME` | `/opt/ops-math` / `/opt/ops-nn` (in perf image) | Canonical AscendC reference repos |

## Exit codes

- `0` -- all checks passed
- `1` -- one or more checks failed
- `2` -- environment prerequisites missing (e.g., simulator not available for merge tier)
