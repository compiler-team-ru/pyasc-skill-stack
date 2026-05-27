# OpenCode skills intervention (skills-on vs skills-off)

Generated 2026-05-27 12:31:44Z from 30 comparison cell(s). The CI matrix is an intended paired OpenCode skills-on/off intervention (same harness, model, prompt, budget, evaluator); a per-pair validity classifier determines whether each off-leg is a comparable baseline. See `docs/evaluation-methodology.md` for the contract.

## Per-profile summary (clean baseline available)

| Profile | Cells compared | Pass-rate on | Pass-rate off (clean) | Off-leg validity | Tokens Δ (avg) | Cost Δ (avg, USD) | Elapsed Δ (avg) | Viability unlocked (clean) | Unresolved (off infra-failed) |
|---|---|---|---|---|---|---|---|---|---|
| `cloud-default` | 12/12 | 67% | 50% | 12 ok | +98809.7 | 0.0 | +32.6s | 3/12 | 0/0 |
| `local-llama-3.1-8b` | 8/8 | 0% | 0% | 8 ok | +228.6 | 0.0 | +97.0s | 0/8 | 0/0 |
| `local-qwen-coder-7b` | 9/10 | 0% | 0% | 9 ok | -45.6 | 0.0 | -16.9s | 0/9 | 0/0 |

## Per-cell deltas (skills-on minus skills-off)

| Op | dtype | Profile | Off validity | Quality Δ | Tokens Δ | Elapsed Δ | Attempts Δ | Viability unlocked (clean) |
|---|---|---|---|---|---|---|---|---|
| abs | float16 | `cloud-default` | ok | -1 | +206,983 | +103.7s | 0 | no |
| abs | float16 | `local-llama-3.1-8b` | ok | 0 | +331 | +111.7s | 0 | no |
| abs | float16 | `local-qwen-coder-7b` | ok | 0 | +4 | -5.5s | 0 | no |
| abs | float32 | `cloud-default` | ok | 0 | +214,490 | +125.1s | 0 | no |
| abs | float32 | `local-llama-3.1-8b` | ok | 0 | -720 | -152.0s | 0 | no |
| abs | float32 | `local-qwen-coder-7b` | ok | 0 | -38 | -29.0s | 0 | no |
| add | float16 | `cloud-default` | ok | +1 | +28,873 | -130.6s | -1 | yes |
| add | float16 | `local-llama-3.1-8b` | ok | 0 | +341 | +120.8s | 0 | no |
| add | float16 | `local-qwen-coder-7b` | ok | 0 | -132 | -28.0s | 0 | no |
| gelu | float16 | `cloud-default` | ok | 0 | +242,517 | +148.9s | 0 | no |
| gelu | float16 | `local-llama-3.1-8b` | ok | 0 | -249 | -19.6s | 0 | no |
| gelu | float16 | `local-qwen-coder-7b` | ok | 0 | -209 | -47.4s | 0 | no |
| gelu | float32 | `cloud-default` | ok | 0 | +179,881 | +435.0s | 0 | no |
| gelu | float32 | `local-llama-3.1-8b` | ok | 0 | +35 | +51.9s | 0 | no |
| gelu | float32 | `local-qwen-coder-7b` | ok | 0 | +98 | +9.4s | 0 | no |
| leaky_relu | float16 | `cloud-default` | ok | +1 | -192 | -155.5s | -2 | yes |
| leaky_relu | float16 | `local-llama-3.1-8b` | ok | 0 | +305 | +124.3s | 0 | no |
| leaky_relu | float16 | `local-qwen-coder-7b` | ok | 0 | -329 | -83.1s | 0 | no |
| matmul | float16 | `cloud-default` | ok | +1 | +33,703 | -89.8s | -2 | yes |
| reduce_sum | float16 | `cloud-default` | ok | 0 | +116,785 | +106.0s | 0 | no |
| reduce_sum | float16 | `local-llama-3.1-8b` | ok | 0 | +1,298 | +383.8s | 0 | no |
| reduce_sum | float16 | `local-qwen-coder-7b` | ok | 0 | +358 | +79.4s | 0 | no |
| reduce_sum | float32 | `cloud-default` | ok | 0 | +44,122 | +86.3s | 0 | no |
| reduce_sum | float32 | `local-llama-3.1-8b` | ok | 0 | +488 | +155.4s | 0 | no |
| reduce_sum | float32 | `local-qwen-coder-7b` | ok | 0 | -205 | -47.0s | 0 | no |
| rms_norm | float16 | `cloud-default` | ok | 0 | +60,724 | +0.0s | 0 | no |
| rms_norm | float32 | `cloud-default` | ok | 0 | +14,707 | -191.6s | -1 | no |
| softmax | float16 | `cloud-default` | ok | 0 | +43,123 | -46.0s | -1 | no |
| softmax | float16 | `local-qwen-coder-7b` | ok | 0 | +43 | -1.0s | 0 | no |

### Cells with missing pair (one of on/off absent)

- matmul/float16/`local-qwen-coder-7b` — available: on
