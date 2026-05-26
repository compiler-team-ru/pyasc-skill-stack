# pyasc-skill-stack

Reusable Agent skill modules for pyasc kernel development on Huawei Ascend NPUs using the **asc2** high-level tile-based API.

With skills installed, a short prompt like _"Develop an abs operator for float16, shapes [1,128], [4,2048], [32,4096]"_ is enough — the agent handles environment setup, design, implementation, review, and verification autonomously.

## Target users

- Ascend NPU application developers
- pyasc operator developers
- Contributors who wish to extend the skill set

## Quick start

### Prerequisites

These must be available on the host before starting:

- `opencode` CLI installed
- CANN Toolkit installed somewhere on disk (see [docs/cann-setup.md](docs/cann-setup.md))
- Simulator libraries for `Ascend950PR_9599` (shipped with CANN — the only platform the stack targets)
- Either: Python 3.10.x (for the local-build path) **or** Docker (for the containerized path)

> **Note on `asc2`.** The `asc2` tile-based API is not yet published to PyPI — the PyPI `pyasc` wheel ships only the `asc` v1 API. To get `asc2` you currently need either the Docker image (Option A) or a build from the `v2` branch of the `pyasc` source (Option B).

> **Architecture support.** Tested on Ubuntu 22.04 on both `x86_64` and `aarch64`. The Docker image (Option A) is multi-arch and auto-selects the right LLVM prebuilt at build time. For Option B (local build), use the LLVM archive matching `uname -m` (the snippet below does this for you). CANN itself must be installed from its arch-specific installer (`linux-x86_64.run` vs `linux-aarch64.run`); see [docs/cann-setup.md](docs/cann-setup.md).

### Step 1. Clone the repository

```bash
git clone git@github.com:aloschilov/pyasc-skill-stack.git
cd pyasc-skill-stack
```

### Step 1b. Install host prerequisites (Ubuntu, optional)

If you're starting from a clean Ubuntu 22.04 / 24.04 host (x86_64 or aarch64), the repo ships a small idempotent helper that checks for and offers to install everything needed for Options A and B:

```bash
bash scripts/install-host-deps.sh                              # interactive: shows plan, asks once
bash scripts/install-host-deps.sh --yes                        # non-interactive (CI)
bash scripts/install-host-deps.sh --skip-docker --skip-opencode  # apt packages only
```

Covers: apt packages (`build-essential`, `git`, `curl`, `ccache`, `clang`, `lld`, `python3.10`, `python3.10-venv`, `python3-pip`), Docker Engine via the official `download.docker.com` repo, and the `opencode` CLI via the upstream installer.

Out of scope for the script: the CANN Toolkit itself (see [docs/cann-setup.md](docs/cann-setup.md)) and any pyenv-managed Python.

### Step 2. Configure paths (once per shell)

The repo does not assume a fixed location for CANN or for your `pyasc` checkout — both are expressed as environment variables with sensible defaults:

```bash
export CANN_HOME="${CANN_HOME:-$HOME/Ascend/cann}"
export PYASC_SRC="${PYASC_SRC:-$HOME/workspace/pyasc}"
```

Common CANN install locations you may need to point `CANN_HOME` at:

- `$HOME/Ascend/cann` (default user install)
- `$HOME/Ascend/ascend-toolkit/latest`
- `$HOME/Ascend/cann-<version>` (e.g. `cann-9.0.0`)
- `/usr/local/Ascend/ascend-toolkit/latest` (system-wide install, and the location used inside the Docker image)

`PYASC_SRC` is only needed for Options B and C below.

### Step 3. Install `pyasc` (with `asc2`) — choose one option

#### Option A — Docker (self-contained, recommended for a clean machine)

The default `docker/Dockerfile` builds `pyasc` from the `v2` source inside the image; no pyasc checkout is required on the host.

```bash
docker build -f docker/Dockerfile -t pyasc-sim:latest docker/
docker run --rm -it \
  --ulimit nofile=1048576:1048576 \
  -v "$(pwd)":/workspace -w /workspace \
  pyasc-sim:latest
```

> The `--ulimit nofile=1048576` is required: the CANN simulator opens hundreds of dump files per core (16 cores × tens of files), and the default Docker `nofile` limit of 1024 causes runtime errors like `Failed opening file ... Too many open files`.

All remaining steps (sourcing CANN, running `opencode`, running kernels) happen **inside the container**. Inside the container, `CANN_HOME` is preset (CANN 9.0.0-beta.2 lives under `/usr/local/Ascend/cann-9.0.0-beta.2`; the install also exposes `/usr/local/Ascend/ascend-toolkit/latest` as a symlink, and sourcing `set_env.sh` updates `$ASCEND_HOME_PATH` accordingly).

An advanced overlay-based variant is also provided for developers who want to graft their local `pyasc` build onto a prebuilt CANN image:

```bash
docker build -f docker/Dockerfile.overlay -t pyasc-sim:overlay docker/
```

See the header of [`docker/Dockerfile.overlay`](docker/Dockerfile.overlay) for when and why to use this variant.

#### Option B — Local build from `pyasc` source (recommended for developers)

Requires CANN and a C++ toolchain on the host. Clone `pyasc` anywhere (`$PYASC_SRC` tells the rest of the workflow where it lives) and build it into a venv:

```bash
git clone https://gitcode.com/cann/pyasc.git "$PYASC_SRC"
( cd "$PYASC_SRC" && git fetch origin +refs/merge-requests/85/head:v2 && git checkout v2 )

python3.10 -m venv .venv
source .venv/bin/activate

LLVM_ARCHIVE_BASE=https://oaitriton.blob.core.windows.net/public/llvm-builds
case "$(uname -m)" in
    x86_64)  LLVM_URL="${LLVM_ARCHIVE_BASE}/llvm-86b69c31-ubuntu-x64.tar.gz"   ;;
    aarch64) LLVM_URL="${LLVM_ARCHIVE_BASE}/llvm-86b69c31-ubuntu-arm64.tar.gz" ;;
    *) echo "Unsupported arch: $(uname -m)" >&2 ; return 1 ;;
esac
mkdir -p "$HOME/.llvm"
curl -fsSL "$LLVM_URL" | tar -xz --strip-components=1 -C "$HOME/.llvm"
export LLVM_INSTALL_PREFIX="$HOME/.llvm"

pip install "$PYASC_SRC"
```

See the upstream [`pyasc` build-from-source guide](https://gitcode.com/cann/pyasc) (file `docs/installation/build-from-source.rst`) for both prebuilt LLVM archive URLs and optional build flags (`PYASC_SETUP_CCACHE`, `PYASC_SETUP_CLANG_LLD`, etc.).

#### Option C — Shortcut: existing system-wide `pyasc` dev install

If `asc2` is already importable from the base `python3.10` (for example, you maintain a `pip install -e "$PYASC_SRC"` globally), create a venv that inherits system site-packages:

```bash
python3.10 -m venv --system-site-packages .venv
source .venv/bin/activate
```

This is the fastest path but depends entirely on your pre-existing Python environment — prefer A or B for a reproducible setup.

### Step 4. Source CANN and start OpenCode

Same commands for all three options (for Docker, run them inside the container — `CANN_HOME` is preset there):

```bash
source "$CANN_HOME/set_env.sh"
export LD_LIBRARY_PATH="$ASCEND_HOME_PATH/tools/simulator/Ascend950PR_9599/lib:$LD_LIBRARY_PATH"
python3.10 -c "import asc, asc2; print('pyasc OK')"
opencode
```

Skill discovery is handled by `opencode.json` in the repo root — no additional installation step is needed.

Then give the agent a short prompt:

```text
Help me develop an abs operator that supports float16 data type.
The shape is mainly [1,128], [4,2048], [32,4096].
```

The agent will autonomously walk through environment check, design, implementation, review, and verification. Do not intervene unless the agent hits an external platform issue.

### Step 5. Verify the result

After the agent finishes, run the generated kernel manually:

```bash
source "$CANN_HOME/set_env.sh"
export LD_LIBRARY_PATH="$ASCEND_HOME_PATH/tools/simulator/Ascend950PR_9599/lib:$LD_LIBRARY_PATH"
python3.10 teams/pyasc-kernel-dev-team/kernels/<kernel_name>/kernel.py -r Model -v Ascend950PR_9599
```

## What the agent does

When skills are discovered correctly, the agent:

1. Loads `pyasc-codegen-workflow` and follows a 4-phase workflow (Phase 0 → 1 → 2 → 3)
2. Initializes the kernel project directory
3. Retrieves documentation from the golden set and asc2 kernel references
4. Writes a design document with asc2 API selection and syntax checks
5. Implements `kernel.py` using the asc2 kernel template and reviewed patterns
6. Conducts self-review and acceptance review against pyasc constraints
7. Runs verification (simulator or JIT fallback) and writes a verification record
8. Delivers a runnable kernel with all workflow artifacts

## What the agent should not require from you

If skills are loaded correctly, the agent should not:

- ask you to manually read internal skill files
- ask you to create `design.md`, `kernel.py`, or review documents
- ask you to run internal phase scripts
- require a long "management" prompt with all phases listed
- stop at the first error without attempting to fix it

If the agent requires any of the above, there is likely a problem with skill discovery or the `AGENTS.md` configuration.

## Project structure

```
pyasc-skill-stack/
├── skills/                           # Skill modules (agent reads these)
│   ├── pyasc-codegen-workflow/       # Core 4-phase workflow (asc2)
│   ├── pyasc-api-patterns/           # asc2 API usage patterns
│   ├── pyasc-syntax-constraints/     # Supported syntax inside @asc2.jit
│   ├── pyasc-docs-search/            # Documentation index
│   ├── pyasc-build-run-verify/       # Build, run, verification
│   ├── pyasc-code-review/            # Code review checklist
│   ├── pyasc-env-check/              # Environment verification
│   └── pyasc-task-focus/             # Task tracking
├── teams/
│   └── pyasc-kernel-dev-team/
│       ├── AGENTS.md                 # Team agent definition
│       ├── quickstart.md             # Manual development guide
│       └── kernels/                  # Generated kernel workspace
├── golden/
│   ├── tutorials/                    # Golden reference tutorial (asc2 vadd)
│   ├── kernels/                      # Golden kernels per tier (abs, reduce_sum, gelu, leaky_relu, softmax)
│   ├── archive/                      # Archived golden kernels (sub, mul — covered by tier 0)
│   └── docs/                         # Local pyasc API documentation
├── tests/                            # Automated test pyramid
│   ├── run-tests.sh                  # Test runner
│   ├── ci-gate.sh                    # CI entry point (pr/merge/nightly)
│   ├── unit/                         # L1: structure/content checks
│   ├── behavior/                     # L2: trigger/action checks
│   └── integration/                  # L3: end-to-end workflow
├── opencode.json                     # Skill discovery config
└── docs/
    └── cann-setup.md                 # CANN environment setup guide
```

## Skills library

| Skill | Purpose |
|-------|---------|
| `pyasc-codegen-workflow` | 4-phase workflow: environment → design → implementation + review → verification |
| `pyasc-api-patterns` | asc2 API usage patterns, tiling with ceildiv, `ConstExpr` guidance |
| `pyasc-syntax-constraints` | Python syntax support/restrictions inside `@asc2.jit` |
| `pyasc-docs-search` | Local-first documentation and tutorial search |
| `pyasc-build-run-verify` | JIT compilation, simulator execution, output verification |
| `pyasc-code-review` | Code review against pyasc constraints |
| `pyasc-env-check` | Python, pyasc, CANN, numpy environment checks |
| `pyasc-task-focus` | Task focus and attention management |

## Testing

```bash
# Quick PR gate (L1 + JIT verification, < 60s)
bash tests/ci-gate.sh --tier pr

# Full test suite
bash tests/run-tests.sh --all
```

See [tests/README.md](tests/README.md) for details.

## Headless run (CI / scripted)

For non-interactive verification, use `opencode run` with a pseudo-TTY wrapper:

```bash
source "$CANN_HOME/set_env.sh"
export LD_LIBRARY_PATH="$ASCEND_HOME_PATH/tools/simulator/Ascend950PR_9599/lib:$LD_LIBRARY_PATH"
script -qc 'opencode run "Help me develop an abs operator that supports float16 data type. The shape is mainly [1,128], [4,2048], [32,4096]." --format json' /dev/null
```

The `script -qc` wrapper provides the pseudo-TTY that `opencode run` requires in headless environments.

## How this matrix is evaluated

The pyasc-skill-stack matrix is an evidence-backed **capability dashboard** and an **OpenCode skills-intervention experiment**. It compares OpenCode runs with pyasc skills enabled against OpenCode runs with pyasc skills disabled. The score belongs to the full generation protocol — task cell, prompt variant, OpenCode harness, model/profile, allowed context, skill mode, budget, and evaluator — not to a bare model.

This is **not** a "model vs skills" comparison. The model is one held-constant variable inside the protocol, alongside the prompt variant, timeout, max attempts, evaluator, and allowed filesystem. The two legs of the comparison are `OpenCode + pyasc-skills` and `OpenCode without pyasc-skills`. Guided prompts, golden references, oracle workarounds, and human hints are separate interventions and are labeled separately rather than folded into the skills effect.

The CI matrix in [`.github/workflows/ci.yml`](.github/workflows/ci.yml) is *designed* as a paired skills-on/off intervention. A per-pair validity classifier in [`tests/tools/compare_skills_value.py`](tests/tools/compare_skills_value.py) checks whether each off-leg actually produced evidence comparable to its on-leg counterpart; off-legs flagged as instrumentation/configuration failures (no resolved model, zero tokens, no artifact) are excluded from the headline pass-rate so the dashboard does not imply a clean 0% no-skills baseline when none was measured. The full contract — protocol taxonomy (P0..P8), failure taxonomy (F1..F13), evidence schema v4 (proposed), and baseline-validity rules — lives in [docs/evaluation-methodology.md](docs/evaluation-methodology.md).

## Capabilities dashboard

A live view of the capabilities matrix is published automatically on every push to `main`:

**[https://aloschilov.github.io/pyasc-skill-stack/](https://aloschilov.github.io/pyasc-skill-stack/)**

The matrix is organized around four **complexity tiers** that represent genuinely distinct generative challenges, rather than listing every elementwise operation individually:

| Tier | Name | What it tests |
|------|------|--------------|
| 0 | Elementwise | Template substitution: `load` -> single `asc2.*()` call -> `store` |
| 1 | Reduction | Output shape differs from input; accumulation logic |
| 2 | Composed | No single asc2 builtin; agent must compose multiple API calls |
| 3 | Advanced | Multi-dimensional tiling, accumulator management, placement |

Each tier has representative operations with prompts. Proving a representative (e.g. `abs` for unary elementwise) gives high confidence that structurally identical operations (e.g. `exp`, `log`, `sqrt`) can also be generated.

The dashboard is generated from `capabilities.yaml` and `evidence/*.json` by `tests/tools/generate_dashboard.py` and deployed via GitHub Pages. Click any status badge to see evidence details and the prompt used.

## License

This project is a port of the [CANN Skills](https://gitcode.com/cann/skills) architecture. See the original project for license terms.
