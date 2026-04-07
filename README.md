# pyasc-skill-stack

Reusable Agent skill modules for pyasc kernel development on Huawei Ascend NPUs.

With skills installed, a short prompt like _"Develop an abs operator for float16, shapes [1,128], [4,2048], [32,4096]"_ is enough — the agent handles environment setup, design, implementation, review, and verification autonomously.

## Target users

- Ascend NPU application developers
- pyasc operator developers
- Contributors who wish to extend the skill set

## Quick start

### Prerequisites

These must be available before starting:

- `opencode` CLI installed
- Python 3.10.x with `pyasc >= 1.1.1` and `torch`
- CANN Toolkit (see [docs/cann-setup.md](docs/cann-setup.md))
- Simulator libraries for `Ascend910B1`

### Step 1. Clone the repository

```bash
git clone git@github.com:aloschilov/pyasc-skill-stack.git
cd pyasc-skill-stack
```

### Step 2. Set up the CANN environment

```bash
source $HOME/Ascend/cann/set_env.sh
export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/tools/simulator/Ascend910B1/lib:$LD_LIBRARY_PATH
```

Quick check:

```bash
python3.10 -c "import asc; print('pyasc OK')"
python3.10 -c "import torch; print('torch OK')"
```

### Step 3. Start OpenCode

From the repository root:

```bash
opencode
```

Skill discovery is handled by `opencode.json` in the repo root — no additional installation step is needed.

Then give the agent a short prompt:

```text
Help me develop an abs operator that supports float16 data type.
The shape is mainly [1,128], [4,2048], [32,4096].
```

The agent will autonomously walk through environment check, design, implementation, review, and verification. Do not intervene unless the agent hits an external platform issue.

### Step 4. Verify the result

After the agent finishes, run the generated kernel manually:

```bash
source $HOME/Ascend/cann/set_env.sh
export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/tools/simulator/Ascend910B1/lib:$LD_LIBRARY_PATH
python3.10 teams/pyasc-kernel-dev-team/kernels/<kernel_name>/kernel.py -r Model -v Ascend910B1
```

## What the agent does

When skills are discovered correctly, the agent:

1. Loads `pyasc-codegen-workflow` and follows a 4-phase workflow (Phase 0 → 1 → 2 → 3)
2. Initializes the kernel project directory
3. Retrieves documentation from the golden set and API references
4. Writes a design document with API selection and syntax checks
5. Implements `kernel.py` using the kernel template and reviewed patterns
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
│   ├── pyasc-codegen-workflow/       # Core 4-phase workflow
│   ├── pyasc-api-patterns/           # API usage patterns
│   ├── pyasc-syntax-constraints/     # Supported syntax inside @asc.jit
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
│   ├── tutorials/                    # Golden reference kernels
│   ├── kernels/                      # Verified golden kernels (abs, sub, mul)
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
| `pyasc-api-patterns` | API usage patterns, dynamic tiling, `ConstExpr` guidance |
| `pyasc-syntax-constraints` | Python syntax support/restrictions inside `@asc.jit` |
| `pyasc-docs-search` | Local-first documentation and tutorial search |
| `pyasc-build-run-verify` | JIT compilation, simulator execution, output verification |
| `pyasc-code-review` | Code review against pyasc constraints |
| `pyasc-env-check` | Python, pyasc, CANN, torch environment checks |
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
source $HOME/Ascend/cann/set_env.sh
export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/tools/simulator/Ascend910B1/lib:$LD_LIBRARY_PATH
script -qc 'opencode run "Help me develop an abs operator that supports float16 data type. The shape is mainly [1,128], [4,2048], [32,4096]." --format json' /dev/null
```

The `script -qc` wrapper provides the pseudo-TTY that `opencode run` requires in headless environments.

## License

This project is a port of the [CANN Skills](https://gitcode.com/cann/skills) architecture. See the original project for license terms.
