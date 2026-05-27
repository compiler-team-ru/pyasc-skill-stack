# CANN Setup for pyasc-skill-stack

## Required Software

| Component | Version | Notes |
|-----------|---------|-------|
| CANN Toolkit | >= 8.5.0 (community or standard) | Ascend computing backend |
| Python | 3.10.x | pyasc and torch are installed here |
| pyasc | >= 1.1.1 | Python JIT for Ascend C |
| pytest | >= 7.0 | Required for pyasc kernel integration tests |
| numpy | < 2.0 | Required by pyasc |
| torch | any | For tensor management and verification |

## Architecture support

The stack is verified on Ubuntu 22.04 on both `x86_64` and `aarch64`. CANN
ships a separate installer per architecture; pick the one matching
`uname -m`:

- `Ascend-cann-toolkit_<version>_linux-x86_64.run`
- `Ascend-cann-toolkit_<version>_linux-aarch64.run`

Both installers lay out the same directory tree under the install prefix,
so every command in this guide is arch-agnostic once `CANN_HOME` is set.
The `Ascend910B1` / `Ascend950PR_9599` simulator libraries are part of the
toolkit and ship pre-compiled for the matching arch — no extra step is
needed.

Quick consistency check after a fresh install:

```bash
arch                                              # x86_64 or aarch64
file "$ASCEND_HOME_PATH/tools/simulator/Ascend950PR_9599/lib/libpem_davinci.so" | head -1
```

The reported ELF class should match the host: `ELF 64-bit LSB shared
object, x86-64, …` on x86_64 hosts and `ELF 64-bit LSB shared object,
ARM aarch64, …` on aarch64 hosts.

## Locating your CANN install

CANN can be installed in several places depending on how you obtained it. This guide uses a `CANN_HOME` environment variable so the commands below work regardless of where CANN lives on your machine. Common locations:

- `$HOME/Ascend/cann` (default user install)
- `$HOME/Ascend/ascend-toolkit/latest`
- `$HOME/Ascend/cann-<version>` (e.g. `cann-9.0.0`)
- `/usr/local/Ascend/ascend-toolkit/latest` (system-wide; also the location used inside the provided Docker image)

Export it once per shell (replace the default if your install lives elsewhere):

```bash
export CANN_HOME="${CANN_HOME:-$HOME/Ascend/cann}"
```

Similarly, if you build `pyasc` from source or reference its tutorials/tests, point `PYASC_SRC` at your checkout:

```bash
export PYASC_SRC="${PYASC_SRC:-$HOME/workspace/pyasc}"
```

## Environment Setup

### 1. Source CANN environment

```bash
source "$CANN_HOME/set_env.sh"
```

This sets `ASCEND_HOME_PATH` and adds CANN binaries to `PATH`.

### 2. Set simulator library path

```bash
export LD_LIBRARY_PATH="$ASCEND_HOME_PATH/tools/simulator/Ascend950PR_9599/lib:$LD_LIBRARY_PATH"
```

This is required for the Model backend (simulator). Without it, pyasc kernels will fail with missing library errors.

### 3. Verify setup

```bash
# Check CANN version
grep '^Version=' "$ASCEND_HOME_PATH/compiler/version.info"

# Check simulator libs exist
ls "$ASCEND_HOME_PATH/tools/simulator/Ascend950PR_9599/lib/"

# Run a pyasc tutorial to verify end-to-end
cd "$PYASC_SRC"
python3.10 python/tutorials/01_add/add.py -r Model -v Ascend950PR_9599
```

Success criteria: The tutorial prints "Sample add run success." with no ISA errors.

## Evaluation pyasc clone (`pyasc-v2-eval`)

The skill-stack harness imports `asc` / `asc2` from a dedicated, read-only
checkout at `/home/aloschilov/workspace/pyasc-v2-eval`. This clone exists
purely so every simulator-verified evidence file pins to a specific SHA on
`compiler-team/pyasc#v2`. Treat the directory as read-only:

- Active pyasc development belongs in `/home/aloschilov/workspace/pyasc-fork`
  (or wherever your team-side clone lives), never here.
- The tree is kept at a detached `HEAD` so `git status` flags any accidental
  branch checkout.
- A `.git/hooks/pre-commit` refuses every commit with a pointer back to
  `pyasc-fork`.
- `EVAL-ONLY.README.md` is excluded via `.git/info/exclude` so it does not
  flip the `pyasc_revision.dirty` bit recorded in evidence files.

Bumping the pinned SHA (only by the evaluation owner):

```bash
cd /home/aloschilov/workspace/pyasc-v2-eval
git fetch origin
git checkout <new-sha-on-origin/v2>

# Re-link the editable so the harness picks the new revision up.
cd /home/aloschilov/workspace/pyasc-skill-stack
pip install -e /home/aloschilov/workspace/pyasc-v2-eval

# Then re-run the matrix and refresh docs/skill-value-q1-findings.md.
```

The CI workflow exports `PYASC_EVAL_ROOT=/home/aloschilov/workspace/pyasc-v2-eval`
so a runner that imports `asc` from any other tree gets a loud warning in
the evidence record.

## Running pyasc Kernels

Always use `python3.10` (not `python` or `python3`) since pyasc and torch are installed under Python 3.10.

```bash
export LD_LIBRARY_PATH="$ASCEND_HOME_PATH/tools/simulator/Ascend950PR_9599/lib:$LD_LIBRARY_PATH"
python3.10 kernel.py -r Model -v Ascend950PR_9599
```

- `-r Model` selects the simulator backend
- `-v Ascend950PR_9599` specifies the platform (the only platform the stack targets)

## Running pyasc Tests

```bash
# Unit tests (mocked launcher, no simulator needed)
cd "$PYASC_SRC"
python3.10 -m pytest python/test/unit/ -v

# Kernel integration tests (requires simulator)
source "$CANN_HOME/set_env.sh"
export LD_LIBRARY_PATH="$ASCEND_HOME_PATH/tools/simulator/Ascend950PR_9599/lib:$LD_LIBRARY_PATH"
python3.10 -m pytest python/test/kernels/test_vadd.py -v
```

## Troubleshooting

### ISA/DMA errors during kernel execution

If the simulator initializes ("PEM MODEL Init Success!") but produces ISA errors like `get_scalar_opcode_type not supporting this op1`, check:

1. `LD_LIBRARY_PATH` includes the correct simulator path
2. `source set_env.sh` was run in the current shell
3. The `-v Ascend950PR_9599` platform flag is specified (not `-v Ascend950PR`)

### `Too many open files` during simulator run (Docker only)

If running inside Docker and you see many lines like:

```
[ERROR] pem_log.cc:... Failed opening file /tmp/pyasc_camodel_*/core*.dump for writing: Too many open files
```

the CANN simulator is being throttled by the container's default `nofile`
limit. Restart the container with a higher limit:

```bash
docker run --rm -it --ulimit nofile=1048576:1048576 \
    -v "$(pwd)":/workspace -w /workspace pyasc-sim:latest
```

### pytest collection errors

If `pytest.Parser` is not found, upgrade pytest:

```bash
pip3.10 install "pytest>=7.0"
```

### Module not found: torch or asc

Ensure you're using `python3.10`:

```bash
python3.10 -c "import asc; print(asc.__version__)"
python3.10 -c "import torch; print(torch.__version__)"
```
