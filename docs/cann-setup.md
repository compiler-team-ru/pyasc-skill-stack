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
export LD_LIBRARY_PATH="$ASCEND_HOME_PATH/tools/simulator/Ascend910B1/lib:$LD_LIBRARY_PATH"
```

This is required for the Model backend (simulator). Without it, pyasc kernels will fail with missing library errors.

### 3. Verify setup

```bash
# Check CANN version
grep '^Version=' "$ASCEND_HOME_PATH/compiler/version.info"

# Check simulator libs exist
ls "$ASCEND_HOME_PATH/tools/simulator/Ascend910B1/lib/"

# Run a pyasc tutorial to verify end-to-end
cd "$PYASC_SRC"
python3.10 python/tutorials/01_add/add.py -r Model -v Ascend910B1
```

Success criteria: The tutorial prints "Sample add run success." with no ISA errors.

## Running pyasc Kernels

Always use `python3.10` (not `python` or `python3`) since pyasc and torch are installed under Python 3.10.

```bash
export LD_LIBRARY_PATH="$ASCEND_HOME_PATH/tools/simulator/Ascend910B1/lib:$LD_LIBRARY_PATH"
python3.10 kernel.py -r Model -v Ascend910B1
```

- `-r Model` selects the simulator backend
- `-v Ascend910B1` specifies the platform (the `1` suffix is required)

## Running pyasc Tests

```bash
# Unit tests (mocked launcher, no simulator needed)
cd "$PYASC_SRC"
python3.10 -m pytest python/test/unit/ -v

# Kernel integration tests (requires simulator)
source "$CANN_HOME/set_env.sh"
export LD_LIBRARY_PATH="$ASCEND_HOME_PATH/tools/simulator/Ascend910B1/lib:$LD_LIBRARY_PATH"
python3.10 -m pytest python/test/kernels/test_vadd.py -v
```

## Troubleshooting

### ISA/DMA errors during kernel execution

If the simulator initializes ("PEM MODEL Init Success!") but produces ISA errors like `get_scalar_opcode_type not supporting this op1`, check:

1. `LD_LIBRARY_PATH` includes the correct simulator path
2. `source set_env.sh` was run in the current shell
3. The `-v Ascend910B1` platform flag is specified (not `-v Ascend910B`)

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
