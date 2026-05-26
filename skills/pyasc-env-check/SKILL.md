---
name: pyasc-env-check
description: pyasc development environment check skill. Verifies Python version, pyasc installation, CANN toolkit, optional torch/torch_npu, and runtime configuration. Trigger — environment check, pyasc installation, CANN setup, Python version verification.
---

# pyasc Environment Check

Quickly verify the development environment for pyasc kernel development.

## Workflow

```
Environment check
    |
    +-- Python version check (3.9-3.12)
    |
    +-- pyasc installation check
    |
    +-- CANN toolkit check
    |
    +-- Optional: torch / torch_npu check
    |
    +-- Optional: NPU device check
```

## Quick check

```bash
bash scripts/check_env.sh
```

## Check items

| Check item | Command | Required |
|-----------|---------|----------|
| Python version | `python3 --version` | Yes (3.9-3.12) |
| pyasc installed | `python3 -c "import asc; print(asc.__version__)"` | Yes |
| CANN toolkit | Check `ASCEND_HOME_PATH` | Yes |
| numpy | `python3 -c "import numpy; print(numpy.__version__)"` | Yes (< 2.0) |
| torch | `python3 -c "import torch; print(torch.__version__)"` | Optional |
| torch_npu | `python3 -c "import torch_npu"` | Optional (for NPU tensors) |
| NPU devices | `npu-smi info` | Optional (for NPU backend) |

## Environment variables

| Variable | Purpose | Required |
|----------|---------|----------|
| `ASCEND_HOME_PATH` | CANN Toolkit path | Yes |
| `PYASC_HOME` | pyasc JIT cache root | Optional |
| `PYASC_CACHE_DIR` | pyasc JIT cache directory | Optional |
| `PYASC_DUMP_PATH` | Dump ASC-IR and generated code | Optional (debugging) |

## CANN version compatibility

| pyasc version | CANN version |
|--------------|-------------|
| v1.0.0 | CANN 8.0.0 |
| v1.1.0 | CANN 8.1.0 |
| v1.2.0+ | CANN 9.0.0 |

This project is verified against **CANN 9.0.0** (`ascendai/cann:9.0.0-beta.2-910b-ubuntu22.04-py3.11`). The base image is multi-arch and runs on both x86_64 and aarch64 Linux hosts.

## Backend availability

| Backend | Requirements |
|---------|-------------|
| Model (simulator) | CANN toolkit installed, simulator libraries in `LD_LIBRARY_PATH` |
| NPU | CANN toolkit + NPU hardware (Atlas A2/A3) |

## Troubleshooting

- **`import asc` fails**: Install pyasc with `pip install pyasc` or build from source
- **CANN not found**: Run `source /usr/local/Ascend/ascend-toolkit/set_env.sh`
- **numpy >= 2.0 error**: Downgrade with `pip install "numpy<2"`
- **Model backend fails**: Check `LD_LIBRARY_PATH` includes CANN simulator libs

## References

- [Environment Requirements](references/env-requirements.md)
