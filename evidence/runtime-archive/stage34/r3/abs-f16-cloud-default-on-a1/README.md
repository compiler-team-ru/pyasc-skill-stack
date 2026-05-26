# abs_float16 kernel

pyasc kernel implementation for the abs_float16 operation using the asc2 tile-based API.

## Files

- `kernel.py` — Kernel implementation (created in Phase 2)
- `docs/design.md` — Design document (created in Phase 1)
- `docs/environment.json` — Environment snapshot (created in Phase 0)
- `test/` — Test data and verification scripts

## Usage

```bash
python3.10 kernel.py -r Model -v Ascend950PR_9599   # Run with simulator
python3.10 kernel.py -r NPU                         # Run with NPU hardware
pytest kernel.py --backend Model --platform Ascend950PR_9599
```
