# opencode profile templates

These JSON files are templates loaded by
[`tests/tools/collect_generative_evidence.py`](../../tests/tools/collect_generative_evidence.py)
via the `--model-profile <name>` flag. Each template provides the per-project
`opencode.json` for one model / provider combination.

## Template substitution

The collector reads the file as text, substitutes `${VAR}` patterns with the
current process environment (via `string.Template.safe_substitute`), and then
parses the result as JSON. Unset variables expand to empty strings.

This is how the local-model profiles target an Ollama sidecar without baking a
hostname into the repo:

```bash
OLLAMA_BASE_URL=http://127.0.0.1:11434/v1 \
  python3.10 tests/tools/collect_generative_evidence.py \
    --op abs --dtype float16 \
    --model-profile local-qwen-coder-7b \
    --skills-mode on
```

The collector merges the `skills.paths` setting on top of the template at
runtime — keep templates skills-agnostic so the same file works for both
`--skills-mode on` and `--skills-mode off`.

## Profiles

| File | Purpose |
|------|---------|
| `cloud-default.json` | Inherits the cloud config from `~/.config/opencode/opencode.json` (configured separately, e.g. from the `OPENCODE_CONFIG` GitHub secret). The template adds only baseline permissions; the cloud provider is supplied by the global config. |
| `local-qwen-coder-7b.json` | Routes every request to `qwen2.5-coder:7b` served by Ollama at `${OLLAMA_BASE_URL}` (default `http://127.0.0.1:11434/v1` via opencode CLI defaults). |
| `local-llama-3.1-8b.json` | Routes every request to `llama3.1:8b` served by the same Ollama endpoint. |
