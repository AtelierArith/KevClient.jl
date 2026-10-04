# tools

One-off utilities that may use Python, always through
[PythonCall.jl](https://github.com/JuliaPy/PythonCall.jl). Inference in
`src/` never imports Python.

## `export_kev_checkpoint.jl`

Stages a Kev checkpoint into a bundle `KevClient.load_model` can read without
Python. `head.pt` is a PyTorch pickle and Julia cannot open it; this tool loads
it, writes `pointer_head.safetensors` and `kev_meta.json`, copies the tokenizer,
and (for a full-weight checkpoint) records the original checkpoint as
`backbone` so its tens of GB are not copied.

```bash
julia --project=tools tools/export_kev_checkpoint.jl \
    jaredpalmer/kev-4b@v1.0 --out runs/kev-4b-bundle

# or a local run directory
julia --project=tools tools/export_kev_checkpoint.jl \
    runs/smoke/00-trial-0/checkpoint --out runs/smoke-bundle

# then, in the main environment
julia --project=. -e 'using KevClient; m = load_model("runs/kev-4b-bundle")'
```

`source` is a local checkpoint directory or `org/model` (with an optional
`@revision`, or `--revision`). `--force` overwrites a non-empty bundle.

### Python environment

The tool prefers the vendored Kev environment, so create it once:

```bash
(cd extern/kev && uv sync)
```

It falls back to `tools/.venv`; to use that instead:

```bash
uv venv tools/.venv
uv pip install --python tools/.venv -r tools/requirements.txt
```

Set `JULIA_PYTHONCALL_EXE=/path/to/python` to point at another interpreter.
Python and torch are needed only here, not for inference.
