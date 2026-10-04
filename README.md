# KevClient.jl

Native Julia inference for [Kev](https://github.com/jaredpalmer/kev) decision
models — yes/no (`noul`), multiple-choice (`choice`) and rating (`score`)
questions over one shared document, with calibrated probabilities. A released
Kev checkpoint runs on CPU, CUDA and Apple GPUs without Python at inference
time. Python is used only to stage a checkpoint (`tools/`, via PythonCall.jl).

## Packages

| Path | What it is |
|---|---|
| `packages/QwenDecisionCore.jl` | Qwen3.5 / Qwen3.8 hybrid backbone (partial RoPE, full attention, Gated DeltaNet, RMS, MLP), safetensors reader/writer, Hugging Face checkpoint resolution, CPU policy, accelerator extensions, and the Choice/Noul/Score types |
| `packages/HFTokenizers.jl` | Self-contained Hugging Face byte-level BPE tokenizer reading `tokenizer.json` (Qwen / GPT-2 / RoBERTa style); no Python |
| `src/KevClient.jl` | The Kev layer: pointer head, packed encoding, TypeSafe request → record, calibrated `KevModel`, LoRA merge, and the `head.pt` bundle loader |
| `tools/` | `export_kev_checkpoint.jl` stages a Kev checkpoint for KevClient; may use Python |

## Quick start

The released Kev checkpoints are LoRA adapters on a frozen Qwen base, so merge
the adapter once and load the merged backbone:

```julia
using KevClient

# 1. Stage head.pt into a Julia-friendly bundle (once; needs Python + torch)
#    julia --project=tools tools/export_kev_checkpoint.jl jaredpalmer/kev-0.8b --out runs/kev-0.8b-bundle

# 2. Fold the adapter into the base and load
merge_lora("path/to/Qwen3.5-0.8B-Base", "path/to/kev-adapter", "runs/merged")
model = load_model("runs/kev-0.8b-bundle"; backbone_dir = "runs/merged")
tok = load_tokenizer("runs/kev-0.8b-bundle")

# 3. Ask questions. Build the record with to_record so the prompt matches a
#    Kev server exactly (Choice options become "name: description", Noul is
#    ["no","yes"], Score levels are the descriptions).
questions = [
    (type = "choice", instructions = "Which team should handle this?",
     criteria = ["returns" => "Exchanges, refunds, wrong or damaged items",
                 "shipping" => "Delivery status, delays, lost packages",
                 "billing" => "Charges, invoices, payment problems"]),
    (type = "noul", instructions = "Does this need urgent human attention?",
     criteria = nothing),
    (type = "score", instructions = "How frustrated is the customer?",
     criteria = ["Calm", "Frustrated", "Very angry"]),
]
record = to_record("Shoes arrived two weeks late and in the wrong size.", questions)
distributions = probs(model, encode(tok, record))
```

`encode` packs `<state> … <q> instructions <opt> o </opt> … <decide>` per
question; on a hybrid backbone each question runs as its own causal row. The
pointer head scores each option's `</opt>` hidden state against `<decide>`, and
the checkpoint's fitted temperature is applied in `probs`.

## Checkpoint bundles

`tools/export_kev_checkpoint.jl` reads `head.pt` (a PyTorch pickle Julia cannot
open) and writes `pointer_head.safetensors` + `kev_meta.json`, plus the
tokenizer. A full-weight checkpoint records the original as `backbone` so its
tens of GB are never copied.

```bash
(cd extern/kev && uv sync)   # or a Python with torch + safetensors
julia --project=tools tools/export_kev_checkpoint.jl jaredpalmer/kev-4b@v1.0 --out runs/kev-4b-bundle
```

## Tests

```bash
julia --project=packages/HFTokenizers.jl -e 'using Pkg; Pkg.test()'
```

`HFTokenizers` is checked against Hugging Face `tokenizers`: tiny committed
fixtures (always) and real Qwen / GPT-2 / RoBERTa tokenizers at pinned
revisions (downloaded on first use, skipped when offline). Regenerate the
oracle with `test/generate_oracle.py` / `test/generate_real_oracle.py`.

## Status

Inference was verified end to end on Kev-0.8B (merged bf16, CPU) against the
public Kev Hugging Face Space: department, escalation and frustration
probabilities agree to about `1e-3`. No generation is performed; the model only
scores prepared questions.
