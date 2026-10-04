"""
    KevClient

Thin Julia wrapper around [Kev](https://github.com/jaredpalmer/kev) decision
models. The heavy lifting — the Qwen3.5 / Qwen3.8 backbone, safetensors
loading, Hugging Face checkpoint resolution, the CPU policy and the accelerator
extensions — lives in [QwenDecisionCore](https://github.com/AtelierArith), and
the byte-level BPE tokenizer in HFTokenizers. KevClient adds what is
Kev-specific: the pointer head, the packed question encoding, the calibrated
`KevModel`, Kev's `head.pt` checkpoint bundle, and the delimiter-token policy.

Everything here is inference on prepared token ids or on text; no generation is
performed.
"""
module KevClient

import JSON
using LinearAlgebra
using QwenDecisionCore
using HFTokenizers

export HFTokenizer, tokenize, decode, load_tokenizer
export SPECIAL_TOKENS, escape_special, user_tokens, special_ids
export KevRecord, KevQuestion, Enc, encode, rows_of
export to_record, render, option_text
export PointerHead, read_pointer_head, score
export KevModel, logits, probs, decide
export KevMeta, load_meta, load_model, merge_lora
export ChoiceQuestion, NoulQuestion, ScoreQuestion

include("kev_tokenizer.jl")
include("pointer_head.jl")
include("encode.jl")
include("request.jl")
include("model.jl")
include("lora.jl")
include("checkpoint.jl")

end # module KevClient
