# Kev-specific tokenizer policy. The byte-level BPE engine is the separate
# HFTokenizers.jl package; here we fix which tokens Kev reuses as delimiters
# and how caller text is escaped before tokenization.
#
# Reuse existing rarely-used Qwen special tokens as delimiters (state, q, opt,
# /opt, decide); see kev/model.py:SPECIAL. They live in the tokenizer's
# `added_tokens`, not its BPE vocab.
const SPECIAL_TOKENS = (
    "<|fim_prefix|>",
    "<|fim_middle|>",
    "<|box_start|>",
    "<|box_end|>",
    "<|fim_suffix|>",
)

"""
    escape_special(text) -> String

Rewrite `<|name|>` to `<¦name¦>` so caller text can never forge a delimiter
token (kev/model.py:user_tokens).
"""
escape_special(text::AbstractString) =
    replace(text, r"<\|([A-Za-z0-9_]+)\|>" => s"<¦\1¦>")

"""
    user_tokens(tok, text) -> Vector{Int}

Escape `<|name|>` and tokenize, matching kev/model.py:user_tokens.
"""
user_tokens(tok::HFTokenizer, text::AbstractString) =
    tokenize(tok, escape_special(text))

"""
    special_ids(tok) -> NTuple{5,Int}

The delimiter token ids in `SPECIAL_TOKENS` order (state, q, opt, /opt, decide).
"""
function special_ids(tok::HFTokenizer)
    return ntuple(5) do index
        token = SPECIAL_TOKENS[index]
        get(tok.vocab, token) do
            throw(ArgumentError("tokenizer has no $token"))
        end
    end
end

"""
    load_tokenizer(source; revision="main") -> HFTokenizer

Load a Kev/Qwen tokenizer from a `tokenizer.json` path, a checkpoint directory,
or a Hugging Face repository ID (resolved through QwenDecisionCore).
"""
function load_tokenizer(source::AbstractString; revision::AbstractString = "main")
    expanded = expanduser(source)
    if !isdir(expanded) && !isfile(expanded)
        directory = QwenDecisionCore.resolve_checkpoint(
            source;
            revision,
            required = ("tokenizer.json",),
            auxiliary = (),
            patterns = (),
            require_weights = false,
        )
        return HFTokenizer(joinpath(directory, "tokenizer.json"))
    end
    return HFTokenizer(source)
end
