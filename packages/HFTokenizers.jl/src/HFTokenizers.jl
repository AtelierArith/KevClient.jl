# Self-contained Hugging Face byte-level BPE tokenizer. Reads `tokenizer.json`
# directly; no Python and no external tokenizer package. Covers the ByteLevel
# BPE structure used by Qwen, GPT-2, RoBERTa and similar: NFC/NFD/NFKC/NFKD or
# no normalization, a Split pre-tokenizer regex, the ByteLevel byte mapping
# with `use_regex` and `add_prefix_space`, BPE merges, and added tokens.
# Model-family policy (which tokens are delimiters, how user text is escaped)
# belongs to the client package.
module HFTokenizers

import JSON
import Unicode

export HFTokenizer, tokenize, decode

include("tokenizer.jl")

end # module HFTokenizers
