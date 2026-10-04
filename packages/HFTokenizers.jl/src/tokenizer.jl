# Self-contained Hugging Face byte-level BPE tokenizer. Reads `tokenizer.json`
# directly; no Python and no external tokenizer package. The engine is not
# model-family specific: it covers the ByteLevel BPE structure used by Qwen
# (Split + ByteLevel with use_regex=false, NFC), GPT-2 / RoBERTa (ByteLevel
# with use_regex=true) and similar. The normalizer, the pre-tokenizer Split
# regex, `use_regex`, and `add_prefix_space` are read from the file.
#
# Scope: BPE with byte-level pre-tokenization. Metaspace / SentencePiece
# (Llama), WordPiece (BERT) and Unigram (T5) are not supported; `byte_fallback`
# is refused. `add_special_tokens` post-processing is not applied (the
# post-processor adds nothing for these tokenizers). Added tokens with
# `lstrip`/`rstrip`/`single_word` are matched with those flags honoured.
#
# Model-specific policy (which tokens are delimiters, how user text is
# escaped) belongs to the client package (e.g. KevClient), not here.

struct HFTokenizer
    vocab::Dict{String,Int}
    id_to_token::Dict{Int,String}
    merges::Dict{Tuple{String,String},Int}
    added::Vector{NamedTuple}                  # (content, id, lstrip, rstrip, single_word)
    added_ids::Set{Int}
    split_pattern::Union{Nothing,Regex}
    bytelevel_pattern::Union{Nothing,Regex}
    add_prefix_space::Bool
    normalizer::Any
    byte_to_char::Vector{Char}                 # byte 0..255 -> display char
    char_to_byte::Dict{Char,UInt8}
end

# The ByteLevel regex (GPT-2 family). `use_regex=false` tokenizers (Qwen)
# leave the piece intact; the Split pre-tokenizer has already cut it.
const BYTELEVEL_PATTERN =
    "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+"

# GPT-2 / ByteLevel bytes_to_unicode: printable ASCII and Latin-1 map to
# themselves, the remaining bytes to U+0100 upward, so every byte is one char.
function bytelevel_tables()
    bs = vcat(
        collect(Int(0x21):Int(0x7e)),
        collect(Int(0xa1):Int(0xac)),
        collect(Int(0xae):Int(0xff)),
    )
    cs = copy(bs)
    extra = 0
    for byte in 0:255
        byte in bs && continue
        push!(bs, byte)
        push!(cs, 256 + extra)
        extra += 1
    end
    byte_to_char = Vector{Char}(undef, 256)
    char_to_byte = Dict{Char,UInt8}()
    for (byte, code) in zip(bs, cs)
        char = Char(code)
        byte_to_char[byte+1] = char
        char_to_byte[char] = UInt8(byte)
    end
    return byte_to_char, char_to_byte
end

function parse_pre_tokenizer(pre_tokenizer)
    split_pattern = nothing
    bytelevel_pattern = nothing
    add_prefix_space = false
    if pre_tokenizer === nothing
        return split_pattern, bytelevel_pattern, add_prefix_space
    elseif pre_tokenizer["type"] == "ByteLevel"
        add_prefix_space = Bool(get(pre_tokenizer, "add_prefix_space", false))
        get(pre_tokenizer, "use_regex", true) && (bytelevel_pattern = Regex(BYTELEVEL_PATTERN))
    elseif pre_tokenizer["type"] == "Sequence"
        for entry in pre_tokenizer["pretokenizers"]
            if entry["type"] == "Split"
                split_pattern = Regex(entry["pattern"]["Regex"])
            elseif entry["type"] == "ByteLevel"
                add_prefix_space = Bool(get(entry, "add_prefix_space", false))
                get(entry, "use_regex", true) && (bytelevel_pattern = Regex(BYTELEVEL_PATTERN))
            else
                throw(ArgumentError("unsupported pre_tokenizer $(entry["type"])"))
            end
        end
    else
        throw(ArgumentError("unsupported pre_tokenizer $(pre_tokenizer["type"])"))
    end
    return split_pattern, bytelevel_pattern, add_prefix_space
end

function parse_tokenizer(file::AbstractString)
    data = JSON.parsefile(file)
    model = data["model"]
    if haskey(model, "type") && model["type"] != "BPE"
        throw(ArgumentError("unsupported tokenizer model $(model["type"])"))
    end
    get(model, "byte_fallback", false) === true &&
        throw(ArgumentError("byte_fallback tokenizers are not supported"))
    vocab = Dict{String,Int}(String(k) => Int(v) for (k, v) in model["vocab"])
    merges = Dict{Tuple{String,String},Int}()
    for (rank, merge) in enumerate(model["merges"])
        # Two schemas: "left right" strings (older) and ["left","right"] arrays.
        left, right = merge isa AbstractString ? split(merge, ' '; limit = 2) :
                      (merge[1], merge[2])
        merges[(String(left), String(right))] = rank
    end
    added = NamedTuple[]
    added_ids = Set{Int}()
    for token in get(data, "added_tokens", [])
        content = String(token["content"])
        id = Int(token["id"])
        push!(added, (
            content = content,
            id = id,
            lstrip = Bool(get(token, "lstrip", false)),
            rstrip = Bool(get(token, "rstrip", false)),
            single_word = Bool(get(token, "single_word", false)),
        ))
        push!(added_ids, id)
        haskey(vocab, content) || (vocab[content] = id)
    end
    sort!(added; by = token -> -ncodeunits(token.content))
    id_to_token = Dict{Int,String}(id => token for (token, id) in vocab)
    split_pattern, bytelevel_pattern, add_prefix_space = parse_pre_tokenizer(data["pre_tokenizer"])
    byte_to_char, char_to_byte = bytelevel_tables()
    return HFTokenizer(
        vocab,
        id_to_token,
        merges,
        added,
        added_ids,
        split_pattern,
        bytelevel_pattern,
        add_prefix_space,
        get(data, "normalizer", nothing),
        byte_to_char,
        char_to_byte,
    )
end

function tokenizer_file(source::AbstractString)
    expanded = expanduser(source)
    if isdir(expanded)
        return joinpath(abspath(expanded), "tokenizer.json")
    elseif isfile(expanded)
        return abspath(expanded)
    end
    throw(ArgumentError("no tokenizer.json at $source; pass a file or a directory"))
end

"""
    HFTokenizer(source)

Load a Hugging Face byte-level BPE tokenizer from a `tokenizer.json` path or a
directory containing one. Resolving a Hub repository is the caller's job (see
`KevClient.load_tokenizer`).
"""
HFTokenizer(source::AbstractString) = parse_tokenizer(tokenizer_file(source))

normalize_with(spec::Nothing, text::AbstractString) = text
function normalize_with(spec, text::AbstractString)
    kind = spec["type"]
    kind == "NFC" && return Unicode.normalize(text, :NFC)
    kind == "NFD" && return Unicode.normalize(text, :NFD)
    kind == "NFKC" && return Unicode.normalize(text, :NFKC)
    kind == "NFKD" && return Unicode.normalize(text, :NFKD)
    kind == "Lowercase" && return lowercase(text)
    if kind == "Sequence"
        return foldl((value, step) -> normalize_with(step, value), spec["normalizers"]; init = text)
    end
    throw(ArgumentError("unsupported normalizer type $kind"))
end

# Split with HF's "Isolated" behavior: every regex match is a piece, and any
# uncovered gap is kept as its own piece (these regexes cover all input).
function split_isolated(pattern::Regex, text::AbstractString)
    pieces = String[]
    position = firstindex(text)
    for match in eachmatch(pattern, text)
        offset = match.offset
        if offset > position
            push!(pieces, String(SubString(text, position, prevind(text, offset))))
        end
        push!(pieces, String(match.match))
        position = offset + ncodeunits(match.match)
    end
    if position <= lastindex(text)
        push!(pieces, String(SubString(text, position, lastindex(text))))
    end
    return pieces
end

# Map a text piece to its ByteLevel representation (one Unicode char per byte).
function bytelevel(tok::HFTokenizer, piece::AbstractString)
    buffer = IOBuffer()
    for char in piece
        for byte in codeunits(string(char))
            write(buffer, tok.byte_to_char[Int(byte)+1])
        end
    end
    return String(take!(buffer))
end

# Greedy BPE: repeatedly merge the adjacent pair with the lowest merge rank
# (all occurrences at once), then map symbols to ids.
function bpe(tok::HFTokenizer, symbols::Vector{Char})
    parts = String[string(char) for char in symbols]
    while length(parts) > 1
        best_rank = typemax(Int)
        best_pair = nothing
        for index in 1:(length(parts)-1)
            rank = get(tok.merges, (parts[index], parts[index+1]), nothing)
            if rank !== nothing && rank < best_rank
                best_rank = rank
                best_pair = (parts[index], parts[index+1])
            end
        end
        best_pair === nothing && break
        merged = String[]
        index = 1
        while index <= length(parts)
            if index < length(parts) &&
               parts[index] == best_pair[1] &&
               parts[index+1] == best_pair[2]
                push!(merged, parts[index] * parts[index+1])
                index += 2
            else
                push!(merged, parts[index])
                index += 1
            end
        end
        parts = merged
    end
    return parts
end

function push_symbols!(tok::HFTokenizer, ids::Vector{Int}, text::AbstractString)
    for symbol in bpe(tok, collect(bytelevel(tok, text)))
        id = get(tok.vocab, symbol, -1)
        id == -1 && throw(ArgumentError("unknown token symbol $(repr(symbol))"))
        push!(ids, id)
    end
    return ids
end

function encode_piece(tok::HFTokenizer, ids::Vector{Int}, segment::AbstractString)
    isempty(segment) && return ids
    # ByteLevel add_prefix_space: prepend a space to each segment (the text
    # between added tokens) that does not already start with whitespace, then
    # pretokenize normally, so the space merges with the following word.
    text = tok.add_prefix_space && !isspace(first(segment)) ? " " * segment : segment
    pieces = tok.split_pattern === nothing ? String[text] :
             split_isolated(tok.split_pattern, text)
    for piece in pieces
        isempty(piece) && continue
        subpieces = tok.bytelevel_pattern === nothing ? String[piece] :
                    split_isolated(tok.bytelevel_pattern, piece)
        for subpiece in subpieces
            isempty(subpiece) && continue
            push_symbols!(tok, ids, subpiece)
        end
    end
    return ids
end

# True when an added token may start at `index` under single_word/lstrip/rstrip.
function added_matches(tok::HFTokenizer, text, index, token)
    if token.single_word
        index > firstindex(text) && iswordchar(text[prevind(text, index)]) && return false
        # Step by characters so `finish` is a valid index past a multibyte char.
        finish = nextind(text, index, length(token.content))
        finish <= lastindex(text) && iswordchar(text[finish]) && return false
    end
    return startswith(SubString(text, index), token.content)
end

iswordchar(char::Char) = isletter(char) || isnumeric(char) || char == '_' || char == '\''

function skip_whitespace_left(text, index)
    while index > firstindex(text)
        previous = prevind(text, index)
        isspace(text[previous]) || break
        index = previous
    end
    return index
end

function skip_whitespace_right(text, index)
    while index <= lastindex(text) && isspace(text[index])
        index = nextind(text, index)
    end
    return index
end

"""
    tokenize(tok, text) -> Vector{Int}

Tokenize text. Added tokens present in the text are recognised as single ids
(Hugging Face behavior); `lstrip`/`rstrip` consume surrounding whitespace.
"""
function tokenize(tok::HFTokenizer, text::AbstractString)
    normalized = normalize_with(tok.normalizer, text)
    ids = Int[]
    buffer = IOBuffer()
    flush!() = encode_piece(tok, ids, String(take!(buffer)))
    index = firstindex(normalized)
    while index <= lastindex(normalized)
        matched = false
        for token in tok.added
            if added_matches(tok, normalized, index, token)
                flush!()
                token.lstrip && (index = skip_whitespace_left(normalized, index))
                push!(ids, token.id)
                index += ncodeunits(token.content)
                token.rstrip && (index = skip_whitespace_right(normalized, index))
                matched = true
                break
            end
        end
        matched && continue
        write(buffer, normalized[index])
        index = nextind(normalized, index)
    end
    flush!()
    return ids
end

"""
    decode(tok, ids) -> String

Inverse of `tokenize` (added tokens emitted literally, everything else through
the byte mapping). For inspection and tests, not for generation.
"""
function decode(tok::HFTokenizer, ids::AbstractVector{<:Integer})
    buffer = IOBuffer()
    for id in ids
        token = get(tok.id_to_token, Int(id), nothing)
        token === nothing && throw(ArgumentError("unknown token id $id"))
        if Int(id) in tok.added_ids
            write(buffer, token)
        else
            for char in token
                byte = get(tok.char_to_byte, char, nothing)
                byte === nothing &&
                    throw(ArgumentError("token $(repr(token)) is not byte-level"))
                write(buffer, byte)
            end
        end
    end
    return String(take!(buffer))
end
