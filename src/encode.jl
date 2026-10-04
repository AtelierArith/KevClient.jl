# Kev's packed encoding of a decision record: the state once, then one branch
# per question (`<q> instructions (<opt> option </opt>)* <decide>`). Port of
# kev/model.py:encode. Tokenization is a callback so this file stays free of a
# tokenizer dependency; `special` carries the five delimiter token ids.
#
# On a hybrid backbone (every current Kev) each question then runs as its own
# causal row: state tokens + that question's branch. `rows_of` splits an
# `Enc` into exactly those rows.

const OPT_NONE = -1   # instruction and state tokens
const OPT_DECIDE = -2 # the <decide> token

struct KevQuestion
    instr::String
    options::Vector{String}
    label::Any
end
KevQuestion(instr::AbstractString, options::AbstractVector; label = nothing) =
    KevQuestion(String(instr), String.(options), label)

struct KevRecord
    state::String
    questions::Vector{KevQuestion}
end
KevRecord(state::AbstractString, questions::AbstractVector) =
    KevRecord(String(state), KevQuestion[q for q in questions])

struct Enc
    ids::Vector{Int}
    seg::Vector{Int}
    pos::Vector{Int}
    opt::Vector{Int}
    option_isolation::Bool
    decide_idx::Vector{Int}
    opt_idx::Vector{Vector{Int}}
    labels::Vector{Any}
    state_tokens::Int
    state_truncated::Bool
end

state_tokens(enc::Enc) = count(==(0), enc.seg)

"""
    encode(tokenize, special, rec; max_state=384, max_branch=1024,
           strict=false, option_isolation=false) -> Enc

Pack one `KevRecord`. `tokenize(text) -> Vector{Int}` is the caller's
tokenizer (apply `escape_special` to user text first); `special` is the tuple
of five delimiter token ids in `SPECIAL_TOKENS` order.
"""
function encode(
    tokenize,
    special::NTuple{5,<:Integer},
    rec::KevRecord;
    max_state::Integer = 384,
    max_branch::Integer = 1024,
    strict::Bool = false,
    option_isolation::Bool = false,
)
    state_tokens = collect(Int, tokenize(rec.state))
    if strict && length(state_tokens) + 1 > max_state
        throw(ArgumentError("state exceeds $max_state tokens: $(length(state_tokens) + 1)"))
    end
    kept = state_tokens[1:min(length(state_tokens), max_state - 1)]
    S = vcat(Int[special[1]], kept)
    ids = copy(S)
    seg = zeros(Int, length(S))
    pos = collect(1:length(S))
    opt = fill(OPT_NONE, length(S))
    decide_idx = Int[]
    opt_idx = Vector{Int}[]
    for (k, q) in enumerate(rec.questions)
        instr = vcat(Int[special[2]], collect(Int, tokenize(q.instr)))
        spans = [vcat(Int[special[3]], collect(Int, tokenize(o)), Int[special[4]]) for o in q.options]
        br = vcat(instr, reduce(vcat, spans; init = Int[]), Int[special[5]])
        if length(br) > max_branch - length(S)
            throw(ArgumentError("branch too long: $(length(br)) tokens with a $(length(S))-token state"))
        end
        base = length(ids)
        p0 = length(S) + 1
        br_opt = vcat(
            fill(OPT_NONE, length(instr)),
            reduce(vcat, [fill(j, length(sp)) for (j, sp) in enumerate(spans)]; init = Int[]),
            Int[OPT_DECIDE],
        )
        br_pos = if option_isolation
            longest = maximum(length.(spans))
            vcat(
                collect(p0:(p0+length(instr)-1)),
                reduce(
                    vcat,
                    [collect((p0 + length(instr)):(p0 + length(instr) + length(sp) - 1)) for sp in spans];
                    init = Int[],
                ),
                Int[p0 + length(instr) + longest],
            )
        else
            collect(p0:(p0+length(br)-1))
        end
        ends = Int[]
        cursor = length(instr)
        for sp in spans
            cursor += length(sp)
            push!(ends, cursor)
        end
        append!(ids, br)
        append!(seg, fill(k, length(br)))
        append!(pos, br_pos)
        append!(opt, br_opt)
        push!(decide_idx, base + length(br))
        push!(opt_idx, [base + e for e in ends])
    end
    return Enc(
        ids,
        seg,
        pos,
        opt,
        option_isolation,
        decide_idx,
        opt_idx,
        Any[q.label for q in rec.questions],
        length(state_tokens) + 1,
        length(state_tokens) + 1 > max_state,
    )
end

"""
    encode(tok::HFTokenizer, rec; kwargs...) -> Enc

Tokenize `rec` with `tok` (escaping user text first) and pack it. The
convenience over `encode(tokenize, special, rec)`.
"""
encode(tok::HFTokenizer, rec::KevRecord; kwargs...) =
    encode(text -> user_tokens(tok, text), special_ids(tok), rec; kwargs...)

"""
    rows_of(enc) -> (state_ids, state_pos, rows)

Split a packed `Enc` into the state and one causal row per question. `rows[k]`
carries the branch ids/positions and the readout offsets *within the branch*,
so `state_ids ∪ row.ids` is what question k may attend to.
"""
function rows_of(enc::Enc)
    Ls = state_tokens(enc)
    state_ids = enc.ids[1:Ls]
    state_pos = enc.pos[1:Ls]
    rows = NamedTuple[]
    start = Ls + 1
    for (k, (d, oi)) in enumerate(zip(enc.decide_idx, enc.opt_idx))
        stop = d
        enc.seg[start] == k && enc.seg[stop] == k ||
            throw(ArgumentError("branch layout mismatch"))
        push!(
            rows,
            (
                ids = enc.ids[start:stop],
                pos = enc.pos[start:stop],
                decide = d - start + 1,
                opts = [o - start + 1 for o in oi],
            ),
        )
        start = stop + 1
    end
    return state_ids, state_pos, rows
end
