# Kev decision model: a QwenDecisionCore.QwenBackbone plus a pointer head. On a
# hybrid backbone every question runs as its own causal row (state + branch),
# which is exactly what `rows_of` produces and what kev/model.py:
# forward_rows_batch does.
struct KevModel{B,H}
    backbone::B
    head::H
    temperature::Float64
    max_options::Int
    option_isolation::Bool
end

"""
    KevModel(backbone, head; temperature=1.0, max_options=255, option_isolation=false)

Combine a loaded `QwenBackbone` with a `PointerHead`. `temperature` is the
checkpoint's fitted calibration (1.0 = raw logits); `max_options` its trained
option limit.
"""
function KevModel(
    backbone::QwenDecisionCore.QwenBackbone,
    head::PointerHead;
    temperature::Real = 1.0,
    max_options::Integer = 255,
    option_isolation::Bool = false,
)
    scale = Float64(temperature)
    isfinite(scale) && scale > 0 || throw(ArgumentError("Temperature must be positive and finite."))
    1 <= max_options <= 255 || throw(ArgumentError("max_options must be between 1 and 255."))
    return KevModel(backbone, head, scale, Int(max_options), option_isolation)
end

"""
    logits(model, enc) -> Vector{Vector{Float32}}

Uncalibrated option scores per question, one row per question.
"""
function logits(model::KevModel, enc::Enc)
    state_ids, _, rows = rows_of(enc)
    out = Vector{Vector{Float32}}(undef, length(rows))
    for (k, row) in enumerate(rows)
        length(row.opts) <= model.max_options ||
            throw(ArgumentError("Question $k exceeds the checkpoint option limit."))
        row_ids = vcat(state_ids, row.ids)
        mask = ones(Int, length(row_ids))
        states = QwenDecisionCore.backbone_hidden(model.backbone, row_ids, mask)
        Ls = length(state_ids)
        h_decide = view(states, :, Ls + row.decide)
        h_opts = states[:, Ls .+ row.opts]
        out[k] = collect(Float32, score(model.head, h_decide, h_opts))
    end
    return out
end

"""
    probs(model, enc) -> Vector{Vector{Float64}}

Calibrated probabilities per question (temperature applied), in option order.
"""
function probs(model::KevModel, enc::Enc)
    return [QwenDecisionCore.probabilities(z, model.temperature) for z in logits(model, enc)]
end

"""
    decide(model, enc, questions) -> Vector

Answer every question of `enc` (ordered as in the record). `questions` are
`ChoiceQuestion` / `NoulQuestion` / `ScoreQuestion` values from
QwenDecisionCore, matched to the record's options in order.
"""
function decide(model::KevModel, enc::Enc, questions::AbstractVector)
    length(questions) == length(enc.decide_idx) ||
        throw(ArgumentError("Question count does not match the encoding."))
    return [
        QwenDecisionCore.answer(q, probs(model, enc)[k][1:QwenDecisionCore.option_count(q)]) for
        (k, q) in enumerate(questions)
    ]
end

decide(model::KevModel, enc::Enc, question) = only(decide(model, enc, [question]))
