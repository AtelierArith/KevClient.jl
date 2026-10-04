# TypeSafe System One request -> Kev record, port of kev/api.py:to_record.
# The exact prompt matters: a Choice option is "name: description" when a
# description is given, a Noul is always ["no", "yes"], and a Score option is
# the level text. Building the record by hand without this gets different
# (wrong) probabilities.

"""
    render(value; indent=0) -> String

Flatten a string | object | array into the text the model sees, keeping object
field names as labels (kev/api.py:render).
"""
render(value::Nothing; indent::Integer = 0) = ""
render(value::Union{AbstractString,Number,Bool}; indent::Integer = 0) = string(value)
function render(value::AbstractVector; indent::Integer = 0)
    pad = "  "^indent
    return join(
        (string(pad, "- ", lstrip(render(item; indent = indent + 1))) for item in value),
        "\n",
    )
end
function render(value::AbstractDict; indent::Integer = 0)
    pad = "  "^indent
    parts = String[]
    for (key, item) in value
        if item isa AbstractDict || item isa AbstractVector
            push!(parts, string(pad, key, ":\n", render(item; indent = indent + 1)))
        else
            push!(parts, string(pad, key, ": ", render(item; indent = indent)))
        end
    end
    return join(parts, "\n")
end

"""
    option_text(name, description) -> String

An option is `"name"`, or `"name: description"` when a description is given.
"""
option_text(name, description) =
    description === nothing || description == "" ? string(name) :
    string(name, ": ", render(description))

criteria_field(criteria, key) =
    criteria === nothing ? nothing :
    criteria isa NamedTuple ? get(criteria, Symbol(key), nothing) : get(criteria, key, nothing)

"""
    to_record(state, questions) -> KevRecord

Build the record Kev actually encodes from a TypeSafe-shaped request.
`questions` is a vector of named tuples with `type` (`"noul"`/`"choice"`/
`"score"`), `instructions`, and `criteria`:

- choice: an ordered collection of `name => description` pairs (a `Vector{Pair}`
  keeps the option order; a `Dict` does not),
- noul: `nothing`, or a named tuple / dict with optional `false` and `true`
  descriptions,
- score: a vector of level descriptions, lowest to highest.

`state` may be a string, object or array; objects and arrays are rendered with
their labels. Mirrors kev/api.py:to_record, so answers match a Kev server.
"""
function to_record(state, questions::AbstractVector)
    built = KevQuestion[]
    for question in questions
        kind = question.type
        options = if kind == "noul"
            criteria = question.criteria
            [
                option_text("no", criteria_field(criteria, "false")),
                option_text("yes", criteria_field(criteria, "true")),
            ]
        elseif kind == "choice"
            [option_text(first(pair), last(pair)) for pair in question.criteria]
        elseif kind == "score"
            [render(level) for level in question.criteria]
        else
            throw(ArgumentError("unsupported question type $kind"))
        end
        push!(built, KevQuestion(render(question.instructions), options))
    end
    return KevRecord(render(state), built)
end
