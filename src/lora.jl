# Fold a Kev LoRA adapter into its frozen Qwen base, matching kev.serve's
# default (LoadOptions.merge): delta = (lora_alpha / r) * B @ A added to the
# base weight, computed in fp32 and rounded once to the output dtype.
#
# peft adapter keys are `base_model.model.layers.<n>.<module>.lora_{A,B}.weight`,
# with torch shapes A (r, in) and B (out, r). read_native_weights stores them as
# A (in, r) and B (r, out), so the (in, out) base weight gains `A * B`.

const LORA_KEY = r"^base_model\.model\.(layers\.\d+\..+)\.lora_([AB])\.weight$"

function base_shard_files(directory)
    single = joinpath(directory, "model.safetensors")
    isfile(single) && return [single]
    index = joinpath(directory, "model.safetensors.index.json")
    isfile(index) || throw(ArgumentError("No model.safetensors or index under $directory"))
    mapping = JSON.parsefile(index)["weight_map"]
    return unique(joinpath(directory, String(file)) for file in values(mapping))
end

canonical_language_key(name) =
    (match = findfirst("language_model.", name)) === nothing ? nothing :
    String(name[first(match):end])

function read_base_weights(directory)
    weights = Dict{String,Any}()
    for file in base_shard_files(directory)
        for (name, value) in read_native_weights(file; convert_array = x -> Float32.(x))
            canonical = canonical_language_key(name)
            canonical === nothing || (weights[canonical] = value)
        end
    end
    return weights
end

function read_adapter(adapter_directory)
    config = JSON.parsefile(joinpath(adapter_directory, "adapter_config.json"))
    scale = Float64(config["lora_alpha"]) / Float64(config["r"])
    path = joinpath(adapter_directory, "adapter_model.safetensors")
    isfile(path) || throw(ArgumentError("No adapter_model.safetensors under $adapter_directory"))
    return scale, read_native_weights(path; convert_array = x -> Float32.(x))
end

"""
    merge_lora(base_directory, adapter_directory, out_directory;
               lora_scale=1.0, dtype=:bf16) -> out_directory

Merge a Kev LoRA adapter into its base and write a loadable backbone to
`out_directory` (`config.json` copied from the base, weights in one
`model.safetensors`). `lora_scale` multiplies the fitted `lora_alpha / r`
(1.0 is the checkpoint's own adapter). `dtype` is `:bf16` (as released) or
`:f32`.
"""
function merge_lora(
    base_directory::AbstractString,
    adapter_directory::AbstractString,
    out_directory::AbstractString;
    lora_scale::Real = 1.0,
    dtype::Symbol = :bf16,
)
    dtype in (:bf16, :f32) || throw(ArgumentError("dtype must be :bf16 or :f32"))
    fitted_scale, adapter = read_adapter(adapter_directory)
    scale = Float32(Float64(lora_scale) * fitted_scale)
    weights = read_base_weights(base_directory)

    deltas = Dict{String,Tuple{Any,Any}}()
    for (key, value) in adapter
        matched = match(LORA_KEY, key)
        matched === nothing && continue
        modpath = matched.captures[1]
        entry = get!(deltas, modpath) do
            (nothing, nothing)
        end
        deltas[modpath] =
            matched.captures[2] == "A" ? (value, entry[2]) : (entry[1], value)
    end
    isempty(deltas) && throw(ArgumentError("adapter has no lora_A/lora_B weights"))

    merged = 0
    for (modpath, (a, b)) in deltas
        (a === nothing || b === nothing) &&
            throw(ArgumentError("adapter is missing one of lora_A/lora_B for $modpath"))
        key = "language_model." * modpath * ".weight"
        base = get(weights, key, nothing)
        base === nothing && throw(ArgumentError("base has no weight for $key"))
        weights[key] = base .+ scale .* (a * b)
        merged += 1
    end
    merged == length(deltas) || throw(ArgumentError("not every adapter module merged"))

    mkpath(out_directory)
    cp(joinpath(base_directory, "config.json"), joinpath(out_directory, "config.json"); force = true)
    write_weights = if dtype === :bf16
        Dict(name => QwenDecisionCore.float32_to_bf16.(value) for (name, value) in weights)
    else
        weights
    end
    write_native_weights(joinpath(out_directory, "model.safetensors"), write_weights)
    return out_directory
end
