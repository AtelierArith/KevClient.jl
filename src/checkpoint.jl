# Loading a Kev checkpoint. `head.pt` is a PyTorch pickle, so a checkpoint is
# first staged into a Julia-friendly bundle by `tools/` (via PythonCall.jl):
#
#   <bundle>/
#     kev_meta.json             # the KNOWN fields of kev.checkpoint.Meta
#     pointer_head.safetensors  # head.pt["head"]: q.weight/q.bias/k.weight/k.bias
#     config.json, model*.safetensors   # only when meta.weights == "full" and
#                                       # meta.backbone is absent
#
# `meta.backbone` (set by the exporter for a full-weight checkpoint) points the
# loader at the original checkpoint directory or Hub id, so 51 GB of weights are
# never copied. A LoRA checkpoint names its frozen base in `meta.base` /
# `meta.base_revision`; the backbone is then resolved from the Hugging Face
# cache exactly like any other base. Nothing is merged, matching kev.serve's
# unmerged path; `--merge` is a future option.

struct KevMeta
    base::String
    base_revision::Union{Nothing,String}
    lora::Int
    head_dim::Int
    option_isolation::Bool
    weights::String
    weights_dtype::String
    temperature::Float64
    backbone::Union{Nothing,String}
    backbone_revision::Union{Nothing,String}
end

optional_string(d, key) = haskey(d, key) && d[key] !== nothing ? String(d[key]) : nothing

function KevMeta(d::AbstractDict)
    return KevMeta(
        String(d["base"]),
        optional_string(d, "base_revision"),
        Int(get(d, "lora", 0)),
        Int(get(d, "head_dim", 256)),
        Bool(get(d, "option_isolation", false)),
        String(get(d, "weights", "lora")),
        String(get(d, "weights_dtype", "fp32")),
        Float64(get(d, "temperature", 1.0)),
        optional_string(d, "backbone"),
        optional_string(d, "backbone_revision"),
    )
end

"""
    load_meta(bundle) -> KevMeta

Read `kev_meta.json` from a staged Kev bundle.
"""
load_meta(bundle::AbstractString) =
    KevMeta(JSON.parsefile(joinpath(bundle, "kev_meta.json")))

"""
    load_model(bundle; device=:cpu, backbone_dir=nothing) -> KevModel

Load a staged Kev bundle. `backbone_dir` overrides where the backbone weights
come from: a local directory, or `nothing` to use the bundle itself for a
full-weight checkpoint (`meta.weights == "full"`) or resolve `meta.base` from
the Hub for a LoRA checkpoint.
"""
function load_model(
    bundle::AbstractString;
    device::Symbol = :cpu,
    backbone_dir::Union{Nothing,AbstractString} = nothing,
)
    meta = load_meta(bundle)
    head = read_pointer_head(joinpath(bundle, "pointer_head.safetensors"))
    head_dim(head) == meta.head_dim || throw(
        ArgumentError(
            "Pointer dimension $(head_dim(head)) does not match meta.head_dim=$(meta.head_dim).",
        ),
    )
    path = if backbone_dir !== nothing
        backbone_dir
    elseif meta.backbone !== nothing
        revision = meta.backbone_revision === nothing ? "main" : meta.backbone_revision
        QwenDecisionCore.resolve_checkpoint(meta.backbone; revision)
    elseif meta.weights == "full"
        bundle
    else
        revision = meta.base_revision === nothing ? "main" : meta.base_revision
        QwenDecisionCore.resolve_checkpoint(meta.base; revision)
    end
    backbone = QwenDecisionCore.QwenBackbone(path; device)
    return KevModel(
        backbone,
        head;
        temperature = meta.temperature,
        option_isolation = meta.option_isolation,
    )
end
