# Pointer readout of a Kev checkpoint: the <decide> hidden state and each
# option's </opt> hidden state are projected to a `dp`-dimensional space and
# scored by a scaled dot product. Mirrors kev/model.py:PointerHead.
#
# Tensor convention: matrices arrive through QwenDecisionCore.read_native_weights,
# which reverses PyTorch's row-major axes, so a torch Linear(d, dp) weight of
# shape (dp, d) is stored here as (d, dp) and applied as `weight' * x`.
struct PointerHead{QW,QB,KW,KB}
    q_weight::QW
    q_bias::QB
    k_weight::KW
    k_bias::KB
    function PointerHead(q_weight, q_bias, k_weight, k_bias)
        size(q_weight) == size(k_weight) ||
            throw(DimensionMismatch("Pointer q/k weights must have the same shape."))
        length(q_bias) == length(k_bias) == size(q_weight, 2) ||
            throw(DimensionMismatch("Pointer biases must match the pointer dimension."))
        return new{typeof(q_weight),typeof(q_bias),typeof(k_weight),typeof(k_bias)}(
            q_weight,
            q_bias,
            k_weight,
            k_bias,
        )
    end
end

head_dim(h::PointerHead) = size(h.q_weight, 2)

pointer_scale(h::PointerHead) = inv(sqrt(Float32(head_dim(h))))

"""
    score(head, h_decide, h_opts) -> Vector

Uncalibrated scores, one per option. `h_decide` is `(d,)`; `h_opts` is `(d, K)`.
Temperature is applied by `KevModel`, not here.
"""
function score(h::PointerHead, h_decide::AbstractVector, h_opts::AbstractMatrix)
    size(h_opts, 1) == length(h_decide) ||
        throw(DimensionMismatch("Hidden width differs between decide and options."))
    q = h.q_weight' * h_decide .+ h.q_bias
    k = h.k_weight' * h_opts .+ h.k_bias
    return vec(k' * q) .* pointer_scale(h)
end

"""
    read_pointer_head(path; prefix="head.")

Load a pointer head from a safetensors file produced from `head.pt` (a
conversion performed by `tools/`; see checkpoint.jl). Keys are the PyTorch
state-dict names, `head.q.weight` by default.
"""
function read_pointer_head(path::AbstractString; prefix::AbstractString = "head.")
    tensors = QwenDecisionCore.read_native_weights(path)
    tensor(name) = get(tensors, prefix * name) do
        throw(ArgumentError("$path is missing $prefix$name."))
    end
    return PointerHead(
        tensor("q.weight"),
        tensor("q.bias"),
        tensor("k.weight"),
        tensor("k.bias"),
    )
end
