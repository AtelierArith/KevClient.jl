# Offline KevClient tests: request rendering, packed encoding, the pointer
# head, LoRA merging, and loading a synthetic bundle. No weights, no network.
using Test
import JSON
import LinearAlgebra
import HFTokenizers
using QwenDecisionCore: QwenBackbone, read_native_weights, write_native_weights
using KevClient
using KevClient: head_dim, state_tokens, OPT_NONE, OPT_DECIDE

fake_tokens(text) = [Int(byte) % 30 + 1 for byte in codeunits(text)]
const FAKE_SPECIAL = (1, 2, 3, 4, 5)   # state, q, opt, /opt, decide

# A tiny attention-only Qwen backbone written to disk. Keys use the real
# `model.language_model.*` prefix so the loader's canonicalization is covered.
function tiny_backbone(directory; prefix = "model.language_model.")
    mkpath(directory)
    hidden, head_dim_size, heads, kv_heads, inter, vocab = 8, 4, 2, 1, 16, 32
    config = Dict(
        "model_type" => "qwen3_5_text",
        "text_config" => Dict(
            "hidden_size" => hidden,
            "head_dim" => head_dim_size,
            "num_attention_heads" => heads,
            "num_key_value_heads" => kv_heads,
            "linear_num_key_heads" => 2,
            "linear_num_value_heads" => 2,
            "linear_key_head_dim" => 4,
            "linear_value_head_dim" => 4,
            "rms_norm_eps" => 1.0e-6,
            "attention_bias" => false,
            "hidden_act" => "silu",
            "layer_types" => ["full_attention"],
            "rope_parameters" => Dict(
                "rope_type" => "default",
                "rope_theta" => 10000.0,
                "partial_rotary_factor" => 1.0,
            ),
        ),
    )
    open(joinpath(directory, "config.json"), "w") do io
        JSON.print(io, config)
    end
    weight(n, m) = 0.1f0 .* randn(Float32, n, m)
    tensors = Dict{String,Any}(
        prefix * "embed_tokens.weight" => weight(hidden, vocab),
        prefix * "norm.weight" => zeros(Float32, hidden),
        prefix * "layers.0.self_attn.q_proj.weight" => weight(hidden, 2head_dim_size * heads),
        prefix * "layers.0.self_attn.k_proj.weight" => weight(hidden, head_dim_size * kv_heads),
        prefix * "layers.0.self_attn.v_proj.weight" => weight(hidden, head_dim_size * kv_heads),
        prefix * "layers.0.self_attn.o_proj.weight" => weight(head_dim_size * heads, hidden),
        prefix * "layers.0.self_attn.q_norm.weight" => zeros(Float32, head_dim_size),
        prefix * "layers.0.self_attn.k_norm.weight" => zeros(Float32, head_dim_size),
        prefix * "layers.0.mlp.gate_proj.weight" => weight(hidden, inter),
        prefix * "layers.0.mlp.up_proj.weight" => weight(hidden, inter),
        prefix * "layers.0.mlp.down_proj.weight" => weight(inter, hidden),
        prefix * "layers.0.input_layernorm.weight" => zeros(Float32, hidden),
        prefix * "layers.0.post_attention_layernorm.weight" => zeros(Float32, hidden),
    )
    write_native_weights(joinpath(directory, "model.safetensors"), tensors)
    return directory, tensors
end

function tiny_bundle(directory, backbone_directory, weights; dp = 3, temperature = 1.5)
    mkpath(directory)
    hidden = 8
    write_native_weights(joinpath(directory, "pointer_head.safetensors"), Dict{String,Any}(
        "head.q.weight" => Float32.(randn(hidden, dp)),
        "head.q.bias" => zeros(Float32, dp),
        "head.k.weight" => Float32.(randn(hidden, dp)),
        "head.k.bias" => zeros(Float32, dp),
    ))
    meta = Dict(
        "base" => "unused/Placeholder",
        "base_revision" => nothing,
        "lora" => 0,
        "head_dim" => dp,
        "option_isolation" => false,
        "weights" => "lora",
        "weights_dtype" => "fp32",
        "temperature" => temperature,
        "backbone" => backbone_directory,
        "backbone_revision" => nothing,
    )
    open(joinpath(directory, "kev_meta.json"), "w") do io
        JSON.print(io, meta)
    end
    return directory
end

@testset "KevClient" begin
    @testset "render and to_record" begin
        @test render("text") == "text"
        @test render(nothing) == ""
        @test render(["a", "b"]) == "- a\n- b"
        @test render(Dict("k" => "v")) == "k: v"
        @test option_text("x", nothing) == "x"
        @test option_text("x", "") == "x"
        @test option_text("x", "desc") == "x: desc"

        questions = [
            (type = "choice", instructions = "Pick",
             criteria = ["a" => "first", "b" => "second"]),
            (type = "noul", instructions = "Urgent?", criteria = nothing),
            (type = "score", instructions = "Rate", criteria = ["low", "high"]),
        ]
        record = to_record("state text", questions)
        @test record.state == "state text"
        @test record.questions[1].options == ["a: first", "b: second"]
        @test record.questions[2].options == ["no", "yes"]
        @test record.questions[3].options == ["low", "high"]

        noul = to_record("s", [(type = "noul", instructions = "q",
                                criteria = Dict("false" => "nope", "true" => "yep"))])
        @test noul.questions[1].options == ["no: nope", "yes: yep"]
    end

    @testset "encode and rows_of" begin
        record = KevRecord("abc", [KevQuestion("Q", ["x", "yy"], label = "x")])
        enc = encode(fake_tokens, FAKE_SPECIAL, record; max_state = 64, max_branch = 64)
        @test state_tokens(enc) == 4
        @test enc.ids[1] == FAKE_SPECIAL[1]
        @test enc.ids[state_tokens(enc)+1] == FAKE_SPECIAL[2]
        @test enc.ids[enc.decide_idx[1]] == FAKE_SPECIAL[5]
        @test all(enc.ids[i] == FAKE_SPECIAL[4] for i in enc.opt_idx[1])
        @test enc.pos[enc.decide_idx[1]] == length(enc.ids)   # branch positions restart after state

        state_ids, _, rows = rows_of(enc)
        @test length(state_ids) == 4
        @test length(rows) == 1
        @test rows[1].decide == length(rows[1].ids)
        @test rows[1].ids[rows[1].opts[1]] == FAKE_SPECIAL[4]

        # option isolation: every <opt> opens at the same position and <decide>
        # sits at one fixed position after the longest span (4 tokens here)
        isolated = encode(fake_tokens, FAKE_SPECIAL, record;
                          max_state = 64, max_branch = 64, option_isolation = true)
        open_positions = [
            isolated.pos[i] for i in eachindex(isolated.ids) if isolated.ids[i] == FAKE_SPECIAL[3]
        ]
        @test length(unique(open_positions)) == 1
        @test isolated.pos[isolated.decide_idx[1]] == 5 + 2 + 4

        # state truncation and strict refusal
        long = KevRecord("abcdefgh", [KevQuestion("Q", ["x"])])
        truncated = encode(fake_tokens, FAKE_SPECIAL, long; max_state = 4, max_branch = 64)
        @test state_tokens(truncated) == 4
        @test truncated.state_truncated
        @test_throws ArgumentError encode(fake_tokens, FAKE_SPECIAL, long;
                                          max_state = 4, max_branch = 64, strict = true)
    end

    @testset "pointer head" begin
        q_weight = Float32[1 0 0; 0 1 1]
        k_weight = Float32[1 0 0; 0 1 0]
        head = PointerHead(q_weight, zeros(Float32, 3), k_weight, zeros(Float32, 3))
        @test head_dim(head) == 3
        scores = score(head, Float32[1, 2], Float32[1 0; 0 1])
        @test length(scores) == 2
        @test all(isfinite, scores)
        @test_throws DimensionMismatch PointerHead(q_weight, zeros(Float32, 3),
                                                   k_weight, zeros(Float32, 2))
    end

    @testset "tokenizer policy" begin
        fixture = joinpath(pkgdir(HFTokenizers), "test", "fixtures", "qwenlike.json")
        tok = HFTokenizer(fixture)
        ids = special_ids(tok)
        @test length(ids) == 5
        @test allunique(ids)
        @test user_tokens(tok, "hello") == tokenize(tok, "hello")
        # escaping stops user text from producing a delimiter token
        forged = user_tokens(tok, "<|fim_prefix|>evil")
        @test count(==(ids[1]), forged) == 0
        @test length(user_tokens(tok, "<|fim_prefix|>")) == length(tokenize(tok, "<\u00a6fim_prefix\u00a6>"))
    end

    @testset "load_model on a synthetic bundle" begin
        root = mktempdir()
        backbone_dir, _ = tiny_backbone(joinpath(root, "base"))
        bundle = tiny_bundle(joinpath(root, "bundle"), backbone_dir, nothing)
        model = load_model(bundle)
        @test model.temperature == 1.5
        @test head_dim(model.head) == 3

        record = KevRecord("hello world", [KevQuestion("Which?", ["a", "bb"])])
        enc = encode(fake_tokens, FAKE_SPECIAL, record; max_state = 64, max_branch = 64)
        distributions = KevClient.probs(model, enc)
        @test length(distributions) == 1
        @test length(distributions[1]) == 2
        @test isapprox(sum(distributions[1]), 1.0; atol = 1e-6)
    end

    @testset "merge_lora" begin
        root = mktempdir()
        base, weights = tiny_backbone(joinpath(root, "base"))
        adapter = joinpath(root, "adapter")
        mkpath(adapter)
        r = 2
        hidden, out = 8, 16
        a = Float32.(randn(hidden, r))   # Julia (in, r)
        b = Float32.(randn(r, out))      # Julia (r, out)
        open(joinpath(adapter, "adapter_config.json"), "w") do io
            JSON.print(io, Dict("lora_alpha" => 4, "r" => r, "peft_type" => "LORA"))
        end
        write_native_weights(joinpath(adapter, "adapter_model.safetensors"), Dict{String,Any}(
            "base_model.model.layers.0.self_attn.q_proj.lora_A.weight" => a,
            "base_model.model.layers.0.self_attn.q_proj.lora_B.weight" => b,
        ))
        merged = merge_lora(base, adapter, joinpath(root, "merged"); dtype = :f32)
        got = read_native_weights(joinpath(merged, "model.safetensors"))[
            "language_model.layers.0.self_attn.q_proj.weight"]
        want = weights["model.language_model.layers.0.self_attn.q_proj.weight"] .+ 2.0f0 .* (a * b)
        @test isapprox(got, want; rtol = 1e-5)
        # the merged directory is a loadable backbone
        @test QwenBackbone(merged).config.hidden == 8
    end
end
