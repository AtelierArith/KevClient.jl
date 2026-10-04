using Test
import JSON
import Downloads
using HFTokenizers

const DIR = @__DIR__
const CACHE = joinpath(DIR, ".cache")

# Tiny committed fixtures: the oracle is offline and always runs.
oracle = JSON.parsefile(joinpath(DIR, "oracle.json"))
strings = oracle["strings"]

@testset "HFTokenizer vs Hugging Face oracle (tiny fixtures)" begin
    for (name, entry) in oracle["fixtures"]
        tok = HFTokenizer(joinpath(DIR, entry["file"]))
        @testset "$name" begin
            for (index, text) in enumerate(strings)
                expected = collect(Int, entry["ids"][index])
                @test tokenize(tok, text) == expected
                @test decode(tok, expected) == entry["decoded"][index]
            end
            if haskey(entry, "escaped_strings")
                for (text, ids) in zip(entry["escaped_strings"], entry["escaped_ids"])
                    @test tokenize(tok, text) == collect(Int, ids)
                end
            end
        end
    end
end

# Real tokenizers at a pinned revision, downloaded on first use and skipped
# when the network is unavailable.
function fetch_tokenizer(repo, revision)
    mkpath(CACHE)
    path = joinpath(CACHE, replace(repo, "/" => "--") * "@" * revision * ".json")
    isfile(path) && return path
    url = "https://huggingface.co/$repo/resolve/$revision/tokenizer.json"
    try
        Downloads.download(url, path)
        return path
    catch
        isfile(path) && rm(path; force = true)
        return nothing
    end
end

real = JSON.parsefile(joinpath(DIR, "real_oracle.json"))
@testset "HFTokenizer vs real tokenizers" begin
    for (name, entry) in real["tokenizers"]
        path = fetch_tokenizer(entry["repo"], entry["revision"])
        if path === nothing
            @info "skipping $name: could not download $(entry["repo"])@$(entry["revision"])"
            @test_skip true
            continue
        end
        tok = HFTokenizer(path)
        @testset "$name" begin
            for (index, text) in enumerate(real["strings"])
                @test tokenize(tok, text) == collect(Int, entry["ids"][index])
            end
            if haskey(entry, "escaped_strings")
                for (text, ids) in zip(entry["escaped_strings"], entry["escaped_ids"])
                    @test tokenize(tok, text) == collect(Int, ids)
                end
            end
        end
    end
end
