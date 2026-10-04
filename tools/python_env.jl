# Python resources used by tools are always entered through PythonCall.jl.
# Prefer the vendored Kev environment (extern/kev/.venv, created by `uv sync`
# there); fall back to a tools-local venv. Override with JULIA_PYTHONCALL_EXE.
ENV["JULIA_CONDAPKG_BACKEND"] = "Null"
const KEV_PYTHON = joinpath(@__DIR__, "..", "extern", "kev", ".venv", "bin", "python")
const TOOLS_PYTHON = joinpath(@__DIR__, ".venv", "bin", "python")
get!(ENV, "JULIA_PYTHONCALL_EXE", isfile(KEV_PYTHON) ? KEV_PYTHON : TOOLS_PYTHON)
