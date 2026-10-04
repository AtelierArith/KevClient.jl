# Julia entry point (PythonCall.jl) for staging a Kev checkpoint for KevClient.
# The worker is _kev_export.py; this file only wires arguments through.
#
#   julia --project=tools tools/export_kev_checkpoint.jl \
#       jaredpalmer/kev-4b@v1.0 --out runs/kev-4b-bundle
#
# Underscore-prefixed helpers are the Python worker; this file is the CLI.
include("python_env.jl")
using PythonCall

sys = pyimport("sys")
sys.path.insert(0, @__DIR__)
exporter = pyimport("_kev_export")
try
    exporter.main(pylist(ARGS))
catch exception
    if exception isa PyException && pyisinstance(exception.v, pybuiltins.SystemExit)
        exit(pyconvert(Int, exception.v.code))
    end
    rethrow()
end
