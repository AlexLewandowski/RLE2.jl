
using Pkg
if !isinteractive()
    Pkg.activate(".")
end

using ArgParse

import Replotuce: get_results

function run_results(
    results_dir;
    kwargs...,
)
    get_results(
        results_dir = results_dir,
        ;kwargs...,
    )
    println()
    println("Saving results to: ", results_dir)
end

function run_results(;
    results_dir,
    plot_results,
    profiler,
    top_n,
    primary_metric_key,
    X_lim,
    recompute,
)
    run_results(
        results_dir,
        plot_results = plot_results,
        profiler = profiler,
        top_n = top_n,
        primary_metric_key = primary_metric_key,
        X_lim = X_lim,
        recompute = recompute,
    )
end

##
## If run outside of a REPL:
##

if !isinteractive()
    function settings(arg_settings::ArgParseSettings = ArgParseSettings())
        @add_arg_table arg_settings begin
            "--results_dir"
            arg_type = String
            default = "./results/experiment/version_0"

            "--plot_results"
            arg_type = Bool
            default = false

            "--profiler"
            arg_type = String
            default = "AgentType"

            "--top_n"
            arg_type = Int
            default = 1

            "--primary_metric_key"
            arg_type = String
            default = "rollout_returns"

            "--X_lim"
            arg_type = Float64
            default = 0.0

            "--recompute"
            arg_type = Bool
            default = false
        end
        return arg_settings
    end

    parsed = parse_args(settings(), as_symbols = true)
    run_results(;parsed...)
else
    @warn "In an interactive session, loading but not running run_results.jl"
end
