
using Pkg
if !isinteractive()
    Pkg.activate(".")
end

using Pkg.TOML
using ArgParse

function run_test(;
    config_file::String
)
    run_test(config_file)
end


function run_test(
    config_file::String;
    return_early = false,
    test_hyperparams = false,
    exp_file = nothing,
)
    parsed, sweep_d, static_d, config_d = config_to_test_parsed(config_file, return_early, test_hyperparams)

    try
        rm(parsed["save_dir"], recursive = true)
    catch
    end

    try
        mkpath(parsed["save_dir"])
    catch
    end

    open(parsed["save_dir"]*"/config_test.toml", "w") do io
        print(io, "[sweep_args]\n")
        TOML.print(io, sweep_d)
        print(io, "[static_args]\n")
        TOML.print(io, static_d)
        print(io, "[config]\n")
        TOML.print(io, config_d)
    end

    try
        mkpath(parsed["_SAVE"])
    catch
    end


    if exp_file === nothing
        exp_file = parsed["exp_file"]
    end
    include(String(@__DIR__)*"/../"*exp_file)

    result = Base.invokelatest(eval(Meta.parse(parsed["exp_func_name"])), parsed, test = true)
    println("Experiment with configuration \"", config_file,  "\" successfully completed!")
    return result
end

function config_to_test_parsed(config_file, return_early, test_hyperparams)
    parsed = Dict()
    config = TOML.parsefile(config_file)
    sweep_dict = config["sweep_args"]
    static_dict = config["static_args"]
    config_dict = config["config"]
    new_sweep = Dict()
    new_static = Dict()
    new_config = Dict()
    for k in collect(keys(sweep_dict))
        d = sweep_dict[k]
        if typeof(d) <: String
            val = eval(Meta.parse(d))[1]
        else
            val = d[1]
        end
        parsed[k] = val
        new_sweep[k] = val
    end

    if return_early
        static_dict["return_early"] = true
    end

    for k in collect(keys(static_dict))
        val = static_dict[k]
        if test_hyperparams
            if k == "num_episodes"
                val = 5
            end
            if k == "init_num_episodes"
                val = 5
            end
            if k == "num_grad_steps"
                val = 5
            end
            if k == "max_episode_length"
                val = 5
            end
            if k == "hidden_size"
                val = 32
            end
            if k == "update_freq"
                val = 2
            end
            if k == "num_layers"
                val = 0
            end
            if k == "batch_size"
                val = 2
            end
        end
        parsed[k] = val
        new_static[k] = val
    end

    for k in collect(keys(config_dict))
        val = config_dict[k]
        parsed[k] = val
        new_config[k] = val
    end

    parsed["_SAVE"] = ".temp/"*splitpath(config["config"]["save_dir"])[end]*"/data/exp_1/"
    parsed["save_dir"] = ".temp/"*splitpath(config["config"]["save_dir"])[end]

    return parsed, new_sweep, new_static, new_config
end



if !isinteractive()
    function settings(arg_settings::ArgParseSettings = ArgParseSettings())
        @add_arg_table arg_settings begin
            "--config_file"
            arg_type = String
            default = "./configs/generalization.toml"
        end
        return arg_settings
    end
    parsed = parse_args(settings(), as_symbols = true)
    run_test(;parsed...)
else
    @warn "In an interactive session, loading but not running run_test.jl"
end
