using Pkg.TOML
using Replotuce

function run_test_rng(config_file)

    parsed, sweep_d, static_d, config_d = config_to_test_parsed(config_file)
    try
        rm(parsed["save_dir"], recursive = true)
    catch
    end

    try
        mkpath(parsed["save_dir"])
    catch
    end

    sweep_d["unique_id"] = [1,2]

    open(parsed["save_dir"]*"/config_test.toml", "w") do io
        print(io, "[sweep_args]\n")
        TOML.print(io, sweep_d)
        print(io, "[static_args]\n")
        TOML.print(io, static_d)
        print(io, "[config]\n")
        TOML.print(io, config_d)
    end

    parsed1 = copy(parsed)
    parsed1["unique_id"] = 1
    parsed1["_SAVE"] = parsed1["save_dir"]*"/data/exp_1/"
    try
        mkpath(parsed1["_SAVE"])
    catch
    end
    include(parsed1["exp_file"])
    Base.invokelatest(eval(Meta.parse(parsed1["exp_func_name"])), parsed1, test = true)

    parsed2 = copy(parsed)
    parsed2["unique_id"] = 2
    parsed2["_SAVE"] = parsed2["save_dir"]*"/data/exp_2/"
    try
        mkpath(parsed2["_SAVE"])
    catch
    end
    include(parsed2["exp_file"])
    Base.invokelatest(eval(Meta.parse(parsed2["exp_func_name"])), parsed2, test = true)

    a = get_dicts(results_dir = parsed["save_dir"])
    dict_keys = collect(keys(a[1]))
    return_dict = a[2]
    metric_keys = collect(keys(return_dict))

    for metric_key in metric_keys
        val1 = return_dict[metric_key][dict_keys[1]][1][1];
        val2 = return_dict[metric_key][dict_keys[2]][1][1];
        try
            @assert (val1 == val2);
        catch
            println("Seed does not control RNG for metric key: ", metric_key)
        end
    end
    0
end
