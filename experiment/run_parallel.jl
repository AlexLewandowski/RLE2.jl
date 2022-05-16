using Pkg
if !isinteractive()
    Pkg.activate(".")
end

using Reproduce
using Pkg.TOML
using FileIO
import Replotuce

function rerun_config(path_to_data_folder)
    path_to_settings = joinpath(path_to_data_folder, "settings.jld2")
    settings = FileIO.load(path_to_settings)

    path_to_toml = dirname(path_to_data_folder) * "/../../settings/"
    files = readdir(path_to_toml)
    toml_files = files[occursin.("toml", files)]

    @assert length(toml_files) == 1

    toml_file = toml_files[1]
    toml = TOML.parsefile(path_to_toml * toml_file)

    parsed = settings["parsed_args"]
    config = toml["config"]

    exp_file = config["exp_file"]
    include(String(@__DIR__) * "/../" * exp_file)
    Base.invokelatest(eval(Meta.parse(config["exp_func_name"])), parsed, test = true)

    println(
        "Experiment with configuration \"",
        path_to_settings,
        "\" successfully completed!",
    )
end

function unfinished_experiments(path_to_experiment)
    path_to_data = joinpath(path_to_experiment, "data/")
    unfinished_paths = []
    for dir in readdir(path_to_data)
        path = joinpath(path_to_data, dir * "/")
        files = readdir(path)
        inds = occursin.("data.jld2", files)
        if sum(inds) == 0
            push!(unfinished_paths, path)
        end
    end
    return unfinished_paths
end

function get_sweep_args(config_path, name)
    if name == "experiment/student_data.jl"
        num_episodes = TOML.parsefile(config_path)["static_args"]["init_num_episodes"]
        sweep_str =
            "seed = \"using Random; rand(MersenneTwister(" *
            #string(rand(1:100000)) *
            string(1) *
            "), 1:10^10, " *
            string(num_episodes) *
            ")\""
        to_break = true
        metric_fn = "gen_ep_dict"
    elseif name == "experiment/student_buffers.jl"
        sweep_str = ""
        if "state_representation" in keys(TOML.parsefile(config_path)["sweep_args"])
            state_representation =
                TOML.parsefile(config_path)["sweep_args"]["state_representation"]
            sweep_str = sweep_str*"state_representation = " * string(state_representation) * "\n"
        elseif "state_representation" in keys(TOML.parsefile(config_path)["static_args"])
            state_representation =
                TOML.parsefile(config_path)["static_args"]["state_representation"]
            sweep_str = sweep_str*"state_representation = " * string(state_representation) * "\n"
        else
            state_representation = "returns"
            sweep_str = sweep_str*"state_representation = " * string(state_representation) * "\n"
        end
        if "recurrent_action_dim" in keys(TOML.parsefile(config_path)["sweep_args"])
            recurrent_action_dim =
                TOML.parsefile(config_path)["sweep_args"]["recurrent_action_dim"]
            sweep_str = sweep_str*"recurrent_action_dim = " * string(recurrent_action_dim) * "\n"
        elseif "recurrent_action_dim" in keys(TOML.parsefile(config_path)["static_args"])
            recurrent_action_dim =
                TOML.parsefile(config_path)["static_args"]["recurrent_action_dim"]
            sweep_str = sweep_str*"recurrent_action_dim = " * string(recurrent_action_dim) * "\n"
        else
            recurrent_action_dim = 0
            sweep_str = sweep_str*"recurrent_action_dim = " * string(recurrent_action_dim) * "\n"
        end
        to_break = true
        metric_fn = ""
    elseif name == "experiment/random_teacher_data.jl"
        num_episodes = TOML.parsefile(config_path)["static_args"]["init_num_episodes"]
        sweep_str =
            "seed = \"using Random; rand(MersenneTwister(" *
            #string(rand(1:100000)) *
            string(1) *
            "), 1:10^10, " *
            string(num_episodes) *
            ")\""
        to_break = true
        metric_fn = "gen_ep_dict"
    elseif name == "experiment/random_optenv_data.jl"
        num_episodes = TOML.parsefile(config_path)["static_args"]["init_num_episodes"]

        sweep_str = "\n"
        seed =
            TOML.parsefile(config_path)["sweep_args"]["seed"]
        sweep_str = sweep_str*"seed = \"" * string(seed) * "\"\n"
        EnvType =
            TOML.parsefile(config_path)["sweep_args"]["EnvType"]
        sweep_str = sweep_str*"EnvType = " * string(EnvType) * "\n"
        to_break = true
        metric_fn = "gen_ep_dict"
    else
        error("Not a valid name")
    end
    return sweep_str, to_break, metric_fn
end


function create_temp_config_file(config_path, n = 1)
    try
        mkpath(".parallel_temp" * string(n))
    catch
    end

    temp_toml_path = joinpath(".parallel_temp" * string(n), splitpath(config_path)[end])

    pre_exp_name = TOML.parsefile(config_path)["config"]["pre_exp_files"][n]

    open(config_path) do file
        open(temp_toml_path, "w") do file_w
            sweep_args, to_break, metric_fn = get_sweep_args(config_path, pre_exp_name)
            i = 1
            for ln in eachline(file)
                if i == 3
                    write(
                        file_w,
                        "exp_file =\"" *
                        TOML.parsefile(config_path)["config"]["pre_exp_files"][n] *
                        "\"\n",
                    )
                elseif i == 8
                    write(file_w, "post_exp = \"" * metric_fn * "\"\n")
                elseif contains(ln, "sweep_args")
                    write(file_w, ln * "\n")
                    if to_break
                        write(file_w, sweep_args)
                        break
                    end
                else
                    write(file_w, ln * "\n")
                end
                i += 1
            end
        end
    end

    return temp_toml_path
end

function create_temp_config_file(config_path; force_version = nothing)
    save_dir = TOML.parsefile(config_path)["config"]["save_dir"]

    if occursin("version", splitpath(config_path)[end-1])
        save_dir = joinpath(splitpath(config_path)[1:end-2]...)
    end

    if occursin("version", save_dir)
        save_dir = joinpath(splitpath(save_dir)[1:end-1]...)
    end

    if force_version === nothing
        i = 0
        old_version = "version_" * string(i)
        if isdir(save_dir)
            files = readdir(save_dir)
            inds = ["version_" == file[1:8] for file in files]
            list_of_vers = [parse(Int, split(f, "version_")[end]) for f in files[inds]]
            if !isempty(list_of_vers)
                i =
                    maximum([parse(Int, split(f, "version_")[end]) for f in files[inds]]) + 1
            end
            old_version = "version_" * string(i)
        end
    else
        old_version = force_version
    end

    temp_toml_path = joinpath(".parallel_temp", splitpath(config_path)[end])

    open(config_path) do file
        open(temp_toml_path, "w") do file_w
            i = 1
            for ln in eachline(file)
                if i == 2
                    write(file_w, ln[1:end-1] * "/" * old_version * "\"\n")
                else
                    write(file_w, ln * "\n")
                end
                i += 1
            end
        end
    end

    return temp_toml_path
end

function run_parallel(
    config_path,
    num_workers = 2;
    plots = false,
    use_diff = false,
    force_version = nothing,
    pre_exp = :ignore,
)
    try
        mkpath(".parallel_temp")
    catch
    end

    if occursin("version", splitpath(config_path)[end-1]) && use_diff == true
        files = readdir(joinpath(splitpath(config_path)[1:end-1]...))
        diff_ind = argmax(occursin.(".diff", files))
        commit_id = split(files[diff_ind], "_0x")[1]
        diff_path = joinpath(splitpath(config_path)[1:end-1]..., files[diff_ind])
        new_diff_path = joinpath(".parallel_temp", files[diff_ind])
        println(diff_path)
        println(new_diff_path)
        cp(diff_path, new_diff_path, force = true)
    elseif use_diff == true
        error("There is no diff file to use!")
    end

    save_dir = TOML.parsefile(config_path)["config"]["save_dir"]

    if pre_exp !== :ignore

        pre_exp_files = []
        try
            pre_exp_files = TOML.parsefile(config_path)["config"]["pre_exp_files"]
        catch
        end

        n = 1
        for pre_exp_file in pre_exp_files
            pre_exp_file_name = splitpath(pre_exp_file)[end][1:end-3]
            temp_config_path = create_temp_config_file(config_path, n)
            data_dir = joinpath(save_dir, pre_exp_file_name)
                run_parallel(
                    temp_config_path,
                    num_workers;
                    plots = plots,
                    use_diff = use_diff,
                    force_version = pre_exp_file_name,
                    pre_exp = :ignore,
                )
            n += 1
        end

        if pre_exp == :only_pre
            return 0
        end
    end

    config_path = create_temp_config_file(config_path, force_version = force_version)

    if use_diff == true
        read(`git stash`)
        HEAD = read(`git rev-parse --short HEAD`, String)[1:end-1]
        read(`git clean -fd src/`)
        read(`git reset --hard`)
        read(`git checkout $commit_id`)
        if !(filesize(new_diff_path) == 0)
            read(`git apply $new_diff_path`)
        end
    end

    save_dir = TOML.parsefile(config_path)["config"]["save_dir"]
    experiment = Experiment(config_path)
    create_experiment_dir(experiment; tldr = "tldr", replace = false)

    ###
    ### Generate diff
    ###

    diff_name = String(read(`git rev-parse --short HEAD`))[1:end-1]
    diff_file = read(`git diff src/`)
    readbuf = IOBuffer(reinterpret(UInt8, diff_file))

    hash = string(experiment.hash, base = 16)
    hash_filename = diff_name * "_0x" * hash * "_1x" * ".diff"

    diff_exp = read(`git diff experiment/`)
    readbuf_exp = IOBuffer(reinterpret(UInt8, diff_exp))

    new_files = read(`./git_diff.sh`)
    readbuf_new = IOBuffer(reinterpret(UInt8, new_files))
    open(joinpath(save_dir, hash_filename), "w") do io
        write(io, String(readuntil(readbuf, 0x00)))
        write(io, String(readuntil(readbuf_new, 0x00)))
        write(io, String(readuntil(readbuf_exp, 0x00)))
    end

    ###
    ### Run Experiment
    ###

    add_experiment(experiment)
    ret = job(experiment; num_workers = num_workers)
    post_experiment(experiment, ret)

    ###
    ### Post Experiment
    ###
    ###
    post_exp = "gen_metric_dict"
    try
        post_exp = TOML.parsefile(config_path)["config"]["post_exp"]
    catch
    end

    if post_exp == "gen_ep_dict"
        plots = false
    end

    if !isempty(post_exp)
        dict_f = getfield(Replotuce, Symbol(post_exp))

        Replotuce.get_results(results_dir = save_dir, plot_results = plots, dict_f = dict_f)
    end


    ###
    ### Post experiment diff
    ###

    if use_diff
        read(`git reset --hard`)
        read(`git checkout $HEAD`)
        HEAD = read(`git rev-parse HEAD`, String)[1:end-1]
        master = read(`git rev-parse master`, String)[1:end-1]
        if HEAD == master
            read(`git checkout master`)
        end
        try
            read(`git stash pop`)
        catch
            @warn "No stashes!"
        end
    end
    return 0
end

##
## If run outside of a REPL:
##

function run_parallel(; config_path, num_workers, plots, use_diff, pre_exp, force_version)
    run_parallel(config_path, num_workers, plots = plots, use_diff = use_diff, pre_exp = pre_exp, force_version = force_version)
end

if !isinteractive()
    function settings(arg_settings::ArgParseSettings = ArgParseSettings())
        @add_arg_table arg_settings begin

            "--config_path"
            arg_type = String
            default = "./configs/generalization.toml"

            "--num_workers"
            arg_type = Int
            default = 10

            "--plots"
            arg_type = Bool
            default = false

            "--use_diff"
            arg_type = Bool
            default = false

            "--pre_exp"
            arg_type = Symbol
            default = :run

            "--force_version"
            default = nothing
        end
        return arg_settings
    end

    parsed = parse_args(settings(), as_symbols = true)
    run_parallel(; parsed...)
else
    @warn "In an interactive session, loading but not running run_parallel.jl"
end
