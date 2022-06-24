using RLE2
using Random
using Reproduce

import Flux
import CUDA
import JLD2

using Flux.Optimise: ADAM, RMSProp, Optimiser, ExpDecay

include("helper_functions.jl")

function run_experiment(config::Dict; test = false)
    if !test
        create_info!(config, config["save_dir"])
    end

    seed = config["seed"]

    num_grad_steps = config["num_grad_steps"]
    num_env_steps = config["num_grad_steps"]
    num_episodes = config["num_episodes"]
    max_num_episodes = config["max_num_episodes"]
    max_episode_length = config["max_episode_length"]

    init_policy = Symbol(config["init_policy"])
    init_num_episodes = config["init_num_episodes"]
    behavior = Symbol(config["behavior"])

    predict_window = config["predict_window"]
    history_window = config["history_window"]

    hidden_size = config["hidden_size"]
    num_layers = config["num_layers"]
    activation = getfield(Flux, Symbol(config["activation"]))

    update_freq = config["update_freq"]

    overlap = config["overlap"]
    gamma = Float32(config["gamma"])
    drop_rate = config["drop_rate"]
    reg = Float32(config["reg"])

    AgentType = config["AgentType"]
    EnvType = config["EnvType"]

    optimizer = getfield(Flux.Optimise, Symbol(config["optimizer"]))
    lr = config["lr"]
    batch_size = config["batch_size"]

    force = Symbol(config["force"])

    gpu = config["gpu"]

    kwargs_dict = Dict()
    # Need better way to specify optional args / specific to experiment
    state_representation = nothing
    try
        state_representation = Symbol(config["state_representation"])
    catch e
        println(e)
    end

    pooling_func = :mean
    try
        pooling_func = Symbol(config["pooling_func"])
    catch e
        println(e)
    end
    kwargs_dict[:pooling_func] = pooling_func

    measurement_funcs = preprocess_list_of_funcs(config["measurement_funcs"])
    callback_funcs = preprocess_list_of_funcs(config["callback_funcs"])

    if gpu
        try
        gpu_devices = collect(CUDA.devices())
        num_devices = length(gpu_devices)
        gpu_ind = Base.mod1(Reproduce.myid(), num_devices)
        println("Reproduce ID: ", Reproduce.myid())
        println("Device selected: ", gpu_ind)
        println("List of devices: ", gpu_devices)
        Flux.device!(gpu_devices[gpu_ind])
        device = Flux.gpu
        catch  e
            @warn "Caught error:"
            showerror(stdout, e)
            println()
            device = Flux.cpu
        end
    else
        device = Flux.cpu
    end

    if contains(EnvType, "OptEnv") && state_representation == :parameters && contains(callback_funcs)
        num_grad_steps = 0
        max_num_episodes = 10
    end


    Random.seed!(2 * seed + 1)
    if contains(EnvType, "OptEnv") && state_representation == :parameters && RLE2.optimize_student in callback_funcs
        num_grad_steps = 0
        max_num_episodes = batch_size + 1
    end

    env, max_agent_steps, embedding_f = RLE2.get_env(
        EnvType,
        seed = seed,
        max_steps = max_episode_length,
        state_representation = state_representation,
    )

    ##
    ## Replay Buffer
    ##

    buffer_rng = MersenneTwister(3 * seed - 1)

    train_buffer = TransitionReplayBuffer(
        env,
        max_num_episodes,
        max_episode_length,
        batch_size,
        gamma,
        history_window = history_window,
        predict_window = predict_window,
        overlap = overlap,
        rng = buffer_rng,
        name = "train_buffer",
    )

    # test_buffer is the buffer used for testing
    # same distribution as train but independent
    test_buffer = deepcopy(train_buffer)
    test_buffer.name = "test_buffer"

    total_reports = 100

    if num_episodes < total_reports
        total_reports = num_episodes
    end

    measurement_freq = floor(Int, num_episodes / total_reports)

    config["total_reports"] = total_reports

    ##
    ## Agent
    ##

    buffers = (train_buffer = train_buffer, test_buffer = test_buffer)
    # buffers = (train_buffer = train_buffer,)

    println(kwargs_dict)
    agent = RLE2.get_agent(
        AgentType,
        buffers,
        env,
        measurement_freq,
        max_agent_steps,
        measurement_funcs,
        gamma,
        update_freq,
        predict_window,
        history_window,
        num_layers,
        hidden_size,
        activation,
        drop_rate,
        optimizer,
        lr,
        device,
        num_grad_steps,
        force,
        seed,
        reg;
        behavior = behavior,
        kwargs_dict...,
    )

    for buffer in buffers
        populate_replay_buffer!(
            buffer,
            env,
            agent,
            policy = init_policy,
            num_episodes = init_num_episodes,
            max_steps = max_agent_steps,
            greedy = false,
        )
    end

    return_early = false
    try
        return_early = config["return_early"]
    catch
    end
    if return_early
        return agent, env
    end


    ##
    ## Training
    ##

    RLE2.reset!(env)
    calculate_and_log_metrics(agent, env, agent.measurement_funcs, agent.measurement_dict, config["_SAVE"])

    for i = 1:num_episodes
        RLE2.train_subagents(agent, reg = reg)
        if force !== :offline
            step = 1
            done = RLE2.get_terminal(env)
            while !done
                RLE2.train_subagents(agent, reg = reg)
                for _ = 1:num_env_steps
                    done = RLE2.interact!(env, agent, greedy = false, buffer = train_buffer)
                    if step == max_agent_steps || done
                        RLE2.finish_episode(train_buffer)
                        done = true
                        break
                    end
                    step += 1
                end
            end
            RLE2.reset!(env)
        end
        [callback_f(agent, env) for callback_f in callback_funcs]
        calculate_and_log_metrics(
            agent,
            env,
            agent.measurement_funcs,
            agent.measurement_dict,
            config["_SAVE"],
        )
    end

    measurements = agent.measurement_dict

    JLD2.@save joinpath(config["_SAVE"], "data.jld2") config measurements

    return agent, env
end
