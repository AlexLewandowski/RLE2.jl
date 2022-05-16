using RLE2
using Random
using Reproduce

import Flux
import CUDA
import JLD2

using Flux.Optimise: ADAM, RMSProp, Optimiser, ExpDecay

include("helper_functions.jl")

function run_experiment(parsed::Dict; test = false)
    if !test
        create_info!(parsed, parsed["save_dir"])
    end

    seed = parsed["seed"]

    num_grad_steps = parsed["num_grad_steps"]
    num_env_steps = parsed["num_grad_steps"]
    num_episodes = parsed["num_episodes"]
    max_num_episodes = parsed["max_num_episodes"]
    max_episode_length = parsed["max_episode_length"]

    init_policy = Symbol(parsed["init_policy"])
    init_num_episodes = parsed["init_num_episodes"]

    predict_window = parsed["predict_window"]
    history_window = parsed["history_window"]

    hidden_size = parsed["hidden_size"]
    num_layers = parsed["num_layers"]
    activation = getfield(Flux, Symbol(parsed["activation"]))

    update_freq = parsed["update_freq"]

    overlap = parsed["overlap"]
    gamma = Float32(parsed["gamma"])

    AgentType = parsed["AgentType"]
    EnvType = parsed["EnvType"]

    optimizer = getfield(Flux.Optimise, Symbol(parsed["optimizer"]))
    lr = parsed["lr"]
    batch_size = parsed["batch_size"]

    force = Symbol(parsed["force"])

    gpu = parsed["gpu"]

    # Need better way to specify optional args / specific to experiment
    state_representation = nothing
    try
        state_representation = Symbol(parsed["state_representation"])
    catch
    end

    online_cbs = RLE2.preprocess_cb(parsed["online_cbs"])
    offline_cbs = RLE2.preprocess_cb(parsed["offline_cbs"])

    if gpu
        gpu_devices = collect(CUDA.devices())
        num_devices = length(gpu_devices)
        gpu_ind = Base.mod1(Reproduce.myid(), num_devices)
        println(Reproduce.myid())
        println(gpu_devices)
        println(gpu_ind)
        device = Flux.gpu
    else
        device = Flux.cpu
    end

    Random.seed!(2 * seed + 1)

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
    # test_buffer = deepcopy(train_buffer)
    # test_buffer.name = "test_buffer"

    total_reports = 20

    if num_episodes < total_reports
        total_reports = num_episodes
    end

    metric_freq = floor(Int, num_episodes / total_reports)

    parsed["total_reports"] = total_reports

    ##
    ## Agent
    ##

    buffers = (train_buffer = train_buffer,)# test_buffer = test_buffer)

    agent = RLE2.get_agent(
        AgentType,
        buffers,
        env,
        metric_freq,
        max_agent_steps,
        online_cbs,
        gamma,
        update_freq,
        update_cache,
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
        reg,
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
        return_early = parsed["return_early"]
    catch
    end
    if return_early
        return agent, env
    end


    ##
    ## Training
    ##

    RLE2.reset!(env)
    calculate_and_log_metrics(agent, env, agent.list_of_cbs, agent.cb_dict, parsed["_SAVE"])

    for i = 1:num_episodes
        RLE2.reset!(env)
        RLE2.train_subagents(agent, i, reg = reg)
        if force !== :offline
            step = 1
            done = RLE2.get_terminal(env)
            while !done
                RLE2.train_subagents(agent, step, reg = reg)
                for _ = 1:num_env_steps
                    done = RLE2.interact!(env, agent, false, buffer = train_buffer)
                    if step == max_agent_steps || done
                        RLE2.finish_episode(train_buffer)
                        done = true
                        break
                    end
                    step += 1
                end
            end
        end
        calculate_and_log_metrics(
            agent,
            env,
            agent.list_of_cbs,
            agent.cb_dict,
            parsed["_SAVE"],
        )
    end

    online_dict = agent.cb_dict

    # RLE2.save_agent(agent, joinpath(parsed["_SAVE"]))
    JLD2.@save joinpath(parsed["_SAVE"], "data.jld2") parsed online_dict
    return agent, env
end