include("approximator_metrics.jl")
include("subagent_metrics.jl")
include("agent_metrics.jl")
include("env_metrics.jl")

function calculate_metrics(
    agent::AbstractAgent,
    env::AbstractEnv,
)
    calculate_metrics(agent, env, agent.measurement_funcs, "")
end


function calculate_metrics(
    agent::AbstractAgent,
    env::AbstractEnv,
    measurement_funcs,
    path = nothing;
    print_results = true,
)
    # if path !== nothing
    #     save_agent(agent, path)
    # end

    old_device = agent.device
    agent.device = Flux.cpu
    # to_device!(agent.π_b, agent.device)
    to_device!(agent, agent.device)

    results_tuple = []
    for f in measurement_funcs
        reset_model!(agent.state_encoder, :zeros)
        result_tuple = f(agent, env)
        if result_tuple !== nothing
            results, names= result_tuple
            if print_results
                println(names)
                println(results)
            end
            @assert length(results) == length(names)
            @assert prod([typeof(name) == String for name in names]) == true f
            @assert prod([isa(result, Number) for result in results]) == true f
            if sum(isnan.(results)) > 0
                println(f)
                println(path)
                save_agent(agent, path)
                error("NaN in results!")
            end
            push!(results_tuple, zip(results, names))
        end
    end
    if print_results
        println()
    end

    agent.device = old_device
    # to_device!(agent.π_b, agent.device)
    to_device!(agent, agent.device)
    return results_tuple
end

function log_metrics(results_tuple, measurement_dict, count; init = false)
    # if init
    #     for result_tuple in results_tuple
    #         for (result, name) in result_tuple
    #             if name === nothing
    #             else
    #                 result = (result, count)
    #                 measurement_dict[name] = zeros(Float32, init)
    #                 measurement_dict[name][count] = result
    #                 # push_dict!(measurement_dict, name, result)
    #             end
    #         end
    #     end
    # else
    for result_tuple in results_tuple
        for (result, name) in result_tuple
            if name === nothing
            else
                result = (result, count)
                push_dict!(measurement_dict, name, result)
            end
        end
    end
    # end
end

function calculate_and_log_metrics(
    agent::AbstractAgent,
    env::AbstractEnv,
    measurement_funcs = nothing,
    measurement_dict = nothing,
    path = nothing;
    force = false,
    init = false,
)
    if measurement_dict === nothing
        measurement_dict = agent.measurement_dict
    end
    if Base.mod(agent.measurement_count, agent.measurement_freq) == 0 || force == true
        println("Agent measurement_count is: ", agent.measurement_count)
        flush(stdout)
        results_tuple = calculate_metrics(agent, env, measurement_funcs, path)
        log_metrics(results_tuple, measurement_dict, agent.measurement_count, init = true)
    else
        if online_returns in measurement_funcs
            results_tuple = calculate_metrics(agent, env, [online_returns], path, print_results = false)
            log_metrics(results_tuple, measurement_dict, agent.measurement_count)
        end
    end
    agent.measurement_count += 1
    return nothing
end


function apply_to_list(f, list, args...; kwargs...)
    results = Float64[]
    names = String[]
    for s in list
        if isa(s, Tuple)
            result_tuple = f(s..., args...; kwargs...)
        else
            result_tuple = f(s, args...; kwargs...)
        end
        if !isnothing(result_tuple)
            result, name = result_tuple
            push!(results, result...)
            push!(names, name...)
        end
    end
    return results, names
end

function get_optimal_dist(agent, env)
    ss = []
    for _ = 1:1
        ep = generate_episode(
            env,
            agent,
            policy = :optimal,
            greedy = false,
            max_steps = 100,
        )
        s = [e.s for e in ep]
        push!(ss, s...)
    end
    valid_states = get_valid_nonterminal_states(env.MDP)
    n_states = length(get_valid_nonterminal_states(env.MDP))
    total = length(ss)
    train_dist = zeros(Float32, n_states)
    for i = 1:n_states
        train_dist[i] = sum(ss .== valid_states[i])
    end
    train_dist = train_dist / total
end

function get_policy_dist(agent, env)
    ss = []
    for i = 1:30
        ep = generate_episode(
            env,
            agent,
            policy = :agent,
            greedy = false,
            max_steps = 1000,
        )
        s = [e.s for e in ep]
        push!(ss, s...)
    end
    valid_states = get_valid_nonterminal_states(env.MDP)
    n_states = length(get_valid_nonterminal_states(env.MDP))
    total = length(ss)
    train_dist = zeros(Float32, n_states)
    for i = 1:n_states
        train_dist[i] = sum(ss .== valid_states[i])
    end
    train_dist = train_dist / total
end

function get_train_dist(agent, env)
    train_buffer = agent.buffers.train_buffer
    N = minimum([curr_size(train_buffer), 10000])
    StatsBase.sample(train_buffer, N)
    ss, o, a, p, r, sp, op, done, info  = get_batch(train_buffer)
    valid_states = get_valid_nonterminal_states(env.MDP)
    n_states = length(get_valid_nonterminal_states(env.MDP))
    total = length(ss)
    train_dist = zeros(Float32, n_states)
    for i = 1:n_states
        train_dist[i] = sum(ss .== valid_states[i])
    end
    train_dist = train_dist / total
end

function get_list_for_metric(agent, env)

    V = env.MDP.V
    Q = get_optimal_Q(env.MDP)
    A = get_optimal_action(env.MDP)

    # eval_list, extra_list = get_list_for_metric(env)
    eval_list = env.eval_list[1]
    extra_list = env.eval_list[2]
    list_of_vars = []
    for key in keys(eval_list)
        if key == :train #TODO best way to get dataset??
            xs_train, ys_train = getfield(eval_list, :train)
            buffer = agent.buffers.train_buffer
            n_states = size(env.MDP.reward_mat)[1]
            xs = [[] for i = 1:n_states]
            ys = [[] for i = 1:n_states]
            num_episodes = sum(buffer._episode_lengths .> 0)
            for i = 1:num_episodes
                ep_len = buffer._episode_lengths[i]
                for j = 1:ep_len
                    ep_s = buffer._episodes[i][j].s
                    push!(xs[ep_s], buffer._episodes[i][j].o)
                    push!(ys[ep_s], buffer._episodes[i][j].s)
                end
            end
            valid_states = get_valid_nonterminal_states(env.MDP)
            xs = xs[valid_states]
            ys = ys[valid_states]
            for i = 1:length(ys)
                if length(ys[i]) > 0
                    s = valid_states[i]
                    push!(list_of_vars, [hcat(xs[i]...), ys[i], s, V, Q, A, string(key) * "_state_" * string(s)])
                else
                    s = valid_states[i]
                    push!(list_of_vars, [xs_train[i], ys_train[i], s, V, Q, A, string(key) * "_state_" * string(s)])
                end
            end
        else
            xs, ys = getfield(eval_list, key)
            for i = 1:length(ys)
                s = ys[i][1]
                push!(list_of_vars, [xs[i], ys[i], s, V, Q, A, string(key) * "_state_" * string(s)])
            end
        end
    end
    if !isnothing(extra_list)
        for key in keys(extra_list)
            xs, ys = getfield(extra_list, key)
            for i in 1:length(ys)
                s = ys[i][1]
                push!(list_of_vars, [xs[i], ys[i], s, V, Q, A, string(key)*"_wall_"*string(s)])
            end
        end
    end

    list = collect(Iterators.product(agent.subagents, list_of_vars)) |> vec

    reset!(env)
    start_state = env.state

    train_dist = get_train_dist(agent, env)
    policy_dist = get_policy_dist(agent, env)
    optimal_dist = get_optimal_dist(agent, env)

    return list, train_dist, policy_dist, optimal_dist, start_state
end

function student_optimization_metrics(agent, env; n_steps = 20)
    reset!(env)
    println("V init: ", sum(agent.subagents[1](get_next_obs_with_f(env, env.f) |> agent.device)))
    optimize_student(env, agent, n_steps = n_steps)
    println("V post: ", sum(agent.subagents[1](get_next_obs_with_f(env, env.f) |> agent.device)))
end
