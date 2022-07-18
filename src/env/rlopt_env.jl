abstract type AbstractRLOptEnv <: AbstractOptEnv end

mutable struct RLOptEnv <: AbstractRLOptEnv
    env::AbstractEnv
    agent::AbstractAgent
    state_representation::Any

    t::Any
    state::Any
    obs::Any
    reward::Any
    done::Any

    init_state::Any

    max_steps::Any
    device::Any
    rng::Any
end

function RLOptEnv(
    env,
    state_representation,
    max_steps;
    device = Flux.cpu,
    stationary = false,
    rng = Random.GLOBAL_RNG,
    kwargs...,
)

    agent = get_agent(env)

    for buffer in agent.buffers
        populate_replay_buffer!(
            buffer,
            env,
            agent,
            policy = :random,
            # policy = :agent,
            num_episodes = 10,
            max_steps = agent.max_agent_steps,
            greedy = false,
        )
    end

    t = 0
    state = nothing
    obs = nothing
    reward = nothing
    done = false

    if stationary
        loop = step_loop
    else
        loop = train_loop
    end
    init_state = [nothing, nothing, 1, 1, loop]
    env = RLOptEnv(
        env,
        agent,
        string(state_representation),
        t,
        state,
        obs,
        reward,
        done,
        init_state,
        max_steps,
        device,
        rng,
    )
    reset!(env)
    return env
end

function step_loop(env::AbstractRLOptEnv; greedy = false, add_exp = false)
    exp, done = RLE2.interact!(env.env, env.agent, greedy = greedy)

    buffer = env.agent.buffers.train_buffer

    if add_exp
        add_exp!(buffer, exp)
        if done
            RLE2.finish_episode(buffer)
        end
    end

    return exp, done
end

function train_loop(env::AbstractRLOptEnv; greedy = false, add_exp = false)
    exp, done = RLE2.interact!(env.env, env.agent, greedy = greedy)

    buffer = env.agent.buffers.train_buffer

    if add_exp
        add_exp!(buffer, exp)
        if done
            RLE2.finish_episode(buffer)
        end
    end

    # train_subagents(env.agent, resample = true)

    ep = [exp]
    ep = pad_episode!(buffer, ep, 2)
    init_batches(buffer, 1)
    fill_buffer!(buffer, ep, 1)
    train_subagents(env.agent, resample = false)

    return exp, done
end

function reset!(env::AbstractRLOptEnv; saved_f = false, greedy = true)
    env.env.rng = MersenneTwister(env.init_state[3])
    env.done = false

    env.t = 1

    buffer = env.agent.buffers.train_buffer

    env.agent = get_agent(env.env, env.init_state[3])
    env.agent.buffers = (train_buffer = buffer,)

    agent = env.agent

    if saved_f == true && !isnothing(env.init_state[1])
        ps, re = Flux.destructure(env.init_state[1].f)
        if !greedy
            ps = [p .+ 0.01f0*randn(Float32, size(p)) for p in ps]
        end
        f = NeuralNetwork(re(ps))
        env.agent.subagents[1].model.f = f
        env.agent.subagents[1].params = f.params
        f = NeuralNetwork(re(ps))
        env.agent.subagents[1].target.sp_network.f.f = f.f
        env.agent.subagents[1].target.sp_network.f.params = f.params
    end

    # for buffer in agent.buffers
    #     populate_replay_buffer!(
    #         buffer,
    #         env.env,
    #         agent,
    #         policy = :random,
    #         num_episodes = 1,
    #         max_steps = agent.max_agent_steps,
    #         greedy = false,
    #     )
    # end

    reset!(env.env)
    # env.env.state = Float32(0.1) * rand(MersenneTwister(env.init_state[3]), Float32, 4) .- Float32(0.05)
    # env.env.state = Float32(0.1) * rand(env.rng, Float32, 4) .- Float32(0.05)
    # println(env.env.state)
    env.obs = get_next_obs(env)
    env.init_state[3] = env.init_state[3] + 1
end

Base.show(io::IO, t::MIME"text/plain", env::AbstractRLOptEnv) = begin
    println()
    println("---------------------------")
    name = "OptEnv"
    L = length(name)
    println(name)
    println(join(["-" for i = 1:L+1]))

    print(io, "  env.env: ")
    println(io, typeof(env.env))

    print(io, "  env.agent: ")
    println(io, typeof(env.agent))
    println("---------------------------")
end

function (env::AbstractRLOptEnv)(a)
    reset_experience!(env.agent.buffers.train_buffer)
    done = false
    exp = 1
    G = 0f0
    # s = deepcopy(env.env.state)
    # println("F pre: ", env.agent.subagents[1](s))
    # loss = mc_buffer_loss
    # println("Loss pre: ", loss(env.agent, env.env, batch_size = -1))
    while !done
        loop = env.init_state[5]
        if contains(env.state_representation, "xon")
            exp, done = loop(env, add_exp = true)
        else
            exp, done = loop(env, add_exp = false)
        end
        r = exp.r
        # r = -mc_buffer_loss(env.agent, env.env)[1][1]
        G += r
    end
    # G = -buffer_loss(env.agent, env.env, batch_size = -1)[1][1]
    # println("F post: ", env.agent.subagents[1](s))
    # println("Loss post: ", loss(env.agent, env.env, batch_size = -1))

    env.reward = G/env.max_steps
    # println(env.reward)
    # println("G: ", G)
    if done
        env.done = true
    end
    env.t += 1
    env.obs = get_next_obs(env)
end

function get_actions(env::AbstractRLOptEnv)
    return Base.OneTo(1)
end

function default_action(env::AbstractRLOptEnv, state)
    return 1, 1.0f0
end

function optimal_action(env::AbstractRLOptEnv, state)
    return 1, 1.0f0
end


function get_info(env::AbstractRLOptEnv)
    return nothing
end

function get_next_obs(env::AbstractRLOptEnv)
    return get_next_obs_with_f(env, env.agent.subagents[1].model.f)
    # return [0]
end

function get_next_obs_with_f(
    env::AbstractRLOptEnv,
    f;
    M = nothing,
    task_id = nothing,
    xy = nothing,
    t = nothing,
)
    if isnothing(t)
        t = env.t
    end

    N = RLE2.curr_size(env.agent.buffers.train_buffer)

    state_rep_str = stop_gradient() do
        split(env.state_representation, "_")
    end

    # aux = get_state(env.env)
    aux = [1f0]

        # aux = stop_gradient() do
        #     Float32.([t...])
        # end
    # aux_dim = length(aux)
    aux_dim = 0


    if state_rep_str[1] == "parameters"
        # Can be differentiated
        P, re = Flux.destructure(f.f)

        return vcat(P, aux) |> env.device

    elseif state_rep_str[1] == "nothing"
        # Can be differentiated
        return aux

    elseif state_rep_str[1] == "PVN"
        # Can be differentiated
        P, re = Flux.destructure(f.f)
        return P

    elseif state_rep_str[1] == "oblivious"
        acc = copy(log(-log(env.acc[1])))
        acc_test = copy(log(-log(env.acc[3])))
        aux = Float32.([log(env.opt.eta), t, acc, acc_test])
        return aux

    elseif contains(state_rep_str[1], "PE")
        # if isnothing(task_id)
        #     N2 = Int(N / env.n_tasks)
        #     ind_range = env.ind_range
        #     task_id = Int(collect(ind_range)[end] / N2)
        # else
        #     N2 = Int(N / env.n_tasks)
        #     ind_range = (1+N2*(task_id-1)):N2*task_id
        # end

        if isnothing(M)
            if length(state_rep_str) == 1
                M = stop_gradient() do
                    minimum([256, env.data_size])
                end
            else
                M = stop_gradient() do
                    parse(Int, state_rep_str[2])
                end
            end
        elseif M == :all
            M = N
        elseif typeof(M) <: Int
            M = M
        else
            error()
        end

        env2 = deepcopy(env.env)
        stop_gradient() do
            reset!(env2)
        end
        done = false
        # stop_gradient() do
        #     for i = 1:5
        # while !done
        #     exp, done = RLE2.interact!(env2, env.agent, policy = :random)
        #     add_exp!(env.agent.buffers.train_buffer, exp)
        #     if done
        #         finish_episode(env.agent.buffers.train_buffer)
        #         reset!(env2)
        #     end
        # end
        #         end
        # end

            if state_rep_str[1] !== "PE-xon"
        while curr_size(env.agent.buffers.train_buffer) < M
            # if state_rep_str[1] == "PE-xon"
            #     # exp, done = RLE2.interact!(env2, env.agent, policy = :agent)
            # else
                exp, done = RLE2.interact!(env2, env.agent, policy = :random)
            add_exp!(env.agent.buffers.train_buffer, exp)
            if done
                finish_episode(env.agent.buffers.train_buffer)
                reset!(env2)
            end
            if curr_size(env.agent.buffers.train_buffer) == M
                finish_episode(env.agent.buffers.train_buffer)
            end
            # end
        end
                end


        data = stop_gradient() do
            StatsBase.sample(env.agent.buffers.train_buffer, M, replacement = true)
            get_batch(env.agent.buffers.train_buffer, env.agent.subagents[1])
        end

        x_ = data[2][:,1,:]
        a = data[3][:,1,:]
        x_sp = data[7][:,1,:]
        y = data[5][:,1,:]
        done = data[8][:,1,:]
        mask = data[end-1][:,1,:]
        maskp = data[end][:,1,:]

        y_ = f(x_)
        y_sp = f(x_)

        # y_ = softmax(f(x_))
        # y_ = Float32.(RLE2.get_greedypolicy(f(x_)))

        # t = stop_gradient() do
        #     positional_encoding([t])
        # end

        # aux = stop_gradient() do
        #     Float32.([t...])
        # end

        # aux_dim = length(aux)

        meta_data_list = []

        if state_rep_str[1] == "PE-xy"
            meta_data_list = [x_, y]

        elseif state_rep_str[1] == "PE-0"
            # Only outputs needed, handled in outer code
            # elseif state_rep_str[1] == "PE-0-grad"

        elseif state_rep_str[1] == "PE-x"
            meta_data_list = [x_]

        elseif state_rep_str[1] == "PE-xon"
            meta_data_list = [x_]

        elseif state_rep_str[1] == "PE-td"

            y_sp = reshape(
                maximum(y_sp, dims = 1),
                size(y),
            )
            y_sp = y_sp .* (1 .- done)
            y_ = sum(y_ .* mask, dims = 1)
            # y_sp = sum(y_sp .* maskp)
            # y_sp = softmax(f(x_sp)).*(1 .- done)
            # y_sp = f(x_sp).*(1 .- done)
            # println(y_sp)
            # println(sum(done))
            # y_sp = f(x_sp)
            meta_data_list = [y_sp, y, x_, x_sp]

        elseif state_rep_str[1] == "PE-xysp"
            y_sp = f(x_sp)
            meta_data_list = [x_, x_sp, y, y_sp, a]

        elseif state_rep_str[1] == "PE-xya"
            ahat = softmax(y_)
            meta_data_list = [x_, y, a, ahat]

        elseif state_rep_str[1] == "PE-ysp"
            y_sp = f(x_sp)
            meta_data_list = [y, y_sp]

        elseif state_rep_str[1] == "PE-y"
            meta_data_list = [y]

            #     for i = 1:1
            #         inds_local = StatsBase.sample(env.rng, 1:N, M, replace = false)
            #         x_local = retrieve_with_inds(env, env.x, inds_local)
            #         y_local = retrieve_with_inds(env, env.y, inds_local)

            #         opt2 = deepcopy(env.opt)
            #         f2 = deepcopy(f)

            #         new_ps = f2.params
            #         old_ps = f.params

            #         if typeof(opt2) <: Flux.ADAM
            #             if !isempty(opt2.state)
            #                 opt2.state = IdDict(
            #                     new_p => (
            #                         deepcopy(env.opt.state[old_p][1]),
            #                         deepcopy(env.opt.state[old_p][2]),
            #                         deepcopy(env.opt.state[old_p][3]),
            #                     ) for (new_p, old_p) in zip(new_ps, old_ps)
            #                 )
            #             end
            #         end

            #         update_f(env, opt2, env.J, f2.f, x_local, y_local)

            #         z_ = f2(x_)
            #         z_ = Z_ - f(x_)

            #         aux_stack = reshape(repeat(aux, M), (aux_dim, M))
            #         z_ = cat(z_, aux_stack, dims = 1)

            #         push!(meta_data_list, z_)
            #     end

            # elseif state_rep_str[1] == "PE-x-grad"
            #     meta_data_list = [x_]
            #     for i = 1:1
            #         inds_local = StatsBase.sample(env.rng, 1:N, M, replace = false)
            #         x_local = retrieve_with_inds(env, env.x, inds_local)# |> env.device
            #         y_local = retrieve_with_inds(env, env.y, inds_local)# |> env.device

            #         opt2 = deepcopy(env.opt)
            #         f2 = deepcopy(f)

            #         new_ps = f2.params
            #         old_ps = f.params

            #         if typeof(opt2) <: Flux.ADAM
            #             if !isempty(opt2.state)
            #                 opt2.state = IdDict(
            #                     new_p => (
            #                         deepcopy(env.opt.state[old_p][1]),
            #                         deepcopy(env.opt.state[old_p][2]),
            #                         deepcopy(env.opt.state[old_p][3]),
            #                     ) for (new_p, old_p) in zip(new_ps, old_ps)
            #                 )
            #             end
            #         end

            #         update_f(env, opt2, env.J, f2.f, x_local, y_local)

            #         z_ = f2(x_)
            #         z_ = z_ - f(x_)

            #         aux_stack = reshape(repeat(aux, M), (aux_dim, M))
            #         z_ = cat(z_, aux_stack, dims = 1)

            #         push!(meta_data_list, z_)
            #     end

            # elseif state_rep_str[1] == "PE-y-grad"
            #     if t == env.max_steps
            #         hist = zeros(size(y_))
            #     else
            #         hist = env.old_f(x_)
            #     end

            #     meta_data_list = [y, hist]

        else
            error("Not valid: ", state_rep_str)
        end

        cat_arr = y_
        x_dim = size(y_)[1]

        for meta_data in meta_data_list
            cat_arr = cat(cat_arr, Float32.(meta_data), dims = 1)
            x_dim += size(meta_data)[1]
        end

        L = 1
        state = reshape(cat_arr, :)
        # state = zeros(Float32, size(state))
        y_dim = 0

        # Max = maximum(abs.(y_))
        # # Max = maximum(abs.(state))
        # if Max > 3
        #     println("Max: ", Max)
        # end
        state =  Float32.(vcat(state, aux, aux_dim, x_dim, y_dim, L, M))
        stop_gradient() do
            StatsBase.sample(env.agent.buffers.train_buffer, env.agent.buffers.train_buffer.batch_size);
        end
        return state

    else
        error("Not a valid state Representation for OptEnv!")
    end

end

using Optim, FluxOptTools
function optimize_value_student(
    agent,
    env::AbstractRLOptEnv;
    n_steps = 10,
    return_gs = false,
    greedy = false,
    cold_start = false,
    all_data = true,
    t = nothing,
    debug = false,
)
    if isnothing(t)
        t = env.max_steps
    end

    RLE2.reset!(env)
    # if !greedy
    # if rand() < 0.1
    #     # println("NO OPT")
    #     # RLE2.reset!(env)
    #     return
    # else
    #     # println("OPT")
    # end
    # end

    # if !greedy
    # if rand() < 0.1
    #     # println("NO OPT")
    #     RLE2.reset!(env, saved_f = true, greedy = false)
    #     return
    # else
    #     # println("OPT")
    # end
    # end

    if isnothing(env.init_state[1])
        f = env.agent.subagents[1].model.f
        opt = Flux.ADAM(0.001)
        # opt = Flux.RMSProp(0.001)
        # opt = Flux.Descent(0.001)
        ps, re = Flux.destructure(f.f)
        f = NeuralNetwork(re(ps))
    else
        f = env.init_state[1]
        opt = env.init_state[2]
        ps = f.params
    end

    gs = nothing
    num_reports = 10
    if n_steps > 10
        report_freq = Int(n_steps / num_reports)
    else
        report_freq = 10
    end
    grad_ps = nothing

    obs = get_next_obs_with_f(env, f);
    # println("SUM V Pre OPT: ", -sum(agent.subagents[1](obs |> agent.device)))

    for i = 1:n_steps
        # if mod(i, report_freq) == 0
        #     if debug
        #         println("V: ", get_mean_v(env, agent, f, deterministic = true))
        #         # println("ACC: ", calc_performance(env, f = f, mode = :train))
        #     end
        # end

        # if env.state_representation == "parameters"
        #     grad_ps = Flux.params(ps)
        # else
            grad_ps = f.params
        # end

        gs = Flux.gradient(() -> begin
                if env.state_representation == "parameters"
                   s = vcat(ps, get_state(env.env))
                else
                   s = get_next_obs_with_f(env, f);
                end
                -sum(agent.subagents[1](s |> agent.device))
            end,
            grad_ps,)

        # for k in keys(gs.grads)
        #     if isa(gs.grads[k], Array)
        #         gs.grads[k] += 0.01f0.*randn(Float32, size(gs.grads[k]))
        #     end
        # end

        Flux.Optimise.update!(opt, grad_ps, gs)

    end

    ps, re = Flux.destructure(f.f)
    env.init_state[1] = f
    env.init_state[2] = opt

    # TODO: Figure out how to handle parameters
    # TODO: Figure out how to best set parameters for subagent
    # if env.state_representation == "parameters"
    #     # opt.state = opt.state[grad_ps]
    #     # env.init_state = [f, opt, iter]
    #     # env.agent.subagents[1].model.f = f
    #     # env.agent.subagents[1].params = f.params
    # else
    #     ps = [p .+ 0.01f0*randn(Float32, size(p)) for p in ps]
    #     f = NeuralNetwork(re(ps))
    #     # f = env.agent.subagents[1].model.f # Continue inner loop training
    #     env.agent.subagents[1].model.f = f
    #     env.agent.subagents[1].params = f.params
    #     f = NeuralNetwork(re(ps))
    #     env.agent.subagents[1].target.sp_network.f.f = f.f
    #     env.agent.subagents[1].target.sp_network.f.params = f.params
    # end
    # env.obs = get_next_obs(env)
    reset!(env, saved_f = true)
    if Base.mod(env.init_state[4], 20) == 0
        println("SUM V POST OPT: ", -sum(agent.subagents[1](env.obs |> agent.device)))
    end
    env.init_state[4] = env.init_state[4] + 1

    if return_gs
        return gs
    end
end

function optimize_student_metrics(
    agent,
    en::AbstractRLOptEnv;
    f = :new,
    value = false,
    est_value = false,
)

    performance = []
    test_performance = []

    adapted_performance = []
    adapted_test_performance = []

    env = deepcopy(en)
    agent = deepcopy(agent)


    if !isnothing(env.init_state[1])
    # println("ini: ", sum([sum(p) for p in env.init_state[1].params]))
    end
    reset!(env, saved_f = true, greedy = true)
    s = deepcopy(env.env.state)
        # println("1: ", sum([sum(p) for p in env.agent.subagents[1].model.f.params]))

    done = false
    G = 0
    count = 0
    while !done
        # println("1: ", sum([sum(p) for p in env.agent.subagents[1].model.f.params]))
        exp, done = step_loop(env, greedy = true)
        G += exp.r
        count += 1
    end



    train_perf = G/env.max_steps
    push!(performance, train_perf)

    reset!(env, saved_f = true, greedy = true)
    done = false
    G = 0
    count = 0

    pre_init_value = env.agent.subagents[1](s |> env.agent.device)

    s_term = nothing
    while !done
        # println("2: ", sum([sum(p) for p in env.agent.subagents[1].model.f.params]))
        s_term = deepcopy(env.env.state)
        exp, done = train_loop(env, greedy = true)
        G += exp.r
        count += 1
    end


    if !isnothing(env.init_state[1])
        post_ps, _ = Flux.destructure(env.agent.subagents[1].model.f.f)
        init_ps, _ = Flux.destructure(env.init_state[1].f)
    else
        post_ps = 0
        init_ps = 0
    end

    post_init_value = env.agent.subagents[1](s |> env.agent.device)

    post_term_value = env.agent.subagents[1](s_term |> env.agent.device)

    reset!(env, saved_f = true)
    pre_term_value = env.agent.subagents[1](s_term |> env.agent.device)


    train_perf = G/env.max_steps
    push!(adapted_performance, train_perf)

    rs = [
        mean(adapted_performance),
        mean(performance),
        mean(LinearAlgebra.norm, init_ps .- post_ps),
        maximum(pre_init_value),
        maximum(post_init_value),
        maximum(pre_term_value),
        maximum(post_term_value),
    ]
    names    = [
        "adapted_performance",
        "nonadapted_performance",
        "norm_diff_params",
        "pre_init_val",
        "post_init_val",
        "pre_term_val",
        "post_term_val",
    ]
    return rs, names
end
