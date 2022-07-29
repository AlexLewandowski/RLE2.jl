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
    explore = false,
    rng = Random.GLOBAL_RNG,
    kwargs...,
)

    internal_seed = rand(rng, 1:100000000)

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
    explore_flag = false
    init_state = [nothing, nothing, 1, 1, loop, internal_seed, explore, explore_flag, 1]
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
    env(1)
    reset!(env)
    return env
end

function step_loop(env::AbstractRLOptEnv; greedy = false, add_exp = false)
    exp, done = RLE2.interact!(env.env, env.agent, greedy = greedy)

    train_buffer = env.agent.buffers.train_buffer
    meta_buffer = env.agent.buffers.meta_buffer

    buffers = [train_buffer, meta_buffer]
    if add_exp
        for buffer in buffers
        add_exp!(buffer, exp)
        if done
            RLE2.finish_episode(buffer)
        end
        end
    end

    return exp, done
end

function train_loop(env::AbstractRLOptEnv; greedy = false, add_exp = false)
    exp, done = RLE2.interact!(env.env, env.agent, greedy = greedy)

    train_buffer = env.agent.buffers.train_buffer
    meta_buffer = env.agent.buffers.meta_buffer

    buffers = [train_buffer, meta_buffer]
    if add_exp && !env.init_state[8]
        for buffer in buffers
        add_exp!(buffer, exp)
        if done
            RLE2.finish_episode(buffer)
        end
        end
    end

    train_subagents(env.agent, resample = true)

    # ep = [exp]
    # ep = pad_episode!(train_buffer, ep, 2)
    # init_batches(train_buffer, 1)
    # fill_buffer!(train_buffer, ep, 1)
    # train_subagents(env.agent, resample = false)

    return exp, done
end

function sample_context(env)
    # context = 50.0*rand(env.rng) + 5
    # context = Float32((10.0*rand(env.rng) + 5)/15f0)
    # context = Float32((19f0*rand(env.rng) + 1)/20f0)
    context = 10/15
    # context = 0.5f0
end

function reset!(env::AbstractRLOptEnv; saved_f = false, greedy = true, resample = true, explore_flag = false, reset_buffer = true)
    # function reset!(env::AbstractRLOptEnv; saved_f = false, greedy = false)
    env.init_state[8] = explore_flag


    env.init_state[3] = env.init_state[3] + 1
    # internal_seed = 73642149
    internal_seed = env.init_state[6]
    # internal_seed = 1
    # env.env.rng = MersenneTwister(internal_seed)
    # env.env.rng = MersenneTwister(1)
    context = sample_context(env)
    # env.env = CartPoleEnv(T = Float32, gravity = 15f0*context, rng = env.env.rng)
    # env.env = PendulumEnv(T = Float32, g = 15f0*context, rng = env.env.rng, continuous = false)
    # env.env = PendulumEnv(T = Float64)
    env.init_state[9] = context
    reset!(env.env)
    env.done = false

    env.t = 1

    train_buffer = env.agent.buffers.train_buffer
    meta_buffer = env.agent.buffers.meta_buffer
    # init_buffer = env.agent.buffers.init_buffer

    env.agent = get_agent(env.env, env.init_state[3])
    # env.agent.buffers = (train_buffer = train_buffer, meta_buffer = meta_buffer,)
    # env.agent.buffers = (train_buffer = init_buffer, meta_buffer = meta_buffer, init_buffer = init_buffer)
    env.agent.buffers = (train_buffer = env.agent.buffers.train_buffer, meta_buffer = meta_buffer,)

    agent = env.agent

    if saved_f == true && !isnothing(env.init_state[1])
        f = env.init_state[1]
        ps, re = Flux.destructure(f.f)
        if !greedy
            ps_old = Flux.params(f.f)
            ps_new = [p .+ Float32.(randn(Float32, size(p)) / sqrt(prod(size(p)))) for p in ps_old]
            ps_new = vcat([reshape(p, :) for p in ps_new]...)
            ps = ps_new
            # ps = [p .+ 0.01f0*randn(Float32, size(p)) for p in ps]
        end
        f = NeuralNetwork(re(ps))
        env.agent.subagents[1].model.f = f
        env.agent.subagents[1].params = f.params
        if env.agent.name == "DQN"
            f = NeuralNetwork(re(ps))
            env.agent.subagents[1].target.sp_network.f.f = f.f
            env.agent.subagents[1].target.sp_network.f.params = f.params
        end
    end

    if reset_buffer
        buffer = agent.buffers.train_buffer
        old_eps = env.agent.subagents[1].model.ϵ
        env.agent.subagents[1].model.ϵ = 0.9
        populate_replay_buffer!(
            buffer,
            env.env,
            agent,
            # policy = :random,
            policy = :agent,
            num_episodes = 1,
            max_steps = agent.max_agent_steps,
            greedy = false,
        )
        env.agent.subagents[1].model.ϵ = 1.0
    else
    env.agent.buffers = (train_buffer = train_buffer, meta_buffer = meta_buffer,)
        end

    reset!(env.env)
    # env.env.rng = MersenneTwister(env.init_state[3])

    # # println(env.env.t)
    # # println(env.env.state)
    env2 = deepcopy(env.env) #TODO deterministic start state to test pendulum!
    env2.rng = MersenneTwister(internal_seed)
    reset!(env2)
    env.env.state = deepcopy(env2.state)
    # # println(get_obs(env.env))
    # # env.env.state = Float32(0.1) * rand(MersenneTwister(1), Float32, 4) .- Float32(0.05)
    # # env.env.state = Float32(0.1) * rand(MersenneTwister(env.init_state[3]), Float32, 4) .- Float32(0.05)
    # # env.env.state = Float32(0.1) * rand(env.rng, Float32, 4) .- Float32(0.05)
    # # println(env.env.state)
    env.obs = get_next_obs(env, resample = resample)
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
    if contains(env.state_representation, "xon2")
        reset_experience!(env.agent.buffers.meta_buffer)
    end
    # println(env.agent.buffers.meta_buffer)
    done = false
    exp = 1
    G = 0f0
    # s = deepcopy(env.env.state)
    # println("F pre: ", env.agent.subagents[1](s))
    # loss = mc_buffer_loss
    # println("Loss pre: ", loss(env.agent, env.env, batch_size = -1))
    # num_evals = 4
    # for i = 1:num_evals
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
    # end
    # G /= num_evals
    # G = -buffer_loss(env.agent, env.env, batch_size = -1)[1][1]
    # println("F post: ", env.agent.subagents[1](s))
    # println("Loss post: ", loss(env.agent, env.env, batch_size = -1))

    # env.reward = G/1200
    # env.reward = (G + randn())/env.max_steps
    env.reward = G/env.max_steps

    # env.reward += -buffer_loss(env.agent, env.env, batch_size = -1)[1][1]
    # println(l)
    # env.reward = G
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

function get_next_obs(env::AbstractRLOptEnv; resample = true)
    return get_next_obs_with_f(env, env.agent.subagents[1].model.f, resample = resample)
    # return [0]
end

function get_next_obs_with_f(
    env::AbstractRLOptEnv,
    f;
    M = nothing,
    xy = nothing,
    t = nothing,
    resample = true,
    context = nothing,
)
    if isnothing(t)
        t = env.t
    end

    N = RLE2.curr_size(env.agent.buffers.meta_buffer)

    state_rep_str = stop_gradient() do
        split(env.state_representation, "_")
    end

    if isnothing(context)
    aux = env.init_state[9]
    else
        aux = context
    end
    # aux = get_state(env.env)
        # aux = stop_gradient() do
        #     Float32.([t...])
        # end

    aux_dim = length(aux)
    # aux = [1f0]
    # aux_dim = 0

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

        # if string(state_rep_str[1]) !== "PE-xon"
        # while curr_size(env.agent.buffers.meta_buffer) < M
        #     # if state_rep_str[1] == "PE-xon"
        #     #     # exp, done = RLE2.interact!(env2, env.agent, policy = :agent)
        #     # else
        #         exp, done = RLE2.interact!(env2, env.agent, policy = :random)
        #     add_exp!(env.agent.buffers.meta_buffer, exp)
        #     if done
        #         finish_episode(env.agent.buffers.meta_buffer)
        #         reset!(env2)
        #     end
        #     if !done && curr_size(env.agent.buffers.meta_buffer) == M
        #         finish_episode(env.agent.buffers.meta_buffer)
        #     end
        #     # end
        # end
        #         end

    # println("SUM: ", sum(env.agent.buffers.meta_buffer._episode_lengths[2:end]))
    # println("lens: ", length.(env.agent.buffers.meta_buffer._episodes[2:end]))

        if curr_size(env.agent.buffers.meta_buffer) <= M
            replacement = true
        else
            replacement = false
        end

        data = stop_gradient() do
            if resample
                StatsBase.sample(env.agent.buffers.meta_buffer, M, replacement = replacement)
            end
            get_batch(env.agent.buffers.meta_buffer, env.agent.subagents[1])
        end

        x_ = data[2][:,1,:]
        a = data[3][:,1,:]
        x_sp = data[7][:,1,:]
        y = data[5][:,1,:]
        done = data[8][:,1,:]
        mask = data[end-1][:,1,:]
        maskp = data[end][:,1,:]

        y_ = f(x_)
        y_sp = f(x_sp)

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

        elseif state_rep_str[1] == "PE-xon2"
            meta_data_list = [x_]

        elseif state_rep_str[1] == "PE-xontd"
            # y_sp = reshape(
            #     maximum(y_sp, dims = 1),
            #     size(y),
            # )
            y_sp = y_sp .* (1 .- done)
            r = y
            target = r .+ y_sp
            # y_ = sum(y_ .* mask, dims = 1)
            # y_sp = sum(y_sp .* maskp)
            # y_sp = softmax(f(x_sp)).*(1 .- done)
            # y_sp = f(x_sp).*(1 .- done)
            # println(y_sp)
            # println(sum(done))
            # y_sp = f(x_sp)
            meta_data_list = [target, x_, a]

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
        # stop_gradient() do
        #     StatsBase.sample(env.agent.buffers.train_buffer, env.agent.buffers.train_buffer.batch_size);
        # end
        # stop_gradient() do
        #     StatsBase.sample(env.agent.buffers.meta_buffer, env.agent.buffers.meta_buffer.batch_size);
        # end

        return state


    else
        error("Not a valid state Representation for OptEnv!")
    end

end

using Optim, FluxOptTools
function optimize_value_student(
    agent,
    env::AbstractRLOptEnv;
    n_steps = 1,
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

    # if env.init_state[5] == train_loop
    # # if typeof(env.env) <: CartPoleEnv
    #     n_steps = 100
    # end

    RLE2.reset!(env, saved_f = true)
    if !greedy && env.init_state[7] == :hard
    if rand() < 0.1
        # println("NO OPT")
        println("NO OPT HARD")
        RLE2.reset!(env, saved_f = false, explore_flag = true)
        return
    else
        # println("OPT")
    end
    end

    if !greedy && env.init_state[7] == :soft
    if rand() < 0.1
        println("NO OPT SOFT")
        RLE2.reset!(env, saved_f = true, greedy = false)
        return
    else
        # println("OPT")
    end
    end

    if isnothing(env.init_state[1])
        f = env.agent.subagents[1].model.f
        lr = agent.subagents[1].optimizer.eta#/n_steps
        opt = Flux.ADAM(lr)
        # opt = Flux.RMSProp(lr)
        ps, re = Flux.destructure(f.f)
        f = NeuralNetwork(re(ps))
    else
        f = env.init_state[1]
        opt = env.init_state[2]
        ps = f.params
    end

    gs = nothing
    num_reports = 10
    if n_steps > num_reports
        report_freq = Int(n_steps / num_reports)
    else
        report_freq = 1
    end
    grad_ps = nothing


    obs = get_next_obs_with_f(env, f);
    # if Base.mod(env.init_state[4], 20) == 0
    #     println("SUM V PRE OPT: ", -sum(agent.subagents[1](obs |> agent.device)))
    # end

    obs = get_next_obs_with_f(env, f, resample = false);
    # println("SUM V OPT: ", -sum(agent.subagents[1](obs |> agent.device)))
    for i = 1:n_steps
        if mod(i, report_freq) == 0
            debug = true
            if debug
                if Base.mod(env.init_state[4], 20) == 0
                    obs = get_next_obs_with_f(env, f, resample = false);
                    println("SUM V OPT pRogress wh wj wh wj: ", -sum(agent.subagents[1](obs |> agent.device)))
                end
            end
        end

        # if env.state_representation == "parameters"
        #     grad_ps = Flux.params(ps)
        # else
            grad_ps = f.params
        # end

        gs = Flux.gradient(() -> begin
                V = 0
                num_evals = 1
                for i = 1:num_evals
                           context = sample_context(env)
                if env.state_representation == "parameters"
                   s = vcat(ps, get_state(env.env))
                else
                   s = get_next_obs_with_f(env, f, resample = false, context = context);
                end
                V += -sum(agent.subagents[1](s |> agent.device))
                end
                V/num_evals
            end,
            grad_ps,)

        # for k in keys(gs.grads)
        #     if isa(gs.grads[k], Array)
        #         S = size(gs.grads[k])
        #         # gs.grads[k] += randn(Float32, S)/prod(S)/10f0
        #         gs.grads[k] += randn(Float32, S)#/10f0
        #     end
        # end
        Flux.Optimise.update!(opt, grad_ps, gs)
    end

    ps, re = Flux.destructure(f.f)
    # ps_old = Flux.params(f.f)
    # # ps_new = [p .+ Float32.(randn(Float32, size(p))/sqrt(prod(size(p)))) for p in ps_old]
    # ps_new = [p for p in ps_old]
    # lr = agent.subagents[1].optimizer.eta
    # new_opt = typeof(opt)()
    # ps_new = vcat([reshape(p, :) for p in ps_new]...)
    # f = re(ps_new)
    # ps_new = Flux.params(f)
    # for i = 1:length(ps_new)
    #     p_old = ps_old[i]
    #     p_new = ps_new[i]
    #     new_opt.state[p_new] = opt.state[p_old]
    # end

    # env.init_state[1] = NeuralNetwork(f)
    # env.init_state[2] = new_opt

    env.init_state[1] = f
    env.init_state[2] = opt

    # TODO: Figure out how to handle parameters
    # TODO: Figure out how to best set parameters for subagent
    # if env.state_representation == "parameters"
    #     # opt.state = opt.state[grad_ps]
    #     # env .init_state = [f, opt, iter]
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
    if Base.mod(env.init_state[4], 20) == 0
        obs = get_next_obs_with_f(env, f, resample = false);
        println("SUM V POST OPT: ", -sum(agent.subagents[1](obs |> agent.device)))
        println("GRAD NORM: ", sum(LinearAlgebra.norm, gs))
    end
    # reset!(env, saved_f = true, resample = false)
    reset!(env, saved_f = true, resample = false, reset_buffer = false)

        # stop_gradient() do
        #     StatsBase.sample(env.agent.buffers.train_buffer, env.agent.buffers.train_buffer.batch_size);
        # end
        # stop_gradient() do
        #     StatsBase.sample(env.agent.buffers.meta_buffer, env.agent.buffers.meta_buffer.batch_size);
        # end
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
    counts = []
    test_performance = []

    adapted_performance = []
    adapted_counts = []
    adapted_test_performance = []

    env = deepcopy(en)
    agent = deepcopy(agent)

    G = 0
    count = 0
    num_evals = 1
    reset!(env, saved_f = true, greedy = true)
    s = deepcopy(RLE2.get_state(env.env))
    for i = 1:num_evals
    reset!(env, saved_f = true, greedy = true)
    done = false
    while !done
        exp, done = step_loop(env, greedy = true)
        G += exp.r
        count += 1
    end
    end
    G /= num_evals
    count /= num_evals

    push!(performance, G)
    push!(counts, count)

    pre_init_value = env.agent.subagents[1](s |> env.agent.device)

    G = 0
    count = 0
    s_term = nothing
    for i = 1:num_evals
    reset!(env, saved_f = true, greedy = true)
    done = false
    while !done
        s_term = deepcopy(RLE2.get_state(env.env))
        exp, done = train_loop(env, greedy = true)
        G += exp.r
        count += 1
    end
    end
    G /= num_evals
    count /= num_evals

    push!(adapted_performance, G)
    push!(adapted_counts, count)

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



    rs = [
        mean(adapted_performance),
        mean(adapted_counts),
        mean(performance),
        mean(counts),
        mean(LinearAlgebra.norm, init_ps .- post_ps),
        maximum(pre_init_value),
        maximum(post_init_value),
        maximum(pre_term_value),
        maximum(post_term_value),
    ]
    names    = [
        "adapted_performance",
        "adapted_counts",
        "nonadapted_performance",
        "nonadapted_counts",
        "norm_diff_params",
        "pre_init_val",
        "post_init_val",
        "pre_term_val",
        "post_term_val",
    ]
    return rs, names
end

function _step!(env::ReinforcementLearningEnvironments.MountainCarEnv, force)
    env.t += 1
    x, v = env.state
    v += force * env.params.power + cos(3 * x) * (-env.params.gravity)
    v = clamp(v, -env.params.max_speed, env.params.max_speed)
    x += v
    x = clamp(x, env.params.min_pos, env.params.max_pos)
    if x == env.params.min_pos && v < 0
        v = 0
    end
    if env.params.goal_pos > 0
        env.done =
            x >= env.params.goal_pos && v >= env.params.goal_velocity ||
            env.t >= env.params.max_steps
    else
        env.done =
            x <= env.params.goal_pos && v >= env.params.goal_velocity ||
            env.t >= env.params.max_steps
    end
    env.state[1] = x
    env.state[2] = v
    nothing
end

function (env::CartPoleEnv{<:Base.OneTo{Int}})(a::Int)
    @assert a in env.action_space
    env.action = a
    my_step!(env, a == 2 ? 1 : -1)
end

function ReinforcementLearningEnvironments.reward(env::CartPoleEnv{A,T}) where {A,T}
    x, xdot, theta, thetadot = env.state
    # println(x)
    # println(theta)
    xthresh = 0.5
    thetathresh = 0.05
    center = 0f0
    # if abs(x-center) < thresh
    # println("theta: ", theta)
    # println("x: ", x)
    if abs(theta-center) < thetathresh
        return 1
    else
        return 0
    end
    if done
        return 0
    end
end

function my_step!(env::CartPoleEnv, a)
    env.t += 1
    force = a * env.params.forcemag
    x, xdot, theta, thetadot = env.state
    costheta = cos(theta)
    sintheta = sin(theta)
    tmp = (force + env.params.polemasslength * thetadot^2 * sintheta) / env.params.totalmass
    thetaacc =
        (env.params.gravity * sintheta - costheta * tmp) / (
            env.params.halflength *
            (4 / 3 - env.params.masspole * costheta^2 / env.params.totalmass)
        )
    xacc = tmp - env.params.polemasslength * thetaacc * costheta / env.params.totalmass
    env.state[1] += env.params.dt * xdot
    env.state[2] += env.params.dt * xacc
    env.state[3] += env.params.dt * thetadot
    env.state[4] += env.params.dt * thetaacc
    env.done =
        abs(env.state[1]) > env.params.xthreshold ||
        abs(env.state[3]) > env.params.thetathreshold ||
        env.t > env.params.max_steps
    nothing
end
