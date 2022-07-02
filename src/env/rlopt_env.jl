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
            num_episodes = 100,
            max_steps = agent.max_agent_steps,
            greedy = false,
        )
    end

    t = 0
    state = nothing
    obs = nothing
    reward = nothing
    done = false

    init_state = nothing
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

function train_loop(env::AbstractRLOptEnv)
    exp, done = RLE2.interact!(env.env, env.agent, greedy = true)
    buffer = env.agent.buffers.train_buffer
    # add_exp!(buffer, exp)
    ep = [exp]
    ep = pad_episode!(ep, 2)
    fill_buffer!(buffer, ep, 1)
    train_subagents(env.agent, resample = false)
    return exp, done
end

function reset!(env::AbstractRLOptEnv; saved_f = false)
    env.done = false
    reset!(env.env)
    buffer = env.agent.buffers.train_buffer
    agent = get_agent(env.env)
    agent.buffers = (train_buffer =  buffer,)
    if saved_f == :new
        p, re = Flux.destructure(env.init_state)
        f = NeuralNetwork(re(p))
        agent.subagents[1].model.f = f
    end
    env.agent = agent
    env.obs = get_next_obs(env)
end


function (env::AbstractRLOptEnv)(a)
    exp, done = train_loop(env)
    env.reward = exp.r
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

    aux = get_state(env.env)

        # aux = stop_gradient() do
        #     Float32.([t...])
        # end

        aux_dim = length(aux)

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

        data = stop_gradient() do
            StatsBase.sample(env.agent.buffers.train_buffer, M)
            get_batch(env.agent.buffers.train_buffer, env.agent.subagents[1])
        end

        x_ = data[2][:,1,:]
        y = data[5][:,1,:]

        y_ = f(x_)
        y_ = Float32.(RLE2.get_greedypolicy(f(x_)))

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
        state = zeros(Float32, size(state))
        y_dim = 0
        return Float32.(vcat(state, aux, aux_dim, x_dim, y_dim, L, M))

    else
        error("Not a valid state Representation for OptEnv!")
    end
end
