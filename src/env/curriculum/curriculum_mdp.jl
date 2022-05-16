abstract type AbstractCurriculumMDP <: AbstractEnv end

include("continuing_cmdp.jl")
include("transformations.jl")

mutable struct EpisodicCurriculumMDP{E} <: AbstractCurriculumMDP
    student_env::E
    student_agent::AbstractAgent
    action_type::Symbol
    state_representation::Any
    state_update::Function
    state_encoder::Any
    student_action_encoder::Any
    teacher_action_encoder::Any
    state::Any
    action::Any
    reward::Any
    done::Any
    student_return_assigned_task::Any
    student_return_target_task::Any
    t::Int
    max_steps::Any
    max_student_steps::Any
    action_space::Any
    memory_size
    memory
    rng::AbstractRNG
end

UnionCurriculumMDP =
    Union{ContinuingCurriculumMDP{E},EpisodicCurriculumMDP{E}} where {E<:AbstractEnv}

include("student_envs.jl")

Base.show(io::IO, t::MIME"text/markdown", env::AbstractCurriculumMDP) = begin
    println(io, "Curriculum MDP")
    println(io, "    Student env: ", typeof(env.student_env).name)
    println(io, "    Student agent: ", env.student_agent.name)
end

Base.show(io::IO, env::AbstractCurriculumMDP) = begin
    println(io, "Curriculum MDP")
    println(io, "    Student env: ", typeof(env.student_env).name)
    println(io, "    Student agent: ", env.student_agent.name)
end

function CurriculumMDP(
    student_env::AbstractEnv,
    state_representation,
    max_steps,
    continuing;
    recurrent_teacher_action_encoder = 0,
    rng = Random.GLOBAL_RNG,
    kwargs...
)

    if :action_type in keys(kwargs)
        action_type = kwargs[:action_type]
    else
        action_type = :start_statE
    end

    if recurrent_teacher_action_encoder == 0
        recurrent_teacher_action_encoder = false
        teacher_encoding_dim = 0
    else
        teacher_encoding_dim = recurrent_teacher_action_encoder
        recurrent_teacher_action_encoder = true
    end

    student_agent = get_agent(student_env)
    buffer = student_agent.buffers.train_buffer
    # buffer.batch_size = 128
    state = nothing

    S = eltype(get_state(student_env))
    O = eltype( get_obs(student_env))
    A = eltype(random_action(student_env))

    action_space = get_configurable_params(student_env)

    init_a = Int(rand(rng, action_space))
    reward = 0.0f0
    done = false

    n_teacher_actions = length(action_space)
    teacher_action_encoder = nothing

    n_states = length(get_obs(student_env))[1]
    n_actions = num_actions(student_env)

    state_rep_str = split(string(state_representation), "_")

    if state_representation == :return
        state = [0.0f0]
        state_update =
            (env, experience, t) ->
                [env.state[end] + 0.99f0^t * Float32(experience.r)]
        state_encoder = nothing
        teacher_action_encoder = x -> x
        student_action_encoder = nothing

    # elseif contains(String(state_representation), "trajectory_resevoirP")
    #     if length(String(state_representation)) == 20
    #         encoding_dim = 8
    #     else
    #         encoding_dim = parse(Int, String(state_representation)[21:end])
    #     end
    #     student_action_encoder = x -> discrete_action_mask(x, n_actions)

    #     state = zeros(Float32, encoding_dim + 1)
    #     state_encoder = rnn(n_actions + 1 * n_states, encoding_dim, seed = 1)
    #     state_update =
    #         (env, experience, t) -> begin
    #             s = experience.s
    #             P = get_greedypolicy_vector(reshape(env.student_agent.π_b(s), :))
    #             return vcat(
    #             env.state_encoder(
    #                 Float32.(
    #                     vcat(
    #                         P,
    #                         s,
    #                     ),
    #                 ),
    #             ),
    #             env.state[end] + 0.99f0^t * Float32(experience.r),
    #         )
    #         end
    # elseif contains(String(state_representation), "trajectory_resevoirQ")
    #     if length(String(state_representation)) == 20
    #         encoding_dim = 8
    #     else
    #         encoding_dim = parse(Int, String(state_representation)[21:end])
    #     end
    #     student_action_encoder = x -> discrete_action_mask(x, n_actions)

    #     state = zeros(Float32, encoding_dim + 1)
    #     state_encoder = rnn(n_actions + 1 * n_states, encoding_dim, seed = 1)
    #     state_update =
    #         (env, experience, t) -> begin
    #             s = experience.s
    #             Q = reshape(env.student_agent.π_b(s), :)
    #             return vcat(
    #             env.state_encoder(
    #                 Float32.(
    #                     vcat(
    #                         Q,
    #                         s,
    #                     ),
    #                 ),
    #             ),
    #             env.state[end] + 0.99f0^t * Float32(experience.r),
    #         )
    #         end

    # elseif contains(String(state_representation), "trajectory_resevoir")
    #     if length(String(state_representation)) == 19
    #         encoding_dim = 8
    #     else
    #         encoding_dim = parse(Int, String(state_representation)[20:end])
    #     end
    #     student_action_encoder = x -> discrete_action_mask(x, n_actions)

    #     state = zeros(Float32, encoding_dim + 1)
    #     state_encoder = rnn(n_actions + 1 * n_states, encoding_dim, seed = 1)
    #     state_update =
    #         (env, experience, t) -> begin
    #             return vcat(
    #             env.state_encoder(
    #                 Float32.(
    #                     vcat(
    #                         env.student_action_encoder(experience.a),
    #                         experience.s,
    #                     ),
    #                 ),
    #             ),
    #             env.state[end] + 0.99f0^t * Float32(experience.r),
    #         )
    #         end

    # elseif state_representation == :trajectory_learning
    #     encoding_dim = 32
    #     student_action_encoder = x -> discrete_action_mask(x, n_actions)
    #     state_encoder = rnn(n_actions + n_states, encoding_dim)
    #     state = []
    #     state_update =
    #         (env, experience, t) -> vcat(
    #             env.state_encoder(
    #                 Float32.(vcat(experience.s, env.student_action_encoder(experience.a))),
    #             ),
    #             env.state[end] + 0.99f0^t * Float32(experience.r),
    #         )
    elseif state_representation == :parameters
        state = get_params_vector(student_agent.π_b)
        state_update = (env, experience, t) -> get_params_vector(env.student_agent.π_b)
        state_encoder = nothing
        student_action_encoder = nothing

    elseif state_representation == :parameters_policy
        state = get_greedypolicy_vector(student_agent.π_b.f)
        state_update = (env, experience, t) -> get_greedypolicy_vector(env.student_agent.π_b.f)
        state_encoder = nothing
        student_action_encoder = nothing

    # elseif state_representation == :last_state
    #     state = vcat(get_state(student_env), 0.0f0)
    #     state_update =
    #         (env, experience, t) ->
    #             vcat(experience.sp, env.state[end] + 0.99f0^t * Float32(experience.r))
    #     state_encoder = nothing
    #     student_action_encoder = nothing
    elseif state_representation == :all
        state = Dict(
            :buffer => nothing,
            :ep => [],
            :params => nothing,
            :params_policy => nothing,
            # :opt => nothing,
            :teacher_action => 1,
            :assigned_return => nothing,
            :target_return => nothing,
        )
        state_update = (env, experience, t) -> begin
            # student = env.student_agent
            # opt = env.student_agent.subagents[1].optimizer
            # opt_params = [getfield(opt, f) for f in fieldnames(typeof(opt))]
            D = env.state
            exp = deepcopy(experience)
            Q = reshape(env.student_agent.π_b(exp.o), :)
            P = get_greedypolicy(Q)
            exp.info = (Q = Q, P = P)
            push_dict!(D, :ep, exp)
            return D
        end
        state_encoder = nothing
        student_action_encoder = nothing
        recurrent_teacher_action_encoder = false

    elseif state_representation in (:historic_policy_table, :local_policy_table)
        n_actions = num_actions(student_env)
        n_states = length(get_valid_nonterminal_states(student_env))
        state = [[], ones(Float32, (n_states, n_actions))/n_actions, 0, 0]
        state_update = (env, experience, t) -> begin
            n_actions = num_actions(env.student_env)
            o = Float32.(experience.o)
            Q = reshape(env.student_agent.π_b(o), :)
            a = Float32.(get_greedypolicy(Q))
            r = experience.r
            env.state[3] += 0.99f0^t*r
            push!(env.state[1], vcat(a,o,positional_encoding(env.state[4])))
            env.state[4] += 1
            return env.state
        end
        state_encoder = nothing
        student_action_encoder = nothing

    elseif state_representation in (:historic_policy_table_action, :local_policy_table_action)
        n_actions = num_actions(student_env)
        n_states = length(get_valid_nonterminal_states(student_env))
        state = [[], ones(Float32, (n_states, n_actions))/n_actions, 0, 0]
        state_update = (env, experience, t) -> begin
            n_actions = num_actions(env.student_env)
            o = Float32.(experience.o)
            a = Float32.(discrete_action_mask(experience.a, n_actions))
            r = experience.r
            env.state[3] += 0.99f0^t*r
            push!(env.state[1], vcat(a,o,positional_encoding(env.state[4])))
            env.state[4] += 1
            return env.state
        end
        state_encoder = nothing
        student_action_encoder = nothing

    elseif state_representation == :ep
        state = [[], get_state(student_env), 0, 0]
        state_update = (env, experience, t) -> begin
            n_actions = num_actions(env.student_env)
            o = Float32.(experience.o)
            a = Float32.(discrete_action_mask(experience.a, n_actions))
            r = experience.r
            env.state[3] += 0.99f0^t*r
            push!(env.state[1], vcat(a,o,positional_encoding(env.state[4])))
            env.state[4] += 1
            return env.state
        end
        state_encoder = nothing
        student_action_encoder = nothing

    elseif state_representation == :epQ
        state = [[], get_state(student_env), 0, 0]
        state_update = (env, experience, t) -> begin
            n_actions = num_actions(env.student_env)
            o = Float32.(experience.o)
            a = env.student_agent.π_b(o)
            r = experience.r
            env.state[3] += 0.99f0^t*r
            push!(env.state[1], vcat(a,o,positional_encoding(env.state[4])))
            env.state[4] += 1
            return env.state
        end
        state_encoder = nothing
        student_action_encoder = nothing

    elseif split(string(state_representation), "_")[1] == "PD-x"

        println(string(state_representation))
        if length(string(state_representation)) == 8
            # batch_size = 128
        else
            batch_size = parse(Int, split(string(state_representation), "_")[2])
            student_agent.buffers.train_buffer.batch_size = batch_size
        end

        state = [[], get_state(student_env), 0, 0]
        state_update = (env, experience, t) -> begin
            n_actions = num_actions(env.student_env)
            o = Float32.(experience.o)
            Q = reshape(env.student_agent.π_b(o), :)
            a = Float32.(get_greedypolicy(Q))
            # a = Q
            r = experience.r
            env.state[3] += 0.99f0^t*r
            push!(env.state[1], vcat(a,o,env.state[4]))
            env.state[4] += 1
            return env.state
        end
        state_encoder = nothing
        student_action_encoder = nothing

    elseif state_representation == :epP
        state = [[], get_state(student_env), 0, 0]
        state_update = (env, experience, t) -> begin
            n_actions = num_actions(env.student_env)
            o = Float32.(experience.o)
            Q = reshape(env.student_agent.π_b(o), :)
            a = Float32.(get_greedypolicy(Q))
            r = experience.r
            env.state[3] += 0.99f0^t*r
            push!(env.state[1], vcat(a,o,positional_encoding(env.state[4])))
            env.state[4] += 1
            return env.state
        end
        state_encoder = nothing
        student_action_encoder = nothing
    else
        error(string(state_representation) * " is not a valid state_representation.")
    end

    if recurrent_teacher_action_encoder
        teacher_action_encoder = rnn(n_teacher_actions, teacher_encoding_dim, seed = 2)
    end

    populate_replay_buffer!(
        buffer,
        student_env,
        student_agent,
        policy = :random,
        num_episodes = 1,
        max_steps = student_agent.max_agent_steps,
    )

    max_student_steps = student_agent.max_agent_steps
    memory_size = 0
    memory = zeros(Float32, memory_size)
    if continuing
        env = ContinuingCurriculumMDP(
            student_env,
            student_agent,
            action_type,
            state_representation,
            state_update,
            state_encoder,
            student_action_encoder,
            teacher_action_encoder,
            state,
            init_a,
            reward,
            done,
            0.0f,
            0,
            0,
            max_student_steps,
            max_steps,
            action_space,
            rng,
        )
    else
        env = EpisodicCurriculumMDP(
            student_env,
            student_agent,
            action_type,
            state_representation,
            state_update,
            state_encoder,
            student_action_encoder,
            teacher_action_encoder,
            state,
            init_a,
            reward,
            done,
            0.0f0,
            0,
            0,
            max_student_steps,
            max_steps,
            action_space,
            memory_size,
            memory,
            rng,
        )
    end

    reset!(env)
    return env
end

function initialize_state(env::AbstractCurriculumMDP)
    Flux.reset!(env.teacher_action_encoder)
    Flux.reset!(env.state_encoder)
    a, p = default_action(env)
    env(a, learning = false)
    # state_representation = env.state_representation
    # if state_representation == :return
    #     a = default_action(env.student_env)
    #     rollout_student(env, a, learning = false)
    #     post_episode_state(env, a)
    # elseif contains(String(state_representation), "trajectory_resevoir")
    #     a = default_action(env.student_env)
    #     rollout_student(env, a, learning = false)
    #     post_episode_state(env, a)
    # elseif state_representation == :parameters
    #       post_episode_state(env, a)
    # elseif state_representation == :last_state
    #     a = default_action(env.student_env)
    #     rollout_student(env, a, learning = false)
    #     post_episode_state(env, a)
    # elseif state_representation == :all
    #     a = default_action(env.student_env)
    #     rollout_student(env, a, learning = false)
    #     post_episode_state(env, a)
    # end
end

function post_episode_state(env::AbstractCurriculumMDP, action)
    env.action = action
    action_mask = Float32.(discrete_action_mask(action, length(env.action_space)))
    state_representation = env.state_representation
    if state_representation == :return
    elseif contains(String(state_representation), "trajectory_resevoir")
    elseif state_representation == :parameters
        env.state = get_params_vector(env.student_agent.π_b)
    elseif state_representation == :parameters_policy
        env.state = get_greedypolicy_vector(env.student_agent.π_b.f)
    elseif state_representation == :last_state
    elseif state_representation == :all
        student = env.student_agent
        D = env.state
        opt = student.subagents[1].optimizer
        opt_params = [getfield(opt, f) for f in fieldnames(typeof(opt))]
        D[:params] = copy(get_params_vector(student.subagents[1]))
        D[:params_policy] = copy(get_greedypolicy_vector(student.subagents[1].model.f))
        # D[:opt] = copy(opt_params)
        D[:teacher_action] = copy(env.action)
        D[:target_return] = copy(env.student_return_target_task)
        D[:assigned_return] = copy(env.student_return_assigned_task)

        a_dim = prod(size(get_actions(env.student_env)))
        s_dim = prod(size(get_obs(env.student_env)))
        t_dim = 0

        buffer = env.student_agent.buffers.train_buffer
        t = 0
        StatsBase.sample(buffer)
        T = buffer.batch_size
        o_batch = buffer._o_batch
        T = s_dim
        o_batch = []
        for i = 1:T
            o = zeros(Float32, s_dim)
            o[i] = 1
            push!(o_batch, o)
        end
        o_batch = hcat(o_batch...)
        Q = env.student_agent.π_b(o_batch)
        P = Float32.(get_greedypolicy(Q))
        # o_batch = o_batch[1:end-1,:]
        buffer_state = reshape(cat(P, o_batch, dims = 1), :)

        Z = env.memory
        buffer_state = vcat(Z,Z,buffer_state, action_mask, Float32.(env.student_return_assigned_task), T, s_dim, a_dim, t, T)
        D[:buffer] = copy(buffer_state)


    elseif split(string(state_representation), "_")[1] == "PD-x"
        step = 0
        a_dim = prod(size(get_actions(env.student_env)))
        s_dim = prod(size(get_obs(env.student_env)))
        t_dim = 0

        buffer = env.student_agent.buffers.train_buffer
        #T = 1000 #s_dim - 1 #tabular correction
        t = 1
        StatsBase.sample(buffer)
        T = buffer.batch_size
        o_batch = buffer._o_batch
        T = s_dim
        o_batch = []
        for i = 1:T
            o = zeros(Float32, s_dim)
            o[i] = 1
            push!(o_batch, o)
        end
        o_batch = hcat(o_batch...)
        Q = env.student_agent.π_b(o_batch)
        P = Float32.(get_greedypolicy(Q))
        # P = Q
        #o_batch = o_batch[1:end-1,:] # tabular correction
        state = reshape(cat(P, o_batch, dims = 1), :)
        aux_dim = 0

        x_dim = s_dim + a_dim
        y_dim = 0
        L = 1
        aux = 0
        aux_dim = 0

        env.state = vcat(state, aux, aux_dim, x_dim, y_dim, L, T) |> Flux.cpu


    elseif state_representation == :historic_policy_table
        step = 0
        D = size(env.state[1][1])[1]
        a_dim = prod(size(get_actions(env.student_env)))
        s_dim = length(get_valid_nonterminal_states(env.student_env))
        P = env.state[2]
        counts = zeros(Int, s_dim)
        for s in env.state[1]
            a = s[1:a_dim]
            s = s[a_dim+1:s_dim + a_dim]
            s = argmax(s)[1]
            P[s, :] = a
        end
        env.state = reshape(P, :)

    elseif state_representation == :historic_policy_table_action
        step = 0
        D = size(env.state[1][1])[1]
        a_dim = prod(size(get_actions(env.student_env)))
        s_dim = length(get_valid_nonterminal_states(env.student_env))
        P = env.state[2]
        counts = zeros(Int, s_dim)
        for s in env.state[1]
            a = s[1:a_dim]
            s = s[a_dim+1:s_dim + a_dim]
            s = argmax(s)[1]
            # count = counts[s]
            P[s, :] = a
            # P[s, :] = Float32.((count*P[s,:] + a)/(count + 1))
            # counts[s] += 1
        end
        env.state = reshape(P, :)

    elseif state_representation == :local_policy_table
        step = 0
        D = size(env.state[1][1])[1]
        a_dim = prod(size(get_actions(env.student_env)))
        s_dim = length(get_valid_nonterminal_states(env.student_env))
        P = ones(Float32, (s_dim, a_dim))/a_dim
        for s in env.state[1]
            a = s[1:a_dim]
            s = s[a_dim+1:s_dim + a_dim]
            s = argmax(s)[1]
            P[s, :] = a
        end
        env.state = reshape(P, :)
    elseif state_representation == :local_policy_table_action
        step = 0
        D = size(env.state[1][1])[1]
        a_dim = prod(size(get_actions(env.student_env)))
        s_dim = length(get_valid_nonterminal_states(env.student_env))
        P = ones(Float32, (s_dim, a_dim))/a_dim
        counts = zeros(Int, s_dim)
        for s in env.state[1]
            a = s[1:a_dim]
            s = s[a_dim+1:s_dim + a_dim]
            s = argmax(s)[1]
            count = counts[s]
            P[s, :] = a
            # P[s, :] = Float32.((count*P[s,:] + a)/(count + 1))
            counts[s] += 1
        end
        env.state = reshape(P, :)
    elseif state_representation == :ep
        step = 0
        D = size(env.state[1][1])[1]
        a_dim = prod(size(get_actions(env.student_env)))
        s_dim = length(get_valid_nonterminal_states(env.student_env))
        t_dim = D - a_dim - s_dim
        T = env.student_agent.max_agent_steps
        while length(env.state[1]) < T
            if step == 0
                step += 1
            end
            push!(env.state[1], zeros(D))
            # push!(env.state[1], zeros(D))
        end
        flat_s = Float32.(reshape(hcat(env.state[1]...), :))
        # env.state = flat_s
        env.state = vcat(flat_s, action_mask, env.state[3], env.state[4], s_dim, a_dim, 1,  T)#, env.state[3], env.state[4]]

    elseif state_representation == :epP
        step = 0
        D = size(env.state[1][1])[1]
        a_dim = prod(size(get_actions(env.student_env)))
        s_dim = prod(size(get_obs(env.student_env)))
        t_dim = D - a_dim - s_dim
        T = env.student_agent.max_agent_steps
        while length(env.state[1]) < T
            if step == 0
                step += 1
            end
            push!(env.state[1], zeros(D))
            # push!(env.state[1], zeros(D))
        end
        flat_s = Float32.(reshape(hcat(env.state[1]...), :))
        env.state = vcat(flat_s, action_mask, env.state[3], env.state[4], s_dim, a_dim, 1, T)#, env.state[3], env.state[4]]
    elseif state_representation == :epQ
        step = 0
        D = size(env.state[1][1])[1]
        a_dim = prod(size(get_actions(env.student_env)))
        s_dim = prod(size(get_obs(env.student_env)))
        t_dim = D - a_dim - s_dim
        T = env.student_agent.max_agent_steps
        while length(env.state[1]) < T
            if step == 0
                step += 1
            end
            push!(env.state[1], zeros(D))
            # push!(env.state[1], zeros(D))
        end
        flat_s = Float32.(reshape(hcat(env.state[1]...), :))
        # env.state = flat_s
        env.state = vcat(flat_s, action_mask, env.state[3], env.state[4], s_dim, a_dim, 1, T)#, env.state[3], env.state[4]]
    end

    if string(env.state_representation)[1:2] == "ep"
        Z = env.memory
        env.state = cat(Z, Z, env.state, dims = 1)
    end

    if !isnothing(env.teacher_action_encoder)
        action_mask = Float32.(discrete_action_mask(action, length(env.action_space)))
        h = env.teacher_action_encoder(action_mask)
        if env.state_representation == :ep
            env.state = vcat(env.state, [h])
        elseif env.state_representation == :epP
            env.state = vcat(env.state, [h])
        elseif env.state_representation == :epQ
            env.state = vcat(env.state, [h])
        else
            env.state = vcat(env.state, h)
        end
    end
    return env.state
end

function reinitialize_state(env::AbstractCurriculumMDP)
    buffer = env.student_agent.buffers.train_buffer
    state_representation = env.state_representation
    if state_representation == :return
        env.state = [0.0f0]
    elseif contains(String(state_representation), "trajectory_resevoir")
        encoding_dim = size(env.state_encoder.layers[1].cell.Wi)[1]
        env.state = zeros(Float32, encoding_dim + 1)
        Flux.reset!(env.state_encoder)
    elseif state_representation == :parameters
        env.state = get_params_vector(env.student_agent.π_b)
    elseif state_representation == :parameters_policy
        env.state = get_greedypolicy_vector(env.student_agent.π_b.f)
    elseif state_representation == :last_state
        env.state = vcat(get_state(env.student_env), 0.0f0)
    elseif state_representation == :all
        env.state = Dict(
            :ep => [],
            :params => nothing,
            :params_policy => nothing,
            # :opt => nothing,
            :teacher_action => 1,
            :target_return => nothing,
        )
    elseif state_representation in (:historic_policy_table, :historic_policy_table_action)
        n_actions = num_actions(env.student_env)
        n_states = length(get_valid_nonterminal_states(env.student_env))
        if length(env.state) == 4
            env.state = [[], ones(Float32, (n_states, n_actions))/n_actions, 0, 0]
        else
            env.state = [[], reshape(env.state, (n_states, n_actions)) , 0, 0]
        end

    elseif split(string(state_representation), "_")[1] == "PD-x"
        if length(string(state_representation)) == 8
            # batch_size = 128
        else
            batch_size = parse(Int, split(string(state_representation), "_")[2])
            env.student_agent.buffers.train_buffer.batch_size = batch_size
        end
        env.state = [[], get_state(env.student_env), 0, 0]

    elseif state_representation in (:ep, :epP, :epQ, :local_policy_table, :local_policy_table_action, :historic_policy_table, :historic_policy_table_action)
        env.state = [[], get_state(env.student_env), 0, 0]
    end

    populate_replay_buffer!(
        buffer,
        env.student_env,
        env.student_agent,
        policy = :random,
        num_episodes = 1,
        max_steps = env.student_agent.max_agent_steps,
    )
end

function reset!(env::AbstractCurriculumMDP)
    reset!(env.student_env)
    env.student_agent = get_agent(env.student_env)
    buffer = env.student_agent.buffers.train_buffer
    # buffer.batch_size = 128
    env.reward = 0.0f0
    env.student_return_target_task = 0.0f0
    env.student_return_assigned_task = 0.0f0
    env.done = false
    env.action = Int(rand(env.rng, env.action_space))
    env.t = 0
    initialize_state(env)
    return nothing
end

function (env::AbstractCurriculumMDP)(teacher_action; learning = true, num_evals = 100)
    @assert teacher_action in env.action_space
    env.t += 1
    Gs = nothing
    steps = nothing
    # if action_type == :tandem
    #     Gs, steps = rollout_student_tandem(env, teacher_action, learning = learning)
    # else
        Gs, steps = rollout_student(env, teacher_action, learning = learning)
    # end

    env.student_return_assigned_task = Float32(Gs)
    if is_tabular(env.student_env)
        num_evals = 1
    end

    # println(env.student_return_assigned_task)
    if env.action_type !== :tandem
        action, _ = default_action(env)
        teacher_action!(env, action)
    end

    G, _ = rollout_returns_nonmean(
        env.student_agent,
        env.student_env,
        num_evals = num_evals,
        greedy = true,
    )

    env.student_return_target_task = Float32(mean(G[1]))
    bool, score, thresh = calc_threshold(env.student_env, G)

    post_episode_state(env, teacher_action)

    if bool
        env.reward = 0.0f0
        env.done = true
    else
        env.reward = -1.0f0
        env.done = false
    end

    env.state, env.done
end

function update_env_state!(env::AbstractCurriculumMDP, experience, t)
    env.state = env.state_update(env, experience, t)
end

function threshold(env::AbstractCurriculumMDP)
    threshold(env.student_env)
end

function rollout_student(
    env::UnionCurriculumMDP{E},
    action = nothing;
    learning = true,
    num_evals = 1,
    greedy = false,
) where {E<:Union{MountainCarEnv,CartPoleEnv}}
    if action === nothing
        action = default_action(env)
    end
    reinitialize_state(env)

    buffer = env.student_agent.buffers.train_buffer
    # reset_experience!(buffer)

    teacher_action!(env, action)
    while curr_size(buffer; online = true) < buffer.batch_size
        populate_replay_buffer!(
            buffer,
            env.student_env,
            env.student_agent,
            policy = :agent,
            num_episodes = 1,
            max_steps = env.student_agent.max_agent_steps,
        )
    end
    mean_G = 0.0f0
    mean_step = 0
    for i = 1:num_evals
        teacher_action!(env, action)
        student_env = env.student_env
        student_agent = env.student_agent
        done = get_terminal(student_env)
        step = 1
        G = 0.0f0
        gamma = 1.0

        while !done
            experience, done = interact!(student_env, student_agent, greedy)
            if learning
                add_exp!(buffer, experience)
                train_offline_subagents(env.student_agent, 1, reg = false)
            end
            update_env_state!(env, experience, step)

            G += gamma * experience.r
            gamma *= student_agent.subagents[1].gamma
            if step == student_agent.max_agent_steps || done
                finish_episode(buffer)
                done = true
                break
            end
            step += 1
        end
        mean_G += G
        mean_step += step
    end

    return mean_G / num_evals, mean_step / num_evals
end

function rollout_student_tandem(
    env::UnionCurriculumMDP{E},
    action = nothing;
    learning = true,
    num_evals = 1,
    greedy = false,
) where {E}
    if action === nothing
        action = default_action(env.student_env)
    end

    mean_G = 0.0f0
    mean_step = 0
    reinitialize_state(env)

    for i = 1:num_evals
        student_env = env.student_env
        student_agent = env.student_agent
        done = get_terminal(student_env)
        step = 1
        G = 0.0f0
        gamma = 1.0
        while !done
            teacher_action!(env, action)
            experience, done = interact!(student_env, student_agent, greedy)
            if learning
                add_exp!(student_agent.buffers.train_buffer, experience)
                if is_tabular(env.student_env)
                    tabular_learning(student_agent.subagents[1], experience)
                else
                    train_offline_subagents(env.student_agent, 1, reg = false)
                end
            end
            update_env_state!(env, experience, step)
            G += gamma * experience.r
            gamma *= student_agent.subagents[1].gamma
            if step == student_agent.max_agent_steps || done
                if learning
                    finish_episode(student_agent.buffers.train_buffer)
                    done = true
                end
                break
            end
            step += 1
        end
        mean_G += G
        mean_step += step
    end
    return mean_G / num_evals, mean_step / num_evals
end

function rollout_student(
    env::UnionCurriculumMDP{E},
    action = nothing;
    learning = true,
    num_evals = 1,
    greedy = false,
) where {E}
    if action === nothing
        action, p = default_action(env.student_env)
    end

    mean_G = 0.0f0
    mean_step = 0
    reinitialize_state(env)

    for i = 1:num_evals
        teacher_action!(env, action)
        student_env = env.student_env
        student_agent = env.student_agent
        done = get_terminal(student_env)
        step = 1
        G = 0.0f0
        gamma = 1.0
        while !done
            experience, done = interact!(student_env, student_agent, greedy)
            if learning
                add_exp!(student_agent.buffers.train_buffer, experience)
                if is_tabular(env.student_env)
                    tabular_learning(student_agent.subagents[1], experience)
                else
                    train_offline_subagents(env.student_agent, 1, reg = false)
                end
            end
            update_env_state!(env, experience, step)
            G += gamma * experience.r
            gamma *= student_agent.subagents[1].gamma
            if step == student_agent.max_agent_steps || done
                if learning
                    finish_episode(student_agent.buffers.train_buffer)
                    done = true
                end
                break
            end
            step += 1
        end
        mean_G += G
        mean_step += step
    end
    return mean_G / num_evals, mean_step / num_evals
end

function Random.seed!(env::AbstractCurriculumMDP, seed) end

function get_info(env::AbstractCurriculumMDP)
    return nothing
end

function get_state(env::AbstractCurriculumMDP)
    return env.state
end

function get_obs(env::AbstractCurriculumMDP)
    return env.state
end

function get_terminal(env::AbstractCurriculumMDP)
    return env.done
end

function get_reward(env::AbstractCurriculumMDP)
    return env.reward
end

function get_actions(env::AbstractCurriculumMDP)
    return get_configurable_params(env.student_env)
end

function random_action(env::AbstractCurriculumMDP)
    return rand(env.rng, get_actions(env))
end
