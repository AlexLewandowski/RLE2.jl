function transform_state(env, s_dict, teacher_action)
    state = nothing
    G = sum([e.r for e in s_dict[:ep]])
    if env.state_representation == :return
        G = 0
        ep = s_dict[:ep]
        t = 1
        for exp in ep
            G += exp.r * 0.99f0^t
            t += 1
        end
        state = G

    elseif contains(String(env.state_representation), "trajectory_resevoirQ")
        reinitialize_state(env)
        ep = s_dict[:ep]
        t = 1
        for exp in ep
            experience = exp
            o = experience.o
            Q = experience.info.Q
            state = vcat(
                env.state_encoder(Float32.(vcat(Q, o),)),
                env.state[end] + 0.99f0^t * Float32(experience.r),
            )
            t += 1
        end
        state = env.state

    elseif contains(String(env.state_representation), "trajectory_resevoirP")
        reinitialize_state(env)
        ep = s_dict[:ep]
        t = 1
        for exp in ep
            experience = exp
            o = experience.o
            Q = experience.info.P
            state = vcat(
                env.state_encoder(Float32.(vcat(Q, o),)),
                env.state[end] + 0.99f0^t * Float32(experience.r),
            )
            t += 1
        end
        state = env.state

    elseif contains(String(env.state_representation), "trajectory_resevoir")
        reinitialize_state(env)
        ep = s_dict[:ep]
        t = 1
        for exp in ep
            update_env_state!(env, exp, t)
            t += 1
        end
        state = env.state

    elseif env.state_representation == :parameters
        state = s_dict[:params]

    elseif split(string(env.state_representation), "_")[1] == "PD-x"
        state = s_dict[:buffer]

    elseif env.state_representation == :parameters_policy
        state = s_dict[:params_policy]

    elseif env.state_representation in
           (:ep, :local_policy_table_action, :historic_policy_table_action)
        reinitialize_state(env)
        G = 0
        ep = s_dict[:ep]
        o0 = ep[1].o
        t = 1

        for exp in ep
            state = update_env_state!(env, exp, t)
            t += 1
        end
        # state[2] = s0

    elseif env.state_representation in (:epP, :local_policy_table, :historic_policy_table)
        reinitialize_state(env)
        G = 0
        ep = s_dict[:ep]
        o0 = ep[1].o
        t = 1
        local_sa = []

        for exp in ep
            experience = exp
            o = Float32.(experience.o)
            a = experience.info.P
            r = experience.r
            env.state[3] += 0.99f0^t * r
            push!(env.state[1], vcat(a, o, positional_encoding(env.state[4])))
            env.state[4] += 1
            state = env.state
            t += 1
        end
        #state[2] = s0

    elseif env.state_representation == :epQ
        reinitialize_state(env)
        G = 0
        ep = s_dict[:ep]
        o0 = ep[1].o
        t = 1
        local_sa = []

        for exp in ep
            experience = exp
            o = Float32.(experience.o)
            a = experience.info.Q
            r = experience.r
            env.state[3] += 0.99f0^t * r
            push!(env.state[1], vcat(a, o, positional_encoding(env.state[4])))
            env.state[4] += 1
            state = env.state
            t += 1
        end
        # state[2] = s0

    elseif env.state_representation == :last_state
        state = s_dict[:ep][end].sp
    else
        error(string(state_representation) * " is not a valid state_representation.")
    end

    if env.state_representation in (
        :ep,
        :epP,
        :epQ,
        :local_policy_table,
        :local_policy_table_action,
        :historic_policy_table,
        :historic_policy_table_action,
    )
        env.state = state
        state = post_episode_state(env, teacher_action)
    end
    return state
end

function transform_ep(env::AbstractCurriculumMDP, ep)
    M = length(ep)
    S = typeof(get_state(env))
    O = typeof(get_obs(env))
    A = typeof(random_action(env))

    new_ep = Vector{Experience{S,O,A}}()
    a = default_action(env.student_env)

    if !isnothing(env.teacher_action_encoder)
        Flux.reset!(env.teacher_action_encoder)
        action_mask = Float32.(discrete_action_mask(a, length(env.configurable_params)))
        h = env.teacher_action_encoder(action_mask)
    end

    for exp in ep
        if !isnothing(env.teacher_action_encoder)
            if !contains(string(env.state_representation), "ep")
                s = vcat(transform_state(env, exp.s, a), h)
            end
        else
            s = transform_state(env, exp.s, a)
        end

        s = copy(s)
        o = copy(s)
        a = copy(exp.a)

        if !isnothing(env.teacher_action_encoder)
            action_mask = Float32.(discrete_action_mask(a, length(env.configurable_params)))
            h = env.teacher_action_encoder(action_mask)
        end

        p = copy(exp.p)
        r = copy(exp.r)
        if !isnothing(env.teacher_action_encoder)
            if !contains(string(env.state_representation), "ep")
                sp = vcat(transform_state(env, exp.s, a), h)
            else
                env.state = sp
                sp = post_episode_state(env, a)
            end
        else
            sp = transform_state(env, exp.sp, a)
        end
        sp = copy(sp)
        op = copy(sp)
        done = copy(exp.done)
        new_exp = Experience(s, o, a, p, r, sp, op, done)
        # new_exp = Experience(copy(s), copy(o), copy(a), copy(p), copy(r), copy(sp), copy(op), copy(done))
        push!(new_ep, new_exp)
    end
    return new_ep
end
