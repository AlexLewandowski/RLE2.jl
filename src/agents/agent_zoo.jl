function get_agent(
    AgentType::String,
    buffers::NamedTuple,
    env,
    measurement_freq::Int64,
    max_agent_steps::Int64,
    measurement_funcs::AbstractArray,
    gamma::Float32,
    update_freq::Int64,
    update_cache::Int64,
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
    reg = false;
    behavior = nothing,
    kwargs...,
)
    rng = MersenneTwister(seed)

    input_dim = length(get_obs(env))
    output_dim = num_actions(env)

    tabular = is_tabular(env)
    discrete = typeof(random_action(env)) == Int

    obs = get_obs(env)
    other_params = [nothing]

    state_encoder = NeuralNetwork(Chain(identity))
    encode_dim = Int(floor(hidden_size / 4))
    if typeof(env) <: Union{UnionCurriculumMDP,AbstractOptEnv,MetaEnv}
        if contains(string(env.state_representation), "ep") || contains(string(env.state_representation), "PD") || contains(string(env.state_representation), "PEN")

            if !contains(env.state_representation, "PEN")
            aux_dim, s_dim, a_dim, L, T = Int.(obs[end-4:end])
            else
                aux_dim = 1
            end


            # encode_dim = Int(hidden_size / 2)
            encode_dim = hidden_size
            encode_dim_s = encode_dim
            encode_dim_a = encode_dim_s

            in_dim = encode_dim
            out_dim = hidden_size
            joint_encode_dim = encode_dim_s + encode_dim_a

            if contains(string(env.state_representation), "buffer") || contains(string(env.state_representation), "PD")
                As = meta_state_encoder(repeat([s_dim],L), repeat([a_dim],L), encode_dim, seed, aux_dim = aux_dim, σ = activation, pooling_f = kwargs[:pooling_func])
            elseif contains(string(env.state_representation), "PEN")

                state_rep_str = stop_gradient() do
                    split(env.state_representation, "_")
                end

                if length(state_rep_str) == 1
                    M = stop_gradient() do
                        minimum([256, env.data_size])
                    end
                else
                    M = stop_gradient() do
                        parse(Int, state_rep_str[2])
                    end
                end

                _, re = Flux.destructure(env.f.f)
                input_dim = size(env.x)[1]
                state_encoder = feed_forward(input_dim*M, hidden_size, hidden_size, num_hidden_layers = 1)
                pen_net = PEN(state_encoder, re, input_dim, num_inputs = M)
                As = pen_net
                # As = meta_state_encoder(repeat([s_dim],L), repeat([a_dim],L), encode_dim, seed, aux_dim = aux_dim, σ = activation, pooling_f = kwargs[:pooling_func])
            else
                out_dim = encode_dim_s + encode_dim_a
                A_s = feed_forward(
                    s_dim,
                    encode_dim_s,
                    encode_dim_s,
                    seed = seed,
                    output_a = Flux.relu,
                )
                A_a = feed_forward(
                    a_dim,
                    encode_dim_a,
                    encode_dim_a,
                    seed = seed,
                    output_a = Flux.relu,
                )
                in_dim = out_dim
                #out_dim = out_dim*2
                A_space = feed_forward(
                    in_dim,
                    out_dim,
                    out_dim,
                    seed = seed,
                    output_a = Flux.relu,
                    output_bias = false,
                )
                in_dim = out_dim
                out_dim = 64
                A_time = rnn(in_dim, out_dim, seed = seed, recurrence = GRU)
                As = Chain([A_time, A_space, A_a, A_s]...) |> device
            end

            state_encoder = As

            input_dim = out_dim + aux_dim
            if env.state_representation == "PEN"
                other_params = [state_encoder]
            else
                other_params = [state_encoder.f]
            end

        end
    end

    action_encoder(x) = discrete ? discrete_action_mask(x, output_dim) : x

    model_params = ModelParams(
        input_dim,
        output_dim,
        discrete,
        predict_window,
        history_window,
        num_layers,
        hidden_size,
        activation,
        device,
        drop_rate,
        seed,
        tabular,
        NamedTuple(),
    )


    if reg == true
        reward_model = get_model("Value", model_params)
        submodels = (reward_model = reward_model,)
    else
        submodels = NamedTuple()
    end

    submodels =
        merge(submodels, (action_encoder = action_encoder, state_encoder = state_encoder))



    if AgentType == "BehaviorCloning"
        model = get_model("Policy", model_params)
        bc_target = (s_network = nothing, sp_network = nothing, func = act_target)
        subagent = Subagent(
            model,
            submodels,
            gamma,
            update_freq,
            update_cache,
            num_grad_steps,
            Flux.Losses.logitcrossentropy,
            optimizer(lr),
            "action_value",
            target = bc_target,
            device = device,
            other_params = other_params,
        )

        subagents = Vector{Any}()
        push!(subagents, subagent)
        π_b = model

    elseif AgentType == "DQN"
        model = get_model("ActionValue", model_params)

        submodels = merge(
            submodels,
            (action_value = model, state_encoder = state_encoder),
        )
        target_submodels = deepcopy(submodels)
        dqn_target =
            (s_network = nothing, sp_network = target_submodels.action_value, func = max_target)
        subagent = Subagent(
            model,
            submodels,
            gamma,
            update_freq,
            update_cache,
            num_grad_steps,
            squared,
            optimizer(lr),
            "action_value",
            target = dqn_target,
            device = device,
            other_params = other_params,
            target_submodels = target_submodels,
        )

        subagents = Vector{Any}()

        push!(subagents, subagent)

        π_b = model

    elseif AgentType == "DoubleDQN"
        model = get_model("ActionValue", model_params)
        submodels = merge(
            submodels,
            (action_value = model, state_encoder = state_encoder),
        )
        target_submodels = deepcopy(submodels)
        dqn_target =
            (s_network = nothing, sp_network = target_submodels.action_value, func = doubledqn_target)
        subagent = Subagent(
            model,
            submodels,
            gamma,
            update_freq,
            update_cache,
            num_grad_steps,
            squared,
            optimizer(lr),
            "action_value",
            target = dqn_target,
            device = device,
            other_params = other_params,
            target_submodels = target_submodels,
        )

        subagents = Vector{Any}()
        push!(subagents, subagent)
        π_b = model

    elseif AgentType == "TD3"
        q_model1 = get_model("ContinuousActionValue", model_params)
        q_model2 = get_model("ContinuousActionValue", model_params)
        policy_model = get_model("Policy", model_params)

        submodels = merge(
            submodels,
            (
                action_value1 = q_model1,
                action_value2 = q_model2,
                state_encoder = state_encoder,
                policy = policy_model,
            ),
        )
        target_submodels = deepcopy(submodels)

        q_target = NamedTuple()
                q_target = (
            s_network = nothing,
            sp_network = [target_submodels.action_value1, target_submodels.action_value2],
            func = td3_target,
        )

        policy_target = (s_network = nothing, sp_network = nothing, func = s_target)
        subagent_q1 = Subagent(
            q_model1,
            submodels,
            gamma,
            update_freq,
            update_cache,
            num_grad_steps,
            squared,
            optimizer(lr),
            "action_value1",
            target = q_target,
            device = device,
            other_params = other_params,
            target_submodels = target_submodels,
        )
        subagent_q2 = Subagent(
            q_model2,
            submodels,
            gamma,
            update_freq,
            update_cache,
            num_grad_steps,
            squared,
            optimizer(lr),
            "action_value2",
            target = q_target,
            device = device,
            other_params = other_params,
            target_submodels = target_submodels,
        )

        policy_loss =
            (estimate, target; agg = identity, state_encoder = nothing) -> begin

                Q1 = submodels.action_value1
                Q2 = submodels.action_value2

                if !isnothing(state_encoder)
                    target = state_encoder(target)
                end

                Q1_ = Q1(target, estimate)
                Q2_ = Q2(target, estimate)

                Q = Q1_

                -agg(Q)
            end

        subagent_policy = Subagent(
            policy_model,
            submodels,
            gamma,
            update_freq,
            update_cache,
            num_grad_steps,
            policy_loss,
            optimizer(lr),
            "policy",
            target = policy_target,
            device = device,
            other_params = other_params,
            train_freq = 2,
            target_submodels = target_submodels,
        )

        subagents = [subagent_q1, subagent_q2, subagent_policy]
        π_b = policy_model

    elseif contains(AgentType, "QRDQN")
        if length(AgentType) == 5
            num_heads = 50
        else
            num_heads = parse(Int, AgentType[6:end])
        end

        QR_model_params = model_params([[:model_local, (num_heads = num_heads,)]])
        model = get_model("QRActionValue", QR_model_params)
        submodels = merge(
            submodels,
            (action_value = model, state_encoder = state_encoder),
        )
        target_submodels = deepcopy(submodels)
        qrdqn_target =
            (s_network = nothing, sp_network = target_submodels.action_value, func = qr_target)
        subagent = Subagent(
            model,
            submodels,
            gamma,
            update_freq,
            update_cache,
            num_grad_steps,
            quantile_huber_loss,
            optimizer(lr),
            "action_value",
            target = qrdqn_target,
            device = device,
            other_params = other_params,
            target_submodels = target_submodels
        )
        subagents = [subagent]
        π_b = model

    elseif AgentType == "TD0"
        model = get_model("Value", model_params)

        submodels = merge(
            submodels,
            (value = model, state_encoder = state_encoder),
        )
        target_submodels = deepcopy(submodels)

        dqn_target = (s_network = nothing, sp_network = submodels.value, func = v_target)
        subagent = Subagent(
            model,
            submodels,
            gamma,
            update_freq,
            update_cache,
            num_grad_steps,
            squared,
            optimizer(lr),
            "value",
            target = dqn_target,
            device = device,
            other_params = other_params,
            target_submodels = target_submodels,
        )

        subagents = [subagent]

        if isnothing(behavior)
            behavior = :deterministic
        end

        if reg == true
            r_target = (s_network = nothing, sp_network = nothing, func = reward_target)
            r_subagent = Subagent(
                reward_model,
                submodels,
                gamma,
                update_freq,
                update_cache,
                num_grad_steps,
                squared,
                optimizer(lr),
                "reward",
                target = r_target,
                device = device,
                other_params = [other_params],
            )
            subagents = push!(subagents, r_subagent)
        end

    elseif AgentType == "MCValue"
        model = get_model("Value", model_params)
        submodels = merge(
            submodels,
            (value = deepcopy(model), state_encoder = deepcopy(state_encoder)),
        )
        dqn_target = (s_network = nothing, sp_network = nothing, func = v_target)
        subagent = Subagent(
            model,
            submodels,
            gamma,
            update_freq,
            update_cache,
            num_grad_steps,
            squared,
            optimizer(lr),
            "value",
            target = dqn_target,
            device = device,
            other_params = other_params,
        )
        subagents = [subagent]

        if isnothing(behavior)
            behavior = :deterministic
        end

        if reg == true
            r_target = (s_network = nothing, sp_network = nothing, func = reward_target)
            r_subagent = Subagent(
                reward_model,
                submodels,
                gamma,
                update_freq,
                update_cache,
                num_grad_steps,
                squared,
                optimizer(lr),
                "reward",
                target = r_target,
                device = device,
                other_params = [other_params],
            )
            subagents = push!(subagents, r_subagent)
        end

        buffer_keys = keys(buffers)
        for k in buffer_keys
            buffers[k].bootstrap = false
        end


    elseif AgentType == "ResidualTD0"
        model = get_model("Value", model_params)
        submodels = merge(
            submodels,
            (value = model, state_encoder = state_encoder),
        )
        target_submodels = deepcopy(submodels)
        dqn_target = (s_network = nothing, sp_network = model, func = v_target)
        subagent = Subagent(
            model,
            submodels,
            gamma,
            update_freq,
            update_cache,
            num_grad_steps,
            squared,
            optimizer(lr),
            "value",
            target = dqn_target,
            device = device,
            other_params = other_params,
            target_submodels = target_submodels,
        )

        subagents = [subagent]

        if isnothing(behavior)
            behavior = :deterministic
        end

        if reg == true
            r_target = (s_network = nothing, sp_network = nothing, func = reward_target)
            r_subagent = Subagent(
                reward_model,
                submodels,
                gamma,
                update_freq,
                update_cache,
                num_grad_steps,
                squared,
                optimizer(lr),
                "reward",
                target = r_target,
                device = device,
                other_params = [other_params],
            )
            subagents = push!(subagents, r_subagent)
        end

    elseif AgentType == "SARSA"
        model = get_model("ActionValue", model_params)
        submodels = (value = deepcopy(model), state_encoder = deepcopy(state_encoder))
        dqn_target = (s_network = nothing, sp_network = model, func = v_target)
        subagent = Subagent(
            model,
            submodels,
            gamma,
            update_freq,
            update_cache,
            num_grad_steps,
            squared,
            optimizer(lr),
            "value",
            target = dqn_target,
            device = device,
            other_params = other_params,
        )
        subagents = [subagent]

        if isnothing(behavior)
            behavior = :deterministic
        end

    elseif AgentType == "ResidualSARSA"
        model = get_model("ActionValue", model_params)
        submodels = (value = deepcopy(model), state_encoder = deepcopy(state_encoder))
        dqn_target = (s_network = nothing, sp_network = nothing, func = v_target)
        subagent = Subagent(
            model,
            submodels,
            gamma,
            update_freq,
            update_cache,
            num_grad_steps,
            squared,
            optimizer(lr),
            "value",
            target = dqn_target,
            device = device,
            other_params = other_params,
        )
        subagents = [subagent]
        if isnothing(behavior)
            behavior = :deterministic
        end

    elseif AgentType == "ResidualQLearn"
        model = get_model("ActionValue", model_params)
        submodels = (action_value = deepcopy(model),)
        dqn_target = (s_network = nothing, sp_network = model, func = max_target)
        subagent = Subagent(
            model,
            submodels,
            gamma,
            update_freq,
            update_cache,
            num_grad_steps,
            squared,
            optimizer(lr),
            "action_value",
            target = dqn_target,
            device = device,
            other_params = other_params,
        )
        subagents = [subagent]
        π_b = model

    elseif AgentType == "PolicyGradient"
        policy_model = get_model("Policy", model_params)

        submodels = (policy = deepcopy(policy_model),)
        policy_target = (s_network = nothing, sp_network = nothing, func = max_target)

        policy_subagent = Subagent(
            policy_model,
            submodels,
            gamma,
            update_freq,
            update_cache,
            num_grad_steps,
            policygradient,
            optimizer(lr),
            "policy",
            target = policy_target,
            device = device,
            other_params = other_params,
        )

        subagents = [policy_subagent]
        π_b = policy_model

    else
        error("Not a valid agent type")
    end

    if behavior == :deterministic
        policy_model_params = model_params([[:seed, seed], [:hidden_size, 64]])
        π_b = get_model("Policy", policy_model_params)
    elseif behavior == :random
        #TODO random policy
    elseif behavior == :grad
        π_b = :grad
    elseif behavior === nothing
    elseif !isnothing(behavior)
        π_b = behavior
    end

    agent = Agent(
        subagents,
        buffers,
        state_encoder,
        action_encoder,
        π_b,
        max_agent_steps,
        measurement_freq,
        0,
        measurement_funcs,
        Dict(),
        device,
        rng,
        AgentType,
    )

    to_device!(agent, agent.device)

    return agent
end
