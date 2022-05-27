function rollout_returns_tanh(
    agent,
    env::AbstractEnv;
    num_evals = 30,
    greedy = true,
    policy = nothing,
    max_steps = nothing,
)
    test_env = env_mode!(env, mode = :tanh)
    rs, ns = rollout_returns(agent, test_env; num_evals = num_evals, greedy = greedy, max_steps = max_steps, policy = policy)

    for i = 1:length(ns)
        ns[i] = ns[i] * "_tanh"
    end

    return rs, ns
end

function rollout_returns_fashion(
    agent,
    env::AbstractEnv;
    num_evals = 30,
    greedy = true,
    policy = nothing,
    max_steps = nothing,
)

    transfer_EnvType = "OptEnv-logLP-fashion-ADAM"

    env, max_agent_steps, embedding_f = RLE2.get_env(
        transfer_EnvType,
        seed = rand(env.rng, 1:10000),
        max_steps = agent.max_agent_steps,
        state_representation = env.state_representation,
    )

    transfer_env = env
    rs, ns = rollout_returns(agent, transfer_env; num_evals = num_evals, greedy = greedy, max_steps = max_steps, policy = policy)

    for i = 1:length(ns)
        ns[i] = ns[i] * "_fashion"
    end

    return rs, ns
end

function rollout_returns_mnist(
    agent,
    env::AbstractEnv;
    num_evals = 30,
    greedy = true,
    policy = nothing,
    max_steps = nothing,
)

    transfer_EnvType = "OptEnv-logLP-mnist-ADAM"

    env, max_agent_steps, embedding_f = RLE2.get_env(
        transfer_EnvType,
        seed = rand(env.rng, 1:10000),
        max_steps = agent.max_agent_steps,
        state_representation = env.state_representation,
    )

    transfer_env = env
    rs, ns = rollout_returns(agent, transfer_env; num_evals = num_evals, greedy = greedy, max_steps = max_steps, policy = policy)

    for i = 1:length(ns)
        ns[i] = ns[i] * "_mnist"
    end

    return rs, ns
end

function rollout_returns_wide(
    agent,
    env::AbstractEnv;
    num_evals = 30,
    greedy = true,
    policy = nothing,
    max_steps = nothing,
)
    test_env = env_mode!(env, mode = :wide)
    rs, ns = rollout_returns(agent, test_env; num_evals = num_evals, greedy = greedy, max_steps = max_steps, policy = policy)

    for i = 1:length(ns)
        ns[i] = ns[i] * "_wide"
    end

    return rs, ns
end

function rollout_returns_narrow(
    agent,
    env::AbstractEnv;
    num_evals = 30,
    greedy = true,
    policy = nothing,
    max_steps = nothing,
)
    test_env = env_mode!(env, mode = :narrow)
    rs, ns = rollout_returns(agent, test_env; num_evals = num_evals, greedy = greedy, max_steps = max_steps, policy = policy)

    for i = 1:length(ns)
        ns[i] = ns[i] * "_narrow"
    end

    return rs, ns
end

function rollout_returns_constant(
    agent,
    env::AbstractEnv;
    num_evals = 10,
    max_steps = nothing,
    greedy = true,
)

    rs = []
    ns = []
    for a = 1:5
        r, n = rollout_returns(agent, env, policy = a, num_evals = num_evals)
        push!(rs, r...)
        n = n.*("_constant="*string(a))
        push!(ns, n...)
    end
    return rs, ns
end

function rollout_returns(
    agent,
    env::AbstractEnv;
    policy = nothing,
    max_steps = nothing,
    num_evals = 30,
    greedy = true,
    gamma = nothing
)
    if max_steps == nothing
        max_steps = agent.max_agent_steps
        # max_steps = 100 #TODO TEMPORARY FOR GEN-RL EXPERIMENTS
    end

    if isnothing(gamma)
        gamma = agent.subagents[1].gamma
    end

    if policy == nothing
        policy = agent.π_b
    end

    Gs = []
    lens = []
    mean_as = []
    var_as = []
    accs = []
    accs_test = []
    losses = []

    aux = 0.0f0
    for _ = 1:num_evals
        #TODO is greedy the right thing to do?
        ep = generate_episode(
            env,
            agent,
            policy = policy,
            greedy = greedy,
            max_steps = max_steps,
        )

        if typeof(env) <: OptEnv
            push!(accs, env.acc[1])
            push!(losses, env.acc[2])
            push!(accs_test, env.acc[3])
        end

        rs = reshape([exp.r for exp in ep], (1, :, 1))
        as = reshape([mean(exp.a) for exp in ep], :)
        push!(mean_as, mean(as))
        push!(var_as, var(as, corrected = false))
        push!(Gs, nstep_returns(gamma, rs)[1])
        push!(lens, Float32(length(rs)))
        if !isnothing(agent)
        end
    end
    returns = [mean(Gs), mean(lens), maximum(lens), minimum(lens), mean(mean_as), mean(var_as)]
    names = ["rollout_returns", "num_steps", "max_steps", "min_steps", "average_action", "action_variance"]

    if typeof(env) <: OptEnv
        push!(names, "student_loss")
        push!(returns, Float32.(mean(losses)))

        push!(names, "student_accuracy")
        push!(returns, Float32.(mean(accs)))

        push!(names, "student_accuracy_test")
        push!(returns, Float32.(mean(accs_test)))

        # push!(names, "student_accuracy_test_max")
        # push!(returns, Float32.(maximum(accs_test)))

        # push!(names, "student_accuracy_test_min")
        # push!(returns, Float32.(minimum(accs_test)))
    end

    return returns, names
end


function estimate_startvalue(agent::AbstractAgent, env::AbstractEnv;)
    vals = 0.0f0
    buffer = agent.buffers.train_buffer
    N_episodes = sum(buffer._episode_lengths .> 0)

    inds = collect(1:N_episodes)
    if N_episodes == length(buffer._episode_lengths)
        deleteat!(inds, buffer._buffer_idx)
    end
    num_evals = length(inds)

    for ind in inds
        s = buffer._episodes[ind][1].o
        if typeof(agent.subagents[1].model) <: AbstractContinuousActionValue
            a = agent.subagents[3](s)
            val = agent.subagents[1](s |> agent.device, a)[1]
        else
            q = agent.subagents[1](s |> agent.device)
            val = maximum(q)
        end

        vals += val
    end
    r = [vals / num_evals]
    name = ["Estimate of start-state value"]
    return r, name
end

function estimate_termvalue(agent::AbstractAgent, env::AbstractEnv;)
    vals = 0.0f0
    buffer = agent.buffers.train_buffer
    N_episodes = sum(buffer._episode_lengths .> 0)

    inds = collect(1:N_episodes)
    if N_episodes == length(buffer._episode_lengths)
        deleteat!(inds, buffer._buffer_idx)
    end
    num_evals = length(inds)

    count = 0
    for ind in inds
        if buffer._episodes[ind][end].done
            count += 1
            s = buffer._episodes[ind][end].o
            if typeof(agent.subagents[1].model) <: AbstractContinuousActionValue
                a = agent.subagents[3](s)
                val = agent.subagents[1](s |> agent.device, a)[1]
            else
                q = agent.subagents[1](s |> agent.device)
                val = maximum(q)
            end
            vals += val
        end
    end

    if count == 0
        count += 1
    end

    r = [vals / count]
    name = ["Estimate of term-state value"]
    return r, name
end

function rollout_returns_eval_env(
    agent,
    env::AbstractEnv;
    num_evals = 10,
    greedy = true,
)
    test_env = env_mode!(env, mode = :test)
    rs, ns = rollout_returns(agent, test_env; num_evals = num_evals, greedy = greedy)

    for i = 1:length(ns)
        ns[i] = ns[i] * "_test"
    end

    return rs, ns
end

function rollout_returns_eval_env_no_nest(
    agent,
    env::AbstractEnv;
    num_evals = 100,
    greedy = true,
)
    test_env = env_mode!(env, mode = :no_nest)
    rs, ns = rollout_returns(agent, test_env; num_evals = num_evals, greedy = greedy)

    for i = 1:length(ns)
        ns[i] = ns[i] * "_test_mnist"
    end

    return rs, ns
end

function rollout_returns_optenv(
    agent,
    env::AbstractOptEnv;
    policy = :agent,
    num_evals = 1,
    max_steps = nothing,
    greedy = true,
    mode = "default"
)
    meta_accs = []
    meta_accs_test = []
    meta_as = []
    meta_lr = []
    meta_losses = []
    meta_norms = []

    if max_steps === nothing
        max_steps = agent.max_agent_steps
    end
    if (contains(env.state_representation, "parameters") || contains(env.state_representation, "PEN")) &&
        (mode == "narrow"  || mode == "wide")
        return zeros(Float32, max_steps), zeros(Float32, max_steps), zeros(Float32, max_steps), zeros(Float32, max_steps), zeros(Float32, max_steps), zeros(Float32, max_steps)
    end

    for _ = 1:num_evals
        reset!(env)
        as = []
        accs = [env.acc[1]]
        accs_test = [env.acc[3]]
        lrs = []
        losses = [env.acc[2]]
        norms = []
        for j = 1:max_steps
            ex, _ = interact!(env, agent, greedy, policy = policy)

            est_norm = calc_norm(env)
            push!(as, ex.a)
            push!(accs, env.acc[1])
            push!(lrs, env.opt.eta)
            push!(losses, env.acc[2])
            push!(accs_test, env.acc[3])
            push!(norms, est_norm)
        end
        push!(meta_as, as)
        push!(meta_accs, accs)
        push!(meta_accs_test, accs_test)
        push!(meta_lr, lrs)
        push!(meta_losses, losses)
        push!(meta_norms, norms)
    end
    return mean(meta_accs), mean(meta_as), mean(meta_lr), mean(meta_losses), mean(meta_accs_test), mean(meta_norms)
end

function rollout_returns_curriculum(
    agent,
    env::AbstractCurriculumMDP;
    policy = :agent,
    num_evals = 1,
    max_steps = nothing,
    greedy = true,
)
    meta_teacher_Gs = []
    meta_teacher_num_steps = []
    meta_inds = []
    meta_as = []

    total_reports = 20
    if max_steps === nothing
        max_steps = env.max_steps
    end

    if max_steps < total_reports
        total_reports = max_steps
    end

    measurement_freq = floor(Int, max_steps / total_reports)

    max_steps += 1

    for _ = 1:num_evals

        if typeof(agent) <: AbstractAgent
            reset_model!(agent.state_encoder)
        end

        teacher_Gs = []
        teacher_num_steps = []
        teacher_as = []
        inds = []
        reset!(env)
        if is_tabular(env.student_env)
            num_student_evals = 1
        else
            num_student_evals = 1
        end

        for j = 1:max_steps
            result = rollout_returns_nonmean(
                env.student_agent,
                env.student_env,
                num_evals = num_student_evals,
                greedy = greedy,
            )[1]
            G = mean(result[1])
            lens = reshape(result[2], :)
            # G = mean(lens .< 199)
            ex, _ = interact!(env, agent, greedy, policy = policy)

            push!(teacher_as, ex.a[1])
            push!(teacher_Gs, G)
            push!(teacher_num_steps, mean(result[2]))
            push!(inds, j - 1)
        end
        push!(meta_teacher_Gs, teacher_Gs)
        push!(meta_teacher_num_steps, teacher_num_steps)
        push!(meta_inds, inds)
        push!(meta_as, teacher_as)
    end

    return mean(meta_teacher_Gs), mean(meta_teacher_num_steps), mean(meta_as), mean(meta_inds)
end

function rollout_returns_nonmean(
    agent,
    env::AbstractEnv;
    policy = nothing,
    num_evals = 1,
    max_steps = nothing,
    greedy = true,
)
    if max_steps == nothing
        max_steps = agent.max_agent_steps
    end

    if policy == nothing
        policy = agent.π_b
    end

    Gs = []
    lens = []
    aux = 0.0f0
    for _ = 1:num_evals
        ep = generate_episode(
            env,
            agent,
            policy = policy,
            greedy = greedy,
            max_steps = max_steps,
        )
        rs = reshape([exp.r for exp in ep], (1, :, 1))
        push!(Gs, nstep_returns(agent.subagents[1].gamma, rs)[1])
        push!(lens, Float32(length(rs)))
    end
    returns = [Gs, lens]
    names = ["rollout_returns", "num_steps"]

    return returns, names
end
