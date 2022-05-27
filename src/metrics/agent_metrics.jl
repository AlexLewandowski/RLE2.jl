import Flux: Dense, Recur, gradient
import StatsBase: var
import LinearAlgebra: svd

function online_returns(agent::AbstractAgent, env::AbstractEnv)
    buffer = agent.buffers.train_buffer
    gamma = agent.subagents[1].gamma
    rs, names = online_returns(buffer, gamma)
    if typeof(env) <: OptEnv
        last_ep_idx = buffer._buffer_idx - 1
        if last_ep_idx == 0
            last_ep_idx = buffer.max_num_episodes
        end
        last_ep = buffer._episodes[last_ep_idx]

        push!(names, "online_student_loss")
        push!(rs, Float32.(last_ep[end].info[1][2]))

        push!(names, "online_student_acc")
        push!(rs, Float32.(last_ep[end].info[1][1]))

        push!(names, "online_student_acc_test")
        push!(rs, Float32.(last_ep[end].info[1][3]))
    end
    return rs, names
end

function online_returns(buffer::AbstractBuffer, gamma)
    last_ep_idx = buffer._buffer_idx - 1
    if last_ep_idx == 0
        last_ep_idx = buffer.max_num_episodes
    end
    last_ep = buffer._episodes[last_ep_idx]
    if !buffer.bootstrap
        r = [last_ep[1].r]
        as = reshape([mean(exp.a) for exp in last_ep], :)
        push!(r, length(as))
    else
        rs = reshape([exp.r for exp in last_ep], (1, :, 1))
        r = nstep_returns(gamma, rs) # nstep_returns outputs an array
        push!(r, length(rs))
        as = reshape([mean(exp.a) for exp in last_ep], :)
    end

        push!(r, mean(as))
        push!(r, var(as, corrected = false))
    names = ["online_returns", "online_num_steps", "online_average_action", "online_action_variance"]
    return r, names
end

function contextualmdp_metric(agent, env, f)
    local_name = string(f)
    list, train_dist, policy_dist, optimal_dist, start_state = get_list_for_metric(agent, env)
    r, n = apply_to_list(f, list)

    rs = []
    ns = []

    n_states = length(get_valid_nonterminal_states(env.MDP))

    train = []
    train_counts = []

    test = []
    test_counts = []

    val = []
    val_counts = []

    for i = 1:n_states
        push!(train, r[i])
        push!(test, r[i+n_states])
        push!(val, r[i+n_states*2])
        push!(train_counts, length(list[i][2][2]))
        push!(test_counts, length(list[i+n_states][2][2]))
        push!(val_counts, length(list[i+n_states*2][2][2]))
    end

    push!(rs, sum(train_counts.*train)/sum(train_counts))
    push!(rs, sum(test_counts.*test)/sum(test_counts))
    push!(rs, sum(val_counts.*val)/sum(val_counts))
    push!(rs, (sum(val_counts.*val) + sum(train_counts.*train) + sum(test_counts.*test))/
        (sum(train_counts) + sum(test_counts) + sum(val_counts)))

    push!(ns, local_name*"_train")
    push!(ns, local_name*"_test")
    push!(ns, local_name*"_val")
    push!(ns, local_name*"_union")

    push!(rs, sum(train_dist.*train)/sum(train_dist))
    push!(rs, sum(train_dist.*test)/sum(train_dist))
    push!(rs, sum(train_dist.*val)/sum(train_dist))
    push!(rs, (sum(train_dist.*val) + sum(train_dist.*train) + sum(train_dist.*test))/
        (sum(train_dist) + sum(train_dist) + sum(train_dist)))
    push!(ns, local_name*"_train_traindist")
    push!(ns, local_name*"_test_traindist")
    push!(ns, local_name*"_val_traindist")
    push!(ns, local_name*"_union_traindist")

    push!(rs, sum(policy_dist.*train)/sum(policy_dist))
    push!(rs, sum(policy_dist.*test)/sum(policy_dist))
    push!(rs, sum(policy_dist.*val)/sum(policy_dist))
    push!(rs, (sum(policy_dist.*val) + sum(policy_dist.*train) + sum(policy_dist.*test))/
        (sum(policy_dist) + sum(policy_dist) + sum(policy_dist)))
    push!(ns, local_name*"_train_policydist")
    push!(ns, local_name*"_test_policydist")
    push!(ns, local_name*"_val_policydist")
    push!(ns, local_name*"_union_policydist")

    push!(rs, sum(optimal_dist.*train)/sum(optimal_dist))
    push!(rs, sum(optimal_dist.*test)/sum(optimal_dist))
    push!(rs, sum(optimal_dist.*val)/sum(optimal_dist))
    push!(rs, (sum(optimal_dist.*val) + sum(optimal_dist.*train) + sum(optimal_dist.*test))/
        (sum(optimal_dist) + sum(optimal_dist) + sum(optimal_dist)))
    push!(ns, local_name*"_train_optimaldist")
    push!(ns, local_name*"_test_optimaldist")
    push!(ns, local_name*"_val_optimaldist")
    push!(ns, local_name*"_union_optimaldist")

    if env.MDP.name == "FourRooms"
        n_wall_states = length(get_valid_wall_states(env.MDP))

        wall_train = []
        wall_train_counts = []

        wall_test = []
        wall_test_counts = []

        wall_val = []
        wall_val_counts = []

        for i = 1:n_wall_states
            ofst = n_states*2+1
            push!(wall_train, r[i+ofst])
            push!(wall_test, r[i+ofst+n_wall_states])
            push!(wall_val, r[i+ofst+n_wall_states*2])
            push!(wall_train_counts, length(list[i+ofst][2][2]))
            push!(wall_test_counts, length(list[i+ofst+n_wall_states][2][2]))
            push!(wall_val_counts, length(list[i+ofst+n_wall_states*2][2][2]))
        end

        push!(rs, sum(wall_train_counts.*wall_train)/sum(wall_train_counts))
        push!(rs, sum(wall_test_counts.*wall_test)/sum(wall_test_counts))
        push!(rs, sum(wall_val_counts.*wall_val)/sum(wall_val_counts))
        push!(ns, local_name*"_wall_train")
        push!(ns, local_name*"_wall_test")
        push!(ns, local_name*"_wall_val")
    end
    push!(rs, rs[1] - rs[2])
    push!(ns, local_name*"_gap")

    push!(rs, rs[5] - rs[6])
    push!(ns, local_name*"_gap_traindist")

    push!(rs, rs[9] - rs[10])
    push!(ns, local_name*"_gap_policydist")

    push!(rs, rs[13] - rs[14])
    push!(ns, local_name*"_gap_optimaldist")

    rs = push!(rs, r...)
    ns = push!(ns, n...)
    return rs, ns
end

function value_cor(agent, env::AbstractEnv)
    contextualmdp_metric(agent, env, value_cor)
end

function value_corspearman(agent, env::AbstractEnv)
    contextualmdp_metric(agent, env, value_corspearman)
end

function action_value_cor(agent, env::AbstractEnv)
    contextualmdp_metric(agent, env, action_value_cor)
end

function action_value_corspearman(agent, env::AbstractEnv)
    contextualmdp_metric(agent, env, action_value_corspearman)
end

function accuracy(agent, env::ContextualMDP)
    contextualmdp_metric(agent, env, accuracy)
end

function accuracy(agent, env::AbstractEnv)
    nothing
end

function MSE_Q_optimal(agent, env::AbstractEnv)
    contextualmdp_metric(agent, env, MSE_Q_optimal)
end

function MSE_Q_all(agent, env::AbstractEnv)
    contextualmdp_metric(agent, env, MSE_Q_all)
end

function MSE_Q_max(agent, env::AbstractEnv; per_class = false)
    contextualmdp_metric(agent, env, MSE_Q_max)
end

function MSE_action_not_taken(agent, env::AbstractEnv; per_class = false)
    contextualmdp_metric(agent, env, MSE_action_not_taken)
end

function action_gap(agent, env::AbstractEnv; per_class = false)
    list, start_state = get_list_for_metric(agent, env)
    r, n = apply_to_list(action_gap, list)
    rs = [r...]
    ns = [n...]
    start_state_result_ind = 6
    push!(rs, rs[2] - rs[1])
    push!(ns, "action_gap_gap")
    push!(rs, rs[start_state_result_ind+start_state] - rs[start_state_result_ind])
    push!(ns, "action_gap_startstate_gap")
    return rs, ns
end

# TODO reconcile counterfactual for different models
function counterfactual(
    agent,
    env;
    buffer::TransitionReplayBuffer,
    loss,
    num_evals = 10,
)
    cf_buffer = deepcopy(buffer)
    counterfactual = 0.0f0
    factual = 0.0f0
    window = buffer.history_window + buffer.predict_window
    for _ = 1:num_evals
        StatsBase.sample(buffer)
        factual += mean(minibatch_loss(agent, buffer, loss))
        s, o, a, p, r, sp, op, done, info  = get_batch(buffer)

        new_r = copy(r)
        new_a = copy(a)

        for i = 1:cf_buffer.batch_size
            flip_a = rand(env.action_space)
            while flip_a == a[1, 1, i]
                flip_a = rand(env.action_space)
            end
            old_s = s[:, 1, i]

            ep = generate_episode(
                env,
                π = agent.π_b,
                state = old_s,
                action = flip_a,
                max_steps = agent.max_agent_steps,
            )

            if !buffer.bootstrap
                ep = montecarlo_episode(ep)
            end
            ep = windowed_episode(ep, window, cf_buffer.overlap)
            fill_buffer!(cf_buffer, ep[1], i)

        end
        counterfactual += mean(minibatch_loss(agent, cf_buffer, loss))
    end
    return [counterfactual / num_evals, factual / num_evals]
end

function buffer_loss(agent::AbstractAgent, env = nothing; num_evals = 100)
    list = collect(Iterators.product(agent.subagents, agent.buffers)) |> vec
    return apply_to_list(
        buffer_loss,
        list,
        state_encoder = agent.state_encoder,
        action_encoder = agent.action_encoder,
    )
end

function mc_start_miscal_loss(agent, env; num_evals = 100)
    list = collect(Iterators.product(agent.subagents, agent.buffers)) |> vec
    return apply_to_list(mc_start_miscal_loss, list,
        state_encoder = agent.state_encoder,
        action_encoder = agent.action_encoder,)
end

function mc_start_loss(agent, env; num_evals = 100)
    list = collect(Iterators.product(agent.subagents, agent.buffers)) |> vec
    return apply_to_list(mc_start_loss, list,
        state_encoder = agent.state_encoder,
        action_encoder = agent.action_encoder,)
end

function mc_buffer_loss(agent, env; num_evals = 100)
    list = collect(Iterators.product(agent.subagents, agent.buffers)) |> vec
    return apply_to_list(mc_buffer_loss, list,
        state_encoder = agent.state_encoder,
        action_encoder = agent.action_encoder,)
end

function td_buffer_loss(agent, env; num_evals = 100)
    list = collect(Iterators.product(agent.subagents, agent.buffers)) |> vec
    return apply_to_list(td_buffer_loss, list, action_encoder = agent.action_encoder)
end

function norm_minimetastate(agent, env; num_evals = 1)
    list = collect(Iterators.product(agent.subagents, agent.buffers)) |> vec
    return apply_to_list(norm_minimetastate, list, action_encoder = agent.action_encoder)
end

##
## Slope of representation
##

function rep_svd(agent, env; num_evals = 1)
    list = collect(Iterators.product(agent.subagents, agent.buffers)) |> vec
    return apply_to_list(rep_svd, list)
end

function rep_proj_residual(agent, env)
    list = collect(Iterators.product(agent.subagents, agent.buffers)) |> vec
    return apply_to_list(rep_proj_residual, list)
end


function rep_proj_slope(agent, env)
    list = collect(Iterators.product(agent.subagents, agent.buffers)) |> vec
    return apply_to_list(rep_proj_slope, list)
end
