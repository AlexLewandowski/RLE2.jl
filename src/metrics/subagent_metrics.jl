import LinearAlgebra: nullspace, pinv, I

function calculate_grad_var(
    agent::AbstractAgent,
    env::AbstractEnv;
    buffer::AbstractBuffer,
    loss,
    num_batches,
    learner = nothing,
    targeter = nothing,
)
    grads_list = []
    m = agent.π.approximator
    for t = 1:num_batches
        StatsBase.sample(buffer)
        grads = gradient(m.params) do
            mean(minibatch_loss(agent, buffer, loss, learner = agent.π))
        end
        push!(grads_list, iterate(grads.grads[agent.π])[1][1][1][1])
    end
    g = grads_list[1]
    gs = []
    for layer in g
        i = 1
        for param in layer
            if param !== nothing
                push!(gs, [])
            end
        end
    end
    num_params = length(gs)
    for g in grads_list
        grads_list_temp = []
        i = 1
        for layer in g
            for param in layer
                if param !== nothing
                    push!(gs[i], param)
                    i += 1
                end
            end
        end
    end
    return mean(mean.(var.(gs)))
end

function estimate_value(
    agent::AbstractAgent,
    env::AbstractEnv;
    buffer = nothing,
    num_evals = 10,
)
    if buffer === nothing
        buffer = agent.buffers.train_buffer
    end
    vals = 0.0f0
    for _ = 1:num_evals
        StatsBase.sample(buffer)
        s, o, a, p, G, sp, op, done, info  = get_batch(buffer)

        Flux.reset!(agent.π)
        #TODO generalize here
        q = agent.π(s[:, 1, :])
        a = argmax(q)
        val = mean(q[a])
        vals += val
    end
    return [vals / num_evals]
end

function buffer_loss(subagent::AbstractSubagent, buffer; state_encoder, action_encoder)
    data = nothing
    try
        data = get_batch(buffer, subagent)
    catch
        StatsBase.sample(buffer)
        data = get_batch(buffer, subagent)
    end

    loss =
        mean(minibatch_loss(subagent, data, subagent.loss, state_encoder, action_encoder))
    name = ["loss_" * subagent.name * "_" * (typeof(buffer) <: Vector ? buffer[1].name : buffer.name)]
    update_count = subagent.update_count
    return loss, name, update_count
end

function td_start_loss(subagent::AbstractSubagent, buffer, action_encoder;) end

function td_start_loss(subagent::Subagent{P}, buffer; action_encoder, kwargs...) where {P<:Value}
    total = 0.0
    eval_loss = absolute
    idxs = buffer._episode_lengths .!== 0
    idxs[buffer._buffer_idx] = 0
    episodes = buffer._episodes[idxs]
    N = length(episodes)
    for episode in episodes
        o = episode[1].o
        op = episode[1].op
        r = episode[1].r
        estimate_o = sum(subagent(o |> subagent.device))
        estimate_op = sum(subagent(op |> subagent.device))
        target = r + estimate_op
        result = eval_loss(estimate_o, target)
        total += result
    end
end

function norm_minimetastate(subagent::AbstractSubagent, buffer; action_encoder)
    sample_size = buffer.batch_size
    StatsBase.sample(buffer, sample_size)
    s, o, a, p, r, sp, op, done, info  = get_batch(buffer)


    input, aux_input, s_dim, a_dim, L, T = get_input(o[:,1,:])

    result_max = maximum(input, dims = 1)
    result_norm = sum(input.^2, dims = 1)

    ns = []
    push!(ns, "maximum_minimetastate_" * subagent.name * "_" * (typeof(buffer) <: Vector ? buffer[1].name : buffer.name))
    push!(ns, "norm_minimetastate_" * subagent.name * "_" * (typeof(buffer) <: Vector ? buffer[1].name : buffer.name))
    return [mean(result_max), mean(result_norm)], ns
end


function td_buffer_loss(subagent::AbstractSubagent, buffer, action_encoder;) end

function td_buffer_loss(subagent::Subagent{P}, buffer; action_encoder, kwargs...) where {P<:Value}
    sample_size = -1
    sample_size = buffer.batch_size
    StatsBase.sample(buffer, sample_size)
    s, o, a, p, r, sp, op, done, info  = get_batch(buffer)

    mask = action_encoder(a) |> subagent.device
    estimate_o = sum(subagent(o |> subagent.device) .* mask, dims = 1)
    estimate_op = subagent(op |> subagent.device) #Target here?
    target = r + estimate_op

    eval_loss = absolute
    result = eval_loss(estimate_o, target)

    n = ["td_loss_" * subagent.name * "_" * (typeof(buffer) <: Vector ? buffer[1].name : buffer.name)]
    return [mean(result)], n
end

function td_buffer_loss(subagent::Subagent{P}, buffer; action_encoder, kwargs...) where {P<:AbstractActionValue}
    sample_size = -1
    sample_size = buffer.batch_size
    StatsBase.sample(buffer, sample_size)
    s, o, a, p, r, sp, op, done, info  = get_batch(buffer)

    mask = action_encoder(a) |> subagent.device
    estimate_o = sum(subagent(o |> subagent.device) .* mask, dims = 1)
    estimate_op = maximum(subagent(op |> subagent.device), dims = 1) #Target here?
    target = r + estimate_op

    eval_loss = absolute
    result = eval_loss(estimate_o, target)

    n = ["td_loss_" * subagent.name * "_" * (typeof(buffer) <: Vector ? buffer[1].name : buffer.name)]
    return [mean(result)], n
end


function mc_buffer_loss(subagent::AbstractSubagent, buffer, action_encoder;) end

function mc_buffer_loss(subagent::Subagent{P}, buffer; kwargs...) where {P<:Value}
    if buffer.bootstrap
        mc_buffer = montecarlo_buffer(buffer, 0.99f0)
    else
        mc_buffer = deepcopy(buffer)
    end
    sample_size = -1
    sample_size = nothing
    StatsBase.sample(mc_buffer, sample_size)

    s, o, a, p, G, sp, op, done, info  = get_batch(mc_buffer, subagent)

    estimate = subagent(o |> subagent.device)
    target = G

    eval_loss = absolute
    result = eval_loss(estimate, target)

    n = ["mc_loss_" * subagent.name * "_" * (typeof(buffer) <: Vector ? buffer[1].name : buffer.name)]
    return [mean(result)], n
end

function mc_buffer_loss(
    subagent::Subagent{P},
    buffer;
    action_encoder,
    state_encoder,
) where {P<:AbstractPolicy}

    return nothing
end


function mc_buffer_loss(
    subagent::Subagent{P},
    buffer;
    action_encoder,
    state_encoder,
) where {P<:AbstractActionValue}
    # total = 0.0
    # eval_loss = absolute
    # mc_buffer = montecarlo_buffer(buffer, 0.99f0)
    # idxs = buffer._episode_lengths .!== 0
    # idxs[buffer._buffer_idx] = 0
    # episodes = buffer._episodes[idxs]
    # N = length(episodes)
    # N = 1
    # ep_count = 1
    # for episode in episodes
    #     i = 1
    #     for exp in episode
    #         L = length(episode)
    #         o = exp.o |> state_encoder
    #         estimate_o = sum(subagent(o |> subagent.device))
    #         target = sum([exp.r for exp in episode[i:L]].*[0.99f0^(t-1) for t = 1:L-i+1])
    #         println("count", i, " - ", L)
    #         println(target)
    #         println(mc_buffer._episodes[ep_count][i].r)
    #         result = eval_loss(estimate_o, target)
    #         total += result
    #         i += 1
    #         N += 1
    #     end
    #     ep_count += 1
    # end
    # n = ["mc_start_loss_" * subagent.name * "_" * buffer.name]
    # println(N)
    # return [total / N], n
    if buffer.bootstrap
        mc_buffer = montecarlo_buffer(buffer, 0.99f0)
    else
        mc_buffer = deepcopy(buffer)
    end

    sample_size = -1
    sample_size = nothing
    StatsBase.sample(mc_buffer, batch_size = sample_size)
    s, o, a, p, G, sp, op, done, info  = get_batch(mc_buffer, subagent)

    mask = action_encoder(a) |> subagent.device
    estimate = subagent(o |> subagent.device, mask)
    target = G |> subagent.device
    # println(size(estimate))
    # println(size(target))
    # TODO add assertions?
    eval_loss = absolute
    result = eval_loss(estimate, target)

    n = ["mc_loss_" * subagent.name * "_" * (typeof(buffer) <: Vector ? buffer[1].name : buffer.name)]
    return [mean(result)], n
end

function mc_start_loss(subagent::AbstractSubagent, buffer, action_encoder;) end

function mc_start_loss(subagent::Subagent{P}, buffer; action_encoder, state_encoder, kwargs...) where {P<:ActionValue}
    total = 0.0
    eval_loss = absolute
    old_buffer = buffer
    buffer = montecarlo_buffer(buffer, 0.99f0)
    idxs = buffer._episode_lengths .!== 0
    idxs[buffer._buffer_idx] = 0
    episodes = buffer._episodes[idxs]
    N = length(episodes)
    i = 1
    for episode in episodes
        L = length(episode)
        o = episode[1].o |> state_encoder
        a = episode[1].a
        mask = action_encoder(a) |> subagent.device
        estimate_o = sum(subagent(o |> subagent.device).*mask)
        target = episode[1].r
        # target = sum([exp.r for exp in old_buffer._episodes[i]].*[0.99f0^(t-1) for t = 1:L])
        result = eval_loss(estimate_o, target)
        total += result
        i += 1
    end
    n = ["mc_start_loss_" * subagent.name * "_" * (typeof(buffer) <: Vector ? buffer[1].name : buffer.name)]
    return [total / N], n
end

function mc_start_loss(subagent::Subagent{P}, buffer; action_encoder, state_encoder, kwargs...) where {P<:Value}
    total = 0.0
    eval_loss = absolute
    old_buffer = buffer
    buffer = montecarlo_buffer(buffer, 0.99f0)
    idxs = buffer._episode_lengths .!== 0
    idxs[buffer._buffer_idx] = 0
    episodes = buffer._episodes[idxs]
    N = length(episodes)
    i = 1
    for episode in episodes
        L = length(episode)
        o = episode[1].o |> state_encoder
        estimate_o = sum(subagent(o |> subagent.device))
        target = episode[1].r
        # target = sum([exp.r for exp in old_buffer._episodes[i]].*[0.99f0^(t-1) for t = 1:L])
        result = eval_loss(estimate_o, target)
        total += result
        i += 1
    end
    n = ["mc_start_loss_" * subagent.name * "_" * (typeof(buffer) <: Vector ? buffer[1].name : buffer.name)]
    return [total / N], n
end

##
## Slope of representation
##

function rep_proj_residual(subagent::AbstractSubagent, buffer)
    sample_size = -1
    sample_size = buffer.batch_size
    StatsBase.sample(buffer, sample_size)
    s, o, a, p, G, sp, op, done, info  = get_batch(buffer)
    rep = representation(subagent, o[:, 1, :] |> subagent.device)
    xtx = transpose(rep) * rep
    id = I(size(xtx)[1])
    proj = rep * LinearAlgebra.pinv(xtx) * transpose(rep)
    final_W = layers(subagent)[end].W
    proj_W = final_W * proj
    r = mean((proj_W - final_W) .^ 2)
    n = ["res_proj_" * subagent.name * "_" * (typeof(buffer) <: Vector ? buffer[1].name : buffer.name)]
    return r, n
end


function rep_proj_slope(subagent::AbstractSubagent, buffer)
    sample_size = -1
    sample_size = buffer.batch_size
    StatsBase.sample(buffer, sample_size)
    s, o, a, p, G, sp, op, done, info  = get_batch(buffer)
    rep = representation(subagent, o[:, 1, :] |> subagent.device)
    null = nullspace(transpose(rep))
    final_W = layers(subagent)[end].W
    proj_W = final_W * null
    r = [mean((proj_W) .^ 2)]
    n = ["proj_null_slope_" * subagent.name * "_" * (typeof(buffer) <: Vector ? buffer[1].name : buffer.name)]
    return r, n
end

function rep_svd(subagent::AbstractSubagent, buffer)
    idx = sum(buffer._episode_lengths .!== 0)
    sample_size = -1
    sample_size = buffer.batch_size
    StatsBase.sample(buffer, sample_size)
    s, o, a, p, G, sp, op, done, info  = get_batch(buffer)
    rep = representation(subagent, o[:, 1, :] |> subagent.device)
    sing_val = svd(rep).S
    r = [minimum(sing_val), maximum(sing_val)]
    n = [
        "singular_value_min_" * subagent.name * "_" * (typeof(buffer) <: Vector ? buffer[1].name : buffer.name),
        "singular_value_max_" * subagent.name * "_" * (typeof(buffer) <: Vector ? buffer[1].name : buffer.name),
    ]
    return r, n
end

##
## Contextual Metrics
##
function accuracy(subagent::AbstractSubagent, env::AbstractEnv) end

function accuracy(subagent::Subagent{P}, data, per_class = false) where {P<:Union{AbstractActionValue, AbstractPolicy}}
    local_xs, local_ys, i, V, Q, A, subname = data
    names = []
    rs = []
    local_name = "accuracy_"*subname

    Q = subagent(local_xs |> subagent.device)
    a_est = [ind[1] for ind in argmax(Q, dims = 1)][:]

    N_test_points = length(local_ys)
    acc_list = [a_est[i] in A[local_ys[i]] for i = 1:N_test_points]


    if contains(subname, "train")
        # inds = acc_list .== 0;
        # println("misclass: ", length(acc_list[inds]))
        # println("tot: ", length(acc_list))
        # println(a_est[inds])
        # println(A[local_ys[inds]])
    end
    r = mean(acc_list)

    push!(rs, r)
    push!(names, local_name)
    return rs, names
end

function value_correlation(subagent::AbstractSubagent, env::AbstractEnv) end

function value_cor(subagent::Subagent{P}, data) where {P<:AbstractActionValue} #TODO get rid of hcats
    xs, ys, i, V, Q, A, subname = data
    names = []
    rs = []
    println(size(ys))
    local_name = "value_cor_"*subname
    if  i !== -1
        inds = ys .== i
        local_xs = xs[:,inds]
        local_ys = ys[inds]
    else
        local_xs = xs
        local_ys = ys
    end
    Q_est = subagent(local_xs |> subagent.device) |> cpu
    Q_truth = hcat([Q[i, :] for i in local_ys]...)
    L = length(local_ys)
    r = [StatsBase.cor(Q_est[:, i], Q_truth[:, i]) for i = 1:L]
    r = mean(r)
    push!(rs, r)
    push!(names, local_name)
    return rs, names
end

function action_value_cor(subagent::AbstractSubagent, env::AbstractEnv) end

function action_value_cor(subagent::Subagent{P}, data) where {P<:AbstractActionValue} #TODO get rid of hcats
    xs, ys, i, V, Q, A, subname = data
    names = []
    rs = []
    local_name = "action_value_cor_"*subname
    if  i !== -1
        inds = ys .== i
        local_xs = xs[:,inds]
        local_ys = ys[inds]
    else
        local_xs = xs
        local_ys = ys
    end
    Q_est = subagent(local_xs |> subagent.device) |> cpu
    Q_truth = hcat([Q[i, :] for i in local_ys]...)
    L = length(local_ys)
    r = [StatsBase.cor(Q_est[:, i], Q_truth[:, i]) for i = 1:L]
    r = mean(r)
    push!(rs, r)
    push!(names, local_name)
    return rs, names
end

function action_value_corspearman(subagent::AbstractSubagent, env::AbstractEnv) end

function action_value_corspearman(subagent::Subagent{P}, data) where {P<:AbstractActionValue} #TODO get rid of hcats
    xs, ys, i, V, Q, A, subname = data
    names = []
    rs = []
    local_name = "action_value_corspearman_"*subname
    if  i !== -1
        inds = ys .== i
        local_xs = xs[:,inds]
        local_ys = ys[inds]
    else
        local_xs = xs
        local_ys = ys
    end
    Q_est = subagent(local_xs |> subagent.device) |> cpu
    Q_truth = hcat([Q[i, :] for i in local_ys]...)
    L = length(local_ys)
    r = [StatsBase.corspearman(Q_est[:, i], Q_truth[:, i]) for i = 1:L]
    r = mean(r)
    push!(rs, r)
    push!(names, local_name)
    return rs, names
end

function MSE_Q_optimal(subagent::AbstractSubagent, env::AbstractEnv) end

function MSE_Q_optimal(subagent::Subagent{P}, data) where {P<:AbstractActionValue} #TODO get rid of hcats
    xs, ys, i, V, Q, A, subname = data
    names = []
    rs = []
    local_name = "MSE_optimal_"*subname
    if  i !== -1
        inds = ys .== i
        local_xs = xs[:,inds]
        local_ys = ys[inds]
    else
        local_xs = xs
        local_ys = ys
    end
    Q_est = subagent(local_xs |> subagent.device) |> cpu
    Q_truth = hcat([Q[i, A[i][1]] for i in local_ys]...)
    Q_est = hcat([Q_est[A[local_ys[i]][1], i] for i = 1:length(local_ys)]...)
    r = mean((Q_est - Q_truth) .^ 2)
    push!(rs, r)
    push!(names, local_name)
    return rs, names
end

function MSE_Q_all(subagent::AbstractSubagent, env::AbstractEnv) end

function MSE_Q_all(subagent::Subagent{P}, data, per_class = false) where {P<:AbstractActionValue}
    xs, ys, i, V, Q, A, subname = data
    names = []
    rs = []
    local_name = "MSE_all_"*subname
    if  i !== -1
        inds = ys .== i
        local_xs = xs[:,inds]
        local_ys = ys[inds]
    else
        local_xs = xs
        local_ys = ys
    end
    Q_est = subagent(local_xs |> subagent.device) |> cpu
    Q_truth = hcat([Q[i, :] for i in local_ys]...)
    r = mean((Q_est - Q_truth) .^ 2)
    push!(rs, r)
    push!(names, local_name)
    return rs, names
end

function MSE_Q_max(subagent::AbstractSubagent, env::AbstractEnv) end

function MSE_Q_max(subagent::Subagent{P}, data, per_class = false) where {P<:AbstractActionValue}
    xs, ys, i, V, Q, A, subname = data
    names = []
    rs = []
    local_name = "MSE_max_"*subname
    if  i !== -1
        inds = ys .== i
        local_xs = xs[:,inds]
        local_ys = ys[inds]
    else
        local_xs = xs
        local_ys = ys
    end
    Q_est = subagent(local_xs |> subagent.device) |> cpu
    Q_truth = hcat([Q[i, :] for i in local_ys]...)
    r = mean(maximum((Q_est - Q_truth) .^ 2, dims = 1))
    push!(rs, r)
    push!(names, local_name)
    return rs, names
end

function MSE_action_not_taken(subagent::AbstractSubagent, env::AbstractEnv) end

function MSE_action_not_taken(subagent::Subagent{P}, data, per_class = false) where {P<:AbstractActionValue}
    local_xs, local_ys, i, V, Q, A, subname = data
    names = []
    rs = []
    local_name = "MSE_action_not_taken_"*subname
    num_states = length(A)
    num_actions = maximum([A[i][1] for i = 1:num_states])
    a_not_taken = forbidden_action(A[i][1], i, num_actions)
    Q_est = subagent(local_xs |> subagent.device) |> cpu
    Q_est = Q_est[a_not_taken, :]
    Q_truth = vcat([Q[i, a_not_taken][1] for i in local_ys]...)
    r = mean((Q_est - Q_truth) .^ 2)
    push!(rs, r)
    push!(names, local_name)
    return rs, names
end

function action_gap(subagent::AbstractSubagent, data)
    xs, ys, i, V, Q, A, subname = data
    names = []
    rs = []
    local_name = "action_gap_"*subname
    if  i !== -1
        inds = ys .== i
        local_xs = xs[:,inds]
        local_ys = ys[inds]
    else
        local_xs = xs
        local_ys = ys
    end
    Q_est = subagent(local_xs |> subagent.device) |> cpu
    Q_max = maximum(Q_est, dims = 1)
    Q_min = minimum(Q_est, dims = 1)
    r = mean(Q_max - Q_min)
    push!(rs, r)
    push!(names, local_name)
    return rs, names
end
