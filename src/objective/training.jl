include("objective_utils.jl")
include("classic_objectives.jl")
include("regularizers.jl")

function onpolicy_training(agent, buffer::AbstractBuffer, step, num_training_eps = 4;)
    if step % num_training_eps == 0
        train_loop(agent, buffer)
        reset_experience!(buffer)
    end
end

function train_subagents(
    agent;
    buffer = nothing,
    eval_loss = nothing,
    num_grad_steps = nothing,
    training = true,
    batch_size = nothing,
)
    for subagent in agent.subagents
        train_subagent(
            agent,
            subagent,
            buffer,
            eval_loss,
            num_grad_steps,
            training,
            batch_size,
        )
        if mod(subagent.update_count, subagent.update_freq) == 0
            update_submodel!(subagent, "state_encoder"; new_ps = get_params(agent.state_encoder))
        end
    end
end

function get_grads(subagent, buffer, eval_loss, state_encoder, action_encoder, reg)
    data = get_batch(buffer, subagent)
    grads = gradient(subagent.params) do
        l = mean(minibatch_loss(subagent, data, eval_loss, state_encoder, action_encoder, reg = reg))
    end
    return grads
end

function train_subagent(
    subagent,
    buffer,
    state_encoder,
    action_encoder;
    eval_loss = nothing,
    num_grad_steps = nothing,
    training = true,
    batch_size = nothing,
    reg = false,
    ind_range = nothing,
)
    if eval_loss === nothing
        eval_loss = subagent.loss
    end

    if num_grad_steps === nothing
        num_grad_steps = subagent.num_grad_steps
    end

    for _ = 1:num_grad_steps
        if subagent.target.sp_network == nothing
            bootstrap = false
        else
            bootstrap = true
        end
        # StatsBase.sample(buffer, batch_size = batch_size, bootstrap = bootstrap, ind_range = ind_range)
        StatsBase.sample(buffer, batch_size = batch_size, bootstrap = bootstrap)
        if typeof(state_encoder) <: SequenceNeuralNetwork
            # update_hidden_states(buffer, subagent.gamma, subagent.device, action_encoder, state_encoder)
            # data = get_batch(buffer, subagent.gamma, subagent.device, action_encoder)
            # println("inds: ", buffer._indicies)
            # println("H pre: ", sum(data[2][1:64,:,:]))
            # println("dat pre: ", sum(data[2][129:end,:,:]))

            # if typeof(state_encoder.f[1][1]) <: Flux.Recur
            #     update_hidden_states(buffer, subagent.gamma, subagent.device, action_encoder, state_encoder, subagent.submodels.state_encoder)
            # end

            # data = get_batch(buffer, subagent.gamma, subagent.device, action_encoder)
            # println("inds: ", buffer._indicies)
            # println("H post: ", sum(data[2][1:64,:,:]))
            # println("dat post: ", sum(data[2][129:end,:,:]))
            # init_hidden_states(buffer, subagent.gamma, subagent.device, action_encoder, state_encoder)
        end

        # b2 = deepcopy(buffer)
        # update_hidden_states(buffer, subagent.gamma, subagent.device, action_encoder, state_encoder)
        # inds = buffer._indicies

        # init_hidden_states(b2, subagent.gamma, subagent.device, action_encoder, state_encoder)
        # err = 0f0
        # for ind in inds
        #     x = ind[1]
        #     y = ind[2]
        #     b2t = sum(b2._episodes[x][y].sp[1:26])
        #     bt = sum(buffer._episodes[x][y].sp[1:26])

        #     err += (b2t - bt)^2
        # end
        # println(err/length(inds))

        learner = subagent.model
        if training
            # TODO Training only or anytime sample?
            if mod(subagent.update_count, subagent.train_freq) == 0
                if typeof(subagent.model.f) <: Tabular
                    tabular_learning(subagent, buffer, eval_loss, state_encoder, action_encoder)
                else
                    grads = get_grads(subagent, buffer, eval_loss, state_encoder, action_encoder, reg)
                    update!(subagent, grads)
                end
            end
            subagent.update_count += 1
            callback(subagent)
        else
            data = get_batch(buffer, subagent)
            loss =
                mean(minibatch_loss(subagent, data, eval_loss, state_encoder, action_encoder))
            name = ["loss_" * subagent.name * "_" * (typeof(buffer) <: Vector ? buffer[1].name : buffer.name)]
            update_count = subagent.update_count
            return loss, name, update_count
        end
    end
end

function tabular_learning(subagent, exp)
    o = exp.o
    op = exp.op
    a = exp.a[1]
    r = exp.r
    done = exp.done
    V = subagent.model.f(o)[a]
    if done
        Vp = 0
    else
        Vp = subagent.model.f(op)
        Vp = maximum(Vp)
    end
    δ = r + subagent.gamma*Vp - V
    update!(subagent.model.f, o, a, δ, subagent.optimizer.eta)
    nothing
end

function tabular_learning(subagent, buffer, eval_loss, state_encoder, action_encoder)
    #L = minibatch_loss(subagent, buffer, eval_loss, state_encoder, action_encoder, reg = reg)
    B = size(buffer._s_batch)[end]
    s, o, a, p, r, sp, op, done, info  = get_batch(buffer)
    ss = [s[:,1,i] for i = 1:B]
    os = [o[:,1,i] for i = 1:B]
    as = [a[:,1,i] for i = 1:B]
    sps = [sp[:,1,i] for i = 1:B]
    ops = [op[:,1,i] for i = 1:B]
    rs = [r[1,1,i] for i = 1:B]
    for (o, a, op, r) in zip(os,as,ops,rs)
        a = a[1]
        V = subagent.model.f(o)[a]
        Vp = subagent.model.f(op)
        Vp = maximum(Vp)
        δ = (r + subagent.gamma*Vp - V)
        update!(subagent.model.f, o, a, δ, subagent.optimizer.eta)
    end
end

function minibatch_loss(
    subagent::Subagent{P},
    data,
    loss,
    state_encoder,
    action_encoder;
    print = false,
    reg = 0f0,
) where {P<:AbstractModel}
    learner = subagent.model

    Flux.trainmode!(learner.f.f)
    reset_model!(learner)
    reset_model!(state_encoder, :zeros)
    reset_model!(subagent.submodels.state_encoder, :zeros)
    _, ob, a, p, r, _, obp, done, info, discounted_reward_sum, mask, ap, maskp = data

    # ob, obp = stop_gradient() do
    #     state_encoder(ob), state_encoder(obp)
    # end

    # # ob, obp = state_encoder(ob), state_encoder(obp)

    # done = stop_gradient() do
    #     done |> subagent.device
    # end

    M = size(r)[3]
    T = size(r)[2]
    # if T > 1
    #     ap = a[:, 2:T, :]
    #     T = T - 1
    #     maskp = stop_gradient() do
    #         action_encoder(ap) #|> subagent.device
    #     end
    # else
    #     ap = Bool.(zeros(size(a)))
    #     maskp = Bool.(zeros(size(a)))
    # end

    gamma = Zygote.dropgrad(subagent.gamma)
    device = Zygote.dropgrad(subagent.device)

    loss_total = zeros(Float32, M) |> subagent.device
    # println(M)

    # for t = 1:T
    t = 1
        o_t = ob[:, t, :]
        a_t = a[:, t, :][:]
        p_t = reshape(p[:, t, :], :)
        r_t = discounted_reward_sum[:, t, :]
        # r_t = r[:, t, :]
        op_t = obp[:, end, :]
        # ap_t = ap[:, t, :][:]
        done_t = done[:, end, :]
        mask_t = mask[:, t, :]
        maskp_t = maskp[:, end, :]
        # TODO: subagent as function arg heurts here
        # estimate = zeros(Float32, (1, M))
        # target = zeros(Float32, (1, M))
        # println(mask_t)
        # if typeof(learner) <: ContinuousPolicy
        #     estimate = learner(subagent.submodels.state_encoder(o_t), mask_t)
        # else
            estimate = learner(state_encoder(o_t), mask_t)
        # end

        # println("EST: ", sum(estimate))
        # println("EST: ", size(estimate))
        # println(r_t)
        target = subagent.target.func(subagent.target, o_t, mask_t, r_t, op_t, maskp_t, done_t, T, t, subagent, gamma, device)

        # println("R: ", r_t)
        # println("target: ", size(target))
        # println("target: ", sum(target))
        # println("S", o_t)

        # reg = 0.01f0
        # reg_term = reg*mean(LinearAlgebra.norm, subagent.params)
        # reg_term = reg*IRM(loss, estimate, target)
        reg_term = 0f0

        #ps = get_policy_params(subagent.model, o_t, nothing)
        # reg_term = reg*sum(LinearAlgebra.norm, gradient( () -> sum(target - estimate), Flux.params(ps)).grads[ps])
        # println(reg_term)

        # loss_term = reshape(loss(estimate, target; agg = identity), :)
        loss_term = reshape(loss(estimate, target, agg = identity; state_encoder = nothing), :)
        # println("LOSS TERM:", mean(loss_term))
        # println("REG TERM:", mean(reg_term))
        # println(loss_term)
        loss_total += loss_term .+ reg_term
    # end
    Flux.testmode!(learner.f.f)
    return loss_total / Float32(T)
end

# function minibatch_loss(
#     subagent::Subagent{P},
#     buffer::TransitionReplayBuffer,
#     loss,
#     state_encoder,
#     action_encoder;
#     print = false,
# ) where {P<:RNNPlanner}
#     learner = subagent.model

#     reset_model!(learner)
#     # _, ob, a, p, r, _, obp, done, info  = get_batch(buffer)

#     # ob  = ob |> subagent.device
#     # obp = obp |> subagent.device
#     # # ob, obp = stop_gradient() do
#     # #     state_encoder(ob), state_encoder(obp)
#     # # end

#     # done = done |> subagent.device

#     # mask = stop_gradient() do
#     #     action_encoder(a) |> subagent.device
#     # end

#     # discounted_reward_sum = stop_gradient() do
#     #     nstep_returns(subagent, r, all = true)
#     # end

#     loss_total = []

#     # M = size(r)[3]
#     # T = size(r)[2]



#     episode_inds = stop_gradient() do
#         sample_episodes(buffer)
#     end

#     for ind in episode_inds
#         ep = buffer._episodes[ind]
#         o = stop_gradient() do
#             [exp.o for exp in ep]
#         end

#         r = stop_gradient() do
#             [exp.r for exp in ep]
#         end

#         a = stop_gradient() do
#             [exp.a[1] for exp in ep]
#         end

#         mask = stop_gradient() do
#             action_encoder(a) #|> subagent.device
#         end

#         T = length(o)

#         time_loss = 0.0f0
#         init_memory(learner, reshape(o[1], (:, 1)))
#         for t = 1:T
#             r_est = unroll(learner, mask[:, t])[1]
#             if loss == absolute
#                 time_loss += (r_est - r[t])
#             else
#                 time_loss += (r_est - r[t])^2
#             end
#         end

#         if loss == absolute
#             push!(loss_total, abs(time_loss) / T)
#         else
#             push!(loss_total, time_loss / T)
#         end

#     end

#     return loss_total
#     # for t = 1:T
#     #     o_t = ob[:, t, :]
#     #     if t == 1
#     #         init_memory(learner, o_t)
#     #     end
#     #     a_t = a[:, t, :]
#     #     p_t = reshape(p[:, t, :], :)
#     #     r_t = reshape(discounted_reward_sum[:, t, :],:)
#     #     op_t = obp[:, t, :]
#     #     done_t = reshape(done[:, t, :],:)
#     #     mask_t = mask[:, t, :]

#     #     target = subagent.target.func(subagent, o_t, mask_t, r_t, op_t, done_t, T, t)
#     #     unroll(learner, mask_t)

#     #     loss_total += loss(estimate, target)
#     # end
#     # return loss_total / Float32(M * T)
# end

function dual(A::AbstractAgent, buffer::AbstractBuffer, loss)

end
