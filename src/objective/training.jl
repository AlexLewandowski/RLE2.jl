include("objective_utils.jl")
include("classic_objectives.jl")
include("regularizers.jl")

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
    num_grad_steps = nothing,
    training = true,
    batch_size = nothing,
    reg = false,
    resample = true,
)

    if num_grad_steps === nothing
        num_grad_steps = subagent.num_grad_steps
    end

    for _ = 1:num_grad_steps
        if subagent.target.sp_network == nothing #TODO handling of bootstrap / MC
            bootstrap = false
        else
            bootstrap = true
        end

        if resample
            StatsBase.sample(buffer, batch_size = batch_size, bootstrap = bootstrap)
        end

        if mod(subagent.update_count, subagent.train_freq) == 0
            if typeof(subagent.model.f) <: Tabular
                tabular_learning(subagent, buffer)
            else
                grads = get_grads(subagent, buffer, subagent.loss, state_encoder, action_encoder, reg)
                update!(subagent, grads)
            end
        end
        subagent.update_count += 1
        callback(subagent)
    end
end

function tabular_learning(subagent::AbstractSubagent, exp::Experience)
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

function tabular_learning(subagent::AbstractSubagent, buffer)
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
