function v_target(target, o_t, mask, r, op_t, maskp, done, T, t, subagent, gamma, device)
    M = size(r)[end]
    if !isnothing(target.sp_network)
        # op_t = subagent.target_submodels.state_encoder(op_t, output_func = target.sp_network)
        op_t = subagent.target_submodels.state_encoder(op_t)
        q_sp = reshape(target.sp_network(op_t, dropgrad = true), size(r))
    else
        q_sp = zeros(Float32, size(r)) |> device
    end
    if !isnothing(target.s_network)
        q_s = reshape(target.s_network(o_t), size(r))
    else
        q_s = zeros(Float32, size(r)) |> device
    end
    target = r + gamma^(T - t + 1) * q_sp .* (1 .- done) - q_s
    return target
end

function act_target(target, o_t, mask, r, op_t, maskp, done, T, t, subagent, gamma, device)
    return mask
end

function s_target(target, o_t, mask, r, op_t, maskp, done, T, t, subagent, gamma, device)
    o_t = stop_gradient() do
            subagent.submodels.state_encoder(o_t)
    end
    return o_t
end

function max_target(target, o_t, mask, r, op_t, maskp, done, T, t, subagent, gamma, device)
    M = size(r)[end]
    if !isnothing(target.sp_network)
        op_t = stop_gradient() do
            subagent.target_submodels.state_encoder(op_t)
        end

        q_sp = reshape(
            maximum((subagent.target.sp_network(op_t, dropgrad = true)), dims = 1),
            size(r),
        )
    else
        if device == Flux.gpu
            q_sp = CUDA.zeros(Float32, size(r))
        else
            q_sp = zeros(Float32, size(r))
        end
    end
    if !isnothing(target.s_network)
        q_s = reshape(sum(target.s_network(o_t) .* mask, dims = 1), size(r))
    else
        q_s = zeros(Float32, size(r)) |> subagent.device
    end
    target = r + subagent.gamma^(T - t + 1) * q_sp .* (1 .- done) - q_s
    return target
end

function q_target(target, o_t, mask, r, op_t, maskp, done, T, t, device, gamma)
    M = size(r)[end]
    if !isnothing(target.sp_network)
        q_sp = reshape(maximum(target.sp_network(op_t), dims = 1), :)
    else
        q_sp = zeros(Float32, M) |> device
    end
    if !isnothing(target.s_network)
        q_s = reshape(sum(target.s_network(o_t) .* mask, dims = 1), :)
    else
        q_s = zeros(Float32, M) |> device
    end
    target = r + gamma^(T - t + 1) * q_sp .* (1 .- done) - q_s
    return target
end

function qr_target(target, o_t, mask, r, op_t, maskp, done, T, t, subagent, gamma, device)
    M = size(r)[end]
    op_t = subagent.target_submodels.state_encoder(op_t)
    q_sp = Zygote.dropgrad(target.sp_network(op_t))
    q_sp_collapsed = Zygote.dropgrad(mean(q_sp, dims = 1)) |> Flux.cpu

    a_max = Zygote.dropgrad(stop_gradient() do
        [a_tup.I[2][1] for a_tup in argmax(q_sp_collapsed, dims = 2)]
    end)
    maskp = Zygote.dropgrad(subagent.submodels.action_encoder(a_max)[:, 1, :]) |> subagent.device

    q_sp = Zygote.dropgrad(sum(q_sp .* reshape(maskp, (1, size(maskp)...)), dims = 2))

    done = reshape(done, (1, size(done)...))
    r = reshape(r, (1, size(r)...))

    return r .+ gamma^(T - t + 1) * q_sp .* (1 .- done)
end

function td3_target(
    target,
    o_t,
    mask,
    r,
    op_t,
    maskp,
    done,
    T,
    t,
    subagent,
    gamma,
    device,
)
    M = size(r)[end]
    if !isnothing(target.sp_network)

        if !isnothing(subagent.submodels.policy)
            sp = subagent.target_submodels.state_encoder(op_t)
            ap = subagent.target_submodels.policy(sp)

            eps = clamp.(0.001f0 .* randn(Float32, size(ap)), -0.01f0, 0.01f0) |> subagent.device
            ap = clamp.(ap + eps, 0.0f0, 1.0f0)

            # eps = rand(Float32, size(ap)) .+ 0.5f0 |> subagent.device
            # eps = clamp.(0.5f0.*randn(Float32, size(ap)) .+ 1f0, 0.5f0, 2f0) |> subagent.device
            # ap = clamp.(ap.*eps, 0.0f0, 1.0f0)

            q_sp1 = reshape(subagent.target.sp_network[1](sp, ap, dropgrad = true), size(r))
            q_sp2 = reshape(subagent.target.sp_network[2](sp, ap, dropgrad = true), size(r))

            q_sp = minimum(cat([q_sp1, q_sp2]..., dims = 1), dims = 1)
        else
            ap = maskp
        end
    else
        if device == Flux.gpu
            q_sp = CUDA.zeros(Float32, size(r))
        else
            q_sp = zeros(Float32, size(r))
        end
    end
    if !isnothing(target.s_network)
        q_s = reshape(sum(target.s_network(o_t) .* mask, dims = 1), size(r))
    else
        q_s = zeros(Float32, size(r)) |> subagent.device
    end
    gamma = subagent.gamma
    target = r + gamma^(T - t + 1) * q_sp .* (1 .- done) - q_s
    return target
end

function doubledqn_target(
    target,
    o_t,
    mask,
    r,
    op_t,
    maskp,
    done,
    T,
    t,
    subagent,
    gamma,
    device,
)
    M = size(r)[end]
    if !isnothing(target.sp_network)
        op_t = subagent.target_submodels.state_encoder(op_t)
        q_sp = subagent.target.sp_network(op_t, dropgrad = true)
        a_sp = subagent.model(op_t, dropgrad = true)
        a_max = argmax(a_sp |> Flux.cpu, dims = 1)
        a_max = [a_[1] for a_ in a_max]
        max_maskp = stop_gradient() do
             subagent.submodels.action_encoder(a_max) |> subagent.device
        end
        q_sp = sum(q_sp.*max_maskp, dims = 1)
    else
        if device == Flux.gpu
            q_sp = CUDA.zeros(Float32, size(r))
        else
            q_sp = zeros(Float32, size(r))
        end
    end
    if !isnothing(target.s_network)
        q_s = reshape(sum(target.s_network(o_t) .* mask, dims = 1), size(r))
    else
        q_s = zeros(Float32, size(r)) |> subagent.device
    end
    gamma = subagent.gamma
    target = r + gamma^(T - t + 1) * q_sp .* (1 .- done) - q_s
    return target
end
