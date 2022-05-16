

function PPOpg(A::AbstractAgent, s, a, p, G, sp, done, mask)
    p_pol = A.π(s)

    T = length(G)

    p_a = reshape(sum(p_pol .* mask, dims = 1), T)
    p_b = reshape(p, T)

    penalty = PPOpenalty(A, s, p_pol)

    loss_val = -sum(log.(p_a) .* G)

    return loss_val + penalty
end

function PPOsquare(A::AbstractAgent, s, a, p, G, sp, done, mask)
    Q = A.π.approximator.f(s)

    T = length(G)

    p_pol = softmax(Q)

    p_a = reshape(sum(p_pol .* mask, dims = 1), T)
    p_b = reshape(p, T)

    Q = reshape(sum(Q .* mask, dims = 1), T)

    penalty = SQpenalty(A, env, s, p_pol)

    val = sum((Q - G) .^ 2)
    return val - penalty
end


function PPOpenalty(A::AbstractAgent, s, p_a)
    p_old = stop_gradient() do
        A.π_old(s)
    end

    penalty = kldivergence(p_old, p_a)
    return penalty
end

function SQpenalty(A::AbstractAgent, s, p_a)
    p_old = stop_gradient() do
        A.π_old(s)
    end
    #penalty =  sum((p_a - p_old).^2)
    penalty = sum(p_a)
end

function get_all_activations(m, x)
    # return 0f0
    l_layers = layers(m)
    activations = l_layers[1](x)
    h = l_layers[1](x)
    L = length(l_layers)
    for l = 2:L
        h = l_layers[l](h)
        activations = vcat(activations, h)
    end
    return activations
end

function reg_td_weights(A::AbstractSubagent, o, op)
    r_ws = reshape(get_all_activations(A.submodels.reward_model, o), :)
    o_ws = reshape(get_all_activations(A, o), :)
    op_ws = reshape(get_all_activations(A, op), :)
    return mean(r_ws + op_ws - 0.99f0*o_ws)
end

function IRM(loss, pred, target)
    M = size(pred)[end]
    w = [1.0f0]
    M = size(pred)[end]
    M2 = Int(M/2)
    pred_1 = pred[:, 1:M2]
    pred_2 = pred[:, M2+1:end]
    target_1 = target[:, 1:M2]
    target_2 = target[:, M2+1:end]
    grads_1 = gradient(Flux.params(w)) do
        mean(loss(w.*pred_1, target_1))
    end
    grads_2 = gradient(Flux.params(w)) do
        mean(loss(w.*pred_2, target_2))
    end
    return sum(grads_1.grads[w].*grads_2.grads[w])
    # return grads_1, grads_2
    # return sum(LinearAlgebra.norm, grads)
end

function grad_2(loss, pred, target)
    grads = gradient(Flux.params(pred)) do
        grads2 = gradient(Flux.params(pred)) do
            loss(pred, target)
        end
        sum(grads2[pred])
    end
    return sum(grads[pred].^2)
end
