function squared(estimate, target; print = false, agg = identity, kwargs...)
    if print == true
        println("EST: ", estimate[1], size(estimate))
        println("TAR: ", target[1], size(target))
    end
    agg((estimate - target) .^ 2)
end

function absolute(estimate, target; print = false, agg = identity, kwargs...)
    if print == true
        println("EST: ", estimate[1])
        println("TAR: ", target[1])
    end
    agg(abs.(estimate - target))
end

function policygradient(log_action_prob, value, print = false, kwargs...)
    if print == true
        println("EST: ", log_action_prob[1])
        println("TAR: ", value[1])
    end
    -log_action_prob .* value
end

function quantile_huber_loss(estimate, target; κ=1.0f0, agg = identity, kwargs...)
    if length(size(estimate)) == length(size(target)) == 3
        @assert size(estimate)[2] == size(target)[2] == 1
        estimate = dropdims(estimate, dims = 2)
        target = dropdims(target, dims = 2)
    end

    N, B = size(estimate)
    Δ = reshape(target, N, B) .- reshape(estimate, N, B)
    huber_loss = Flux.Losses.huber_loss(target, estimate; δ = κ, agg = identity)

    if typeof(estimate) <: CUDA.CuArray
        cum_prob = range(0.5f0 / N; length=N, step=1.0f0 / N)
    else
        cum_prob = CUDA.range(0.5f0 / N; length=N, step=1.0f0 / N)
    end

    ro = Zygote.dropgrad(abs.(cum_prob .- (Δ .< 0)))
    loss = ro .*(estimate-target).^2
    # loss = ro .* huber_loss
    reshape(sum(loss, dims = 1), :)
end
