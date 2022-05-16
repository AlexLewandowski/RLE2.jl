abstract type AbstractActionValue{A} <: AbstractModel{A} end
abstract type AbstractDiscreteActionValue{A} <: AbstractActionValue{A} end
abstract type AbstractContinuousActionValue{A} <: AbstractActionValue{A} end

mutable struct ContinuousActionValue{A<:AbstractApproximator} <: AbstractContinuousActionValue{A}
    f::A
    ϵ::Any
end

function ContinuousActionValue(A; ϵ = 0.9)
    ContinuousActionValue{typeof(A)}(A, ϵ)
end

function forward(model::AbstractContinuousActionValue, s; kwargs...)
    error()
end

function forward(model::AbstractContinuousActionValue, s, a; kwargs...)
    input = cat(s,a, dims = 1)
    model.f(input; kwargs...)
end

mutable struct ActionValue{A<:AbstractApproximator} <: AbstractDiscreteActionValue{A}
    f::A
    ϵ::Any
end

function ActionValue(A; ϵ = 0.99)
    ActionValue{typeof(A)}(A, ϵ)
end

function sample(m::AbstractDiscreteActionValue, s; rng, greedy = true)
    Q = reshape(m(s), :)
    if greedy == true
        a = argmax(Q)
        p_a = 1.0f0
    else
        num_actions = size(Q)[1]
        # epsilon = 0.1f0
        # if rand(rng) < epsilon
        epsilon =  m.ϵ
        if rand(rng) < epsilon
            Q = Q |> Flux.cpu
            max_Q, _ = findmax(Q)
            if any(isnan.(Q))
                a = rand(rng, collect(1:num_actions))
            else
                a = rand(rng, collect(1:num_actions)[Q.==max_Q])
            end
        else
            a = rand(rng, 1:num_actions)
        end
        if a == argmax(Q)
            p_a = epsilon + (1 - epsilon) / num_actions
        else
            p_a = epsilon / num_actions
        end

    end
    return a, p_a
end

mutable struct QRActionValue{A<:AbstractApproximator} <: AbstractDiscreteActionValue{A}
    f::A
    N::Int64 #Number of quantiles
end

function QRActionValue(A; N, ϵ = 0.9)
    QRActionValue{typeof(A)}(A, N)
end

function QR_Q(Q::QRActionValue, s)
    reshape(Q.f(s), (Q.N, :, length(size(s)) == 1 ? 1 : size(s)[end]))
end

function QR_Q(Q::QRActionValue, s, mask)
    dropdims(sum(q(s) .* reshape(mask, (1, size(mask)...)), dims = 2), dims = 2)
end

function forward(model::QRActionValue, s)
    reshape(model.f(s), (model.N, :, size(s)[2:end]...))
end

function forward(model::QRActionValue, s, mask)
    sum(forward(model, s) .* reshape(mask, (1, size(mask)...)), dims = 2)
end

function sample(m::QRActionValue, s; rng, greedy = true)
    Q = dropdims(mean(m(s), dims = 1), dims = 1)[:] |> Flux.cpu
    if greedy == true
        a = argmax(Q)
        p_a = 1.0f0
    else
        num_actions = size(Q)[1]
        epsilon = 0.1f0
        if rand(rng) > epsilon
            max_Q, _ = findmax(Q)
            a = rand(rng, collect(1:num_actions)[Q.==max_Q])
        else
            a = rand(rng, 1:num_actions)
        end
        if a == argmax(Q)
            p_a = 1 - epsilon + epsilon / num_actions
        else
            p_a = (1 - epsilon) / num_actions
        end
    end
    return a, p_a
end
