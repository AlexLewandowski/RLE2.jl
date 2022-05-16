import Flux: Dense, Chain, relu, softmax, Recur, logsoftmax
import Distributions: Normal, pdf, logpdf, Uniform

"""
A policy is a functional object to generate an action given a state.
Currently, the policy does this by implementing three functions.

- π(s) generates a score vector
 - discrete: action values, continuous
 - continuous: parameters for distribution
- get_prob(π, π(s)) gets a probability distributionb for π at s
  - Discrete: softmax over score π(s)
  - Continuous: return the parameterized distribution
- sample(π, s; greedy) samples an action from π at s

The reason these are separate is because we may have stateful policies.
As such, we would like to separate the interaction (sampling) from
optimization which just calls π(s).

sample(⋅) usually calls π then get_prob and then samples.
"""

abstract type AbstractPolicy{A} <: AbstractModel{A} end

mutable struct DiscretePolicy{A<:AbstractApproximator} <: AbstractPolicy{A}
    f::A
    d
end

mutable struct ContinuousPolicy{A<:AbstractApproximator} <: AbstractPolicy{A}
    f::A
    d
end

(model::AbstractPolicy)(s) = model.f(s)
(model::AbstractPolicy)(s, a) = model(s)

function sample(π::DiscretePolicy, s; rng, greedy = false)
    q = π(s)
    if greedy == true
        a = argmax(q)
        p_a = 1.0f0
    else
        p = get_prob(π, q)
        a = StatsBase.sample(rng, StatsBase.Weights(p))
        p_a = p[a]
    end
    return a, p_a
end

function sample(m::ContinuousPolicy, s; rng, greedy = false)
    out = m(s) |> Flux.cpu
    if greedy == true
        # a = clamp.(m(s)[1], -1.0f0, 1.0f0)
        a = clamp.(out, 0.0f0, 1.0f0)
    else
        eps = 0.0001f0*randn(Float32, size(out))
        a = clamp.(out + eps, 0f0, 1f0)

        epsilon = 0.1f0
        if rand(rng) > epsilon
            a = a
        else
            rand_num = rand(1:2)
            if rand_num == 1
                a = a.*10f0
            else
                a = a./10f0
            end
            a = clamp.(a, 0f0, 1f0)
        end

        #eps = clamp.(0.01f0 .* randn(Float32, size(out)), -0.1f0, 0.1f0)
        # eps = clamp.(0.1f0.*randn(Float32, size(out)) .+ 1f0, 0.5f0, 2f0)
        # a = clamp.(a.*eps, 0f0, 1f0)
    end
    a = a |> Flux.cpu
    if length(a) == 1
        a = a[1]
    end

    p_a = 1f0
    return a, p_a
end

function forward(model::DiscretePolicy, s)
    model(s)
end

function forward(model::DiscretePolicy, s, mask)
    model(s)
end

# function sample(π::AbstractContinuousModel, s; greedy = false)
#     if greedy
#         a = π(s)
#         p = ones(Float32, size(π(s)))
#     else
#         a = (rand.(π.d, 1) + π(s))
#         p = get_prob(π, a, s)
#     end
#     if length(p) == 1
#         a = a
#         p = p[1][1]
#     else
#         a = [x[1] for x in a]
#         p = [x[1] for x in p]
#     end
#     return Float32.(a), Float32.(p)
# end

function reset_model!(π::Union{AbstractModel,Symbol}) end

get_prob(π::DiscretePolicy, q) = π.d(q)
get_prob(π::DiscretePolicy, q, a) = π.d(q)[a]

function get_logprob(π::DiscretePolicy, s)
    Q = π(s)
    # P = Q .- maximum(Q)
    # Z = log(sum(exp.(P)))
    return logsoftmax(Q)
    # return P .- Z
end

get_logprob(π::DiscretePolicy, s::AbstractArray{Float32,1}, a::Int64) = get_logprob(π, s)[a]
get_logprob(π::DiscretePolicy, s::AbstractArray{Float32,2}, as::AbstractArray{Int64,2}) = begin
    @assert length(as) == size(s)[end]
    P = get_logprob(π, s)
    mask = get_mask(π, as)
    return sum(P .* mask, dims = 1)
end

# function get_prob(π::AbstractContinuousModel, s, a)
#     #p = pdf.(π.d, q .- π(s))
#     p = 1f0 / 0.1f0 * exp.(-0.5f0 * (a .- π(s)) .^ 2 / 0.1f0^2)
#     @assert -Inf < p[1] < Inf
#     return Float32.(p)
# end

# function get_logprob(π::AbstractContinuousModel, s, a)
#     #p = logpdf.(π.d, q .- π(s))
#     p = -0.5f0 * (a .- π(s)) .^ 2 / 0.1f0^2
#     return p
# end


function Base.rand(π::DiscretePolicy)
    # na = num_actions(π)
    return rand(1:na)
end

# function Base.rand(π::AbstractContinuousModel)
#     # TODO assumes range and dimensionality of action size..
#     return Float32.(rand(Uniform(-2f0, 2f0)))
# end


###
###


function Policy(A; discrete)
    if discrete
        DiscretePolicy{typeof(A)}(A, ϵgreedy)
    else
        ContinuousPolicy{typeof(A)}(A, Normal(0.0f0, 0.1f0))
    end
end
