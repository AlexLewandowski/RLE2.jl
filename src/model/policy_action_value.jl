abstract type AbstractPolicyActionValue{A} <: AbstractActionValue{A} end

mutable struct PolicyActionValue{A<:AbstractApproximator} <: AbstractPolicyActionValue{A}
    f::A
    p::Any
end

function PolicyActionValue(A)
    embedding_dim = 10
    input_dim = length(get_params_vector(A))
    p = Dense(input_dim, embedding_dim, relu)
    PolicyActionValue{typeof(A)}(A, p)
end

function get_policy_params(Q::PolicyActionValue, s, ps = nothing)
    if isnothing(ps)
        ps = construct_params_vector(Q)
    end
    tail_size = size(s)[2:end]
    stacked_ps = repeat(Q.p(ps), prod(tail_size))
    ps =  reshape(stacked_ps, (:, tail_size...))
    return ps
end

function forward(Q::PolicyActionValue, s; ps = nothing)
    Q.f(cat(s, get_policy_params(Q, s, ps), dims = 1))
end

function forward(Q::PolicyActionValue, s, mask; ps = nothing)
    sum(Q.f(cat(s, get_policy_params(Q, s, ps), dims = 1)) .* mask, dims = 1)
end
