
mutable struct PersistActionValue{A<:AbstractApproximator} <: AbstractModel{A}
    f::A
    last_action::Int64
    count::Int64
    predict_window::Int64
    d::Any
end

# mutable struct PersistContinuousPolicy{A<:AbstractApproximator} <: AbstractContinuousModel
#     f::A
#     last_action::Int64
#     count::Int64
#     predict_window::Int64
#     d
# end

# AbstractPersistPolicy = Union{PersistContinuousPolicy,PersistDiscretePolicy}

function PersistActionValue(A; predict_window::Int64 = 0, discrete)
    PersistActionValue{typeof(A)}(A, 0, 0, predict_window, softmax)
end

function sample(Q::PersistActionValue, s; greedy = false)
    if (greedy == true) && Q.count == 0
        q = Q(s)
        p = zeros(size(q))
        a = argmax(q)
        Q.last_action = copy(a)
        Q.count += Q.predict_window == 0 ? 0 : 1
        p[a] = 1.0f0
        p_a = p[a]
    elseif Q.count == 0
        p = get_prob(Q, Q(s))
        a = StatsBase.sample(StatsBase.Weights(p))
        Q.last_action = copy(a)
        Q.count += Q.predict_window == 0 ? 0 : 1
        p_a = p[a]
    elseif Q.count == Q.predict_window
        a = Q.last_action
        p_a = 1.0f0
        reset_model!(Q)
    elseif Q.count < Q.predict_window
        a = Q.last_action
        p_a = 1.0f0
        Q.count += 1
    else
        error("Something went wrong with persist count: ", Q.count, " ", Q.predict_window)
    end
    return a, p_a
end

function reset_model!(π::PersistActionValue)
    π.count = 0
end
