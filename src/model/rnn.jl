mutable struct RNNActionValue{A<:AbstractApproximator} <: AbstractModel{A}
    f::A
    count::Int64
    history_window::Int64
    predict_window::Int64
    open_loop::Bool
    freeze::Bool
    d::Any
end

# mutable struct RNNContinuousPolicy{A<:AbstractApproximator} <: AbstractContinuousModel
#     f::A
#     count::Int64
#     history_window::Int64
#     predict_window::Int64
#     open_loop::Bool
#     freeze::Bool
#     d
# end

function RNNActionValue(
    A,
    history_window::Int64 = 0,
    predict_window::Int64 = 0,
    open_loop::Bool = true,
)
    #@assert any(x -> typeof(x) <: Recur, A.f.layers)
    return RNNActionValue{typeof(A)}(
        A,
        0,
        history_window,
        predict_window,
        open_loop,
        false,
        ϵgreedy,
    )
end

function unroll_new(π::RNNActionValue, s)
    x = copy(s)
    layers = π.f.layers
    for layer in layers
        if typeof(layer) <: Recur
            x, h = layer.cell(layer.init, x)
        else
            x = layer(x)
        end
    end
    return x
end

# function unroll(π::RNNDiscretePolicy, s)
#     if π.freeze
#         x = copy(s)
#         layers = π.approximator.f.layers
#         for layer in layers
#             if typeof(layer) <: Recur
#                 x, h = layer.cell(layer.state, x)
#             else
#                 x = layer(x)
#             end
#         end
#         return x
#     else
#         π.count += 1
#         return π.f(s)
#     end
# end


function unroll(π::RNNActionValue, s)
    i = 0
    x = copy(s)
    sp = copy(s)
    layers = π.f.layers
    for layer in layers
        if typeof(layer) <: Recur
            if i == 0
                i = 1
            end
            if π.freeze
                x, _ = layer.cell(layer.state, x)
            end
        end
        if i == 0
            if length(size(s)) > 1
                batch_dim = size(s)[2]
                x = zeros(Float32, (size(layer.W)[1]..., batch_dim))
            else
                x = zeros(Float32, size(layer.W)[1])
            end
        else
            sp = copy(x)
            x = layer(x)
        end
    end
    if !π.freeze
        π.count += 1
    end
    return x#, sp
end

function (π::RNNActionValue)(s)
    if π.count == 0
        q = init_memory(π, s)
        π.count += 1
    elseif π.count < π.history_window
        q = unroll(π, s)
    elseif π.count < π.history_window + π.predict_window
        if π.open_loop
            z = zeros(size(s))
            q = unroll(π, z)
        else
            q = unroll(π, s)
        end
    else
        error("Something went wrong with rnn count")
    end

    if π.count == π.predict_window + π.history_window
        reset_model!(π)
    end
    # q_new = unroll_new(π, s)
    # q_max = reshape(maximum(q, dims = 1), :)
    # q_new_max = reshape(maximum(q_new, dims = 1), :)
    # if q_new_max > q_max
    # # if sum(q_new) > sum(q)
    #     q = q_new
    # end

    return q
end

function sample(π::RNNActionValue, s; greedy = false)
    q = π(s)

    # # TODO RESET MECHANISM

    # q_new = unroll_new(π, s)
    # q_max = reshape(maximum(q, dims = 1), :)
    # q_new_max = reshape(maximum(q_new, dims = 1), :)
    # if q_new_max > q_max
    # # if sum(q_new) > sum(q)
    #     reset_model!(π)
    #     q = π(s)
    # end
    if greedy == true
        p = zeros(size(q))
        action = argmax(q)
        p[action] = 1.0f0
    else
        p = get_prob(π, q)
    end
    a = StatsBase.sample(StatsBase.Weights(p))
    p_a = p[a]
    return a, p_a
end

function reset_model!(π::RNNActionValue)
    Flux.reset!(π.f)
    π.count = 0
end

function init_memory(π::RNNActionValue, s)
    init_memory(π.f, s)
end

function init_memory(f::NeuralNetwork, s)
    i = 0
    if length(size(s)) == 1
        s = reshape(s, (:,1))
    end

    memory = f(s, :state_encoder)
    layer = f(:state_dynamics).layers[1]
    if typeof(layer) <: Recur{C} where {C<:Flux.LSTMCell}
        L = Int(size(memory)[1] / 2)
        layer.state = (memory[1:L, :], memory[L+1:end, :])
        out_mem = layer.state[1]
    elseif typeof(layer) <: Recur{C} where {C<:Flux.RNNCell}
        println(size(layer.state))
        println(size(memory))
        layer.state = memory
        out_mem = memory
    else
        return 0
    end

    return out_mem
end
