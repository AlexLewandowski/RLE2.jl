# mutable struct RNNContinuousPlanner{A<:AbstractApproximator,M} <: AbstractContinuousModel
#     f::A
#     memory::M
#     count::Int64
#     history_window::Int64
#     predict_window::Int64
#     open_loop::Bool
#     freeze::Bool
#     d::Any
#     action_dim::Int64
#     plan::AbstractArray
#     replan::Int64
#     planning_algo::Any
# end

mutable struct RNNPlanner{A<:AbstractApproximator,M} <: AbstractModel{A}
    f::A
    memory::M
    count::Int64
    history_window::Int64
    predict_window::Int64
    open_loop::Bool
    freeze::Bool
    d::Any
    action_dim::Int64
    plan::AbstractArray
    replan::Int64
    planning_algo::Any
end

function RNNPlanner(
    f;
    history_window::Int64 = 0,
    predict_window::Int64 = 0,
    open_loop::Bool = true,
    discrete,
    action_dim,
    replan = 0,
    planning_algo,
)
    mem_layer = f(:state_dynamics).layers[end]
    state = mem_layer.state
    if typeof(mem_layer) <: Recur{C} where {C<:Flux.LSTMCell}
        memory_size = size(state[1])[1]
    elseif typeof(mem_layer) <: Recur{C} where {C<:Flux.GRUCell}
        memory_size = size(state[1])[1]
    elseif typeof(mem_layer) <: Recur{C} where {C<:Flux.RNNCell}
        memory_size = size(state)[1]
    end
    memory = randn(Float32, (memory_size, 1))
    if typeof(mem_layer.cell.Wh) <: Array
    else
        memory = gpu(memory)
    end
    return RNNPlanner{typeof(f),typeof(memory)}(
        f,
        memory,
        0,
        history_window,
        predict_window,
        open_loop,
        false,
        ϵgreedy,
        action_dim,
        [],
        replan,
        planning_algo,
    )
end

##
## Model-specific overrides
##

function num_actions(π::RNNPlanner)
    π.action_dim
end

function sample(π::RNNPlanner, s; greedy = false)
    N = size(π.plan)[end]
    if N == 0 || N == π.replan
        as_env, as_m = π.planning_algo[1](π, s)
        model_q = eval_plan(π, s, as_m)
        π.planning_algo[2][1] += model_q[1]
        π.plan = as_env
    end
    a = π.plan[1, 1]
    π.plan = π.plan[:, 2:end]
    return a, 1.0f0
end

function reset_model!(m::RNNPlanner)
    for head in keys(m.f.f)
        Flux.reset!(m.f(head))
    end
    stop_gradient() do
        m.planning_algo[2][1] = 0.0f0
    end
    m.count = 0
end

(model::RNNPlanner)(s) = model.f.reward_head(get_memory(model))

function init_memory(model::RNNPlanner, s)
    model.memory = init_memory(model.f, s)
    return model.memory
end

function get_memory(model::RNNPlanner)
    return model.memory
end

function unroll(model::RNNPlanner, a)
    a_hat = model.f(a, :action_encoder)
    h = model.f(a_hat, :state_dynamics)
    # println(size(h))
    # println(size(model.memory))
    # model.memory = h
    r = model.f(h,:reward_head)
    return r
end

function random_plan(π, s, n = nothing)
    if isnothing(n)
        n = Int64(min(BigInt(2)^(π.predict_window), 64)) #TODO
    end
    n = 1
    max_q = -Inf
    best_as = nothing
    as_env, as = gen_action_seq(π, n)
    qs = eval_plan(π, s, as)
    # qs_env = eval_plan(env, s, as)
    # println(maximum(qs_env))
    # println(minimum(qs_env))

    best_plan = argmax(qs)
    worst_plan = argmin(qs)

    return as_env[:, :, best_plan], as[:, :, best_plan]
end

function gen_action_seq(π::RNNPlanner)
    L = π.predict_window
    a = rand(π)
    T = typeof(a)

    as = zeros(T, L)
    as[1] = a
    for i = 2:L
        a = rand(π)
        as[i] = a
    end
    return as, get_mask(π, reshape(as, (1, size(as)...)))
end

function gen_action_seq(π::RNNPlanner, n::Int64)
    a_env, a = gen_action_seq(π)
    T = typeof(a[1])
    T_env = typeof(a_env[1])
    as = zeros(T, (size(a)..., n))
    as_env = zeros(T_env, (1, size(a_env)..., n))
    as_env[1, :, 1] = a_env
    as[:, :, 1] = a
    if n > 1
        for i = 2:n
            a_env, a = gen_action_seq(π)
            as_env[:, :, i] = a_env
            as[:, :, i] = a
        end
    end
    return as_env, as
end


# TODO: all-action with reward model
# function all_action_value(model::AbstractRNNPlanner, s, π)
#     T = model.predict_window

#     n = num_actions(π)

#     if length(size(s)) == 1
#         s = hcat([copy(s) for i = 1:n]...)
#     end

#     reset_model!(model)
#     reset_model!(π)

#     as = get_mask(π, reshape([1, 2], (1, 2)))

#     q = zeros(Float32, n)
#     init_memory(model, s)

#     for i = 1:T
#         if i == 1
#             a = as
#         else
#             a = reshape(rand([1, 2], 2), (1, 2))
#             a = get_mask(π, a)
#         end
#         q_, s = model(s, a)

#         q_ = action_value(model, s)
#         q_ = reshape(sum(q_ .* a, dims = 1), :)

#         q += copy(q_)
#     end
#     return q
# end

function eval_plan(model::RNNPlanner, s::Any, f::Chain)
    T = model.predict_window + 1

    if length(size(s)) == 1
        s = hcat([copy(s) for i = 1:1]...)
    end

    n = size(s)[end]

    reset_model!(model)
    memory = model.f(:state_dynamics)[1].state
    println("INIT", sum(memory))
    q_finitehorizon = zeros(Float32, n)
    init_memory(model, s)

    for i = 1:T
        memory = model.f(:state_dynamics)[1].state
        println(string(i)*": ", sum(memory))
        a = f(memory)
        println("a: ", a)
        r = unroll(model, a)
        # println("r: ", r)

        q_finitehorizon += copy(r)
    end
    return q_finitehorizon
end

function eval_plan(
    model::RNNPlanner,
    s::Any,
    as::Union{Array{Type,2},CuArray{Float32,2}},
) where {Type<:Union{AbstractFloat,Int}}
    @assert length(size(as)) > 1
    @assert length(size(as)) < 4
    if length(size(as)) == 2
        as = reshape(as, (size(as)..., 1))
    end
    T = size(as)[2]
    n = size(as)[end]

    if length(size(s)) == 1
        s = hcat([copy(s) for i = 1:n]...)
    end

    reset_model!(model)
    q = zeros(Float32, n)
    init_memory(model, s)

    for i = 1:T
        a = as[:, i, :]
        r = unroll(model, a)
        q += copy(r)
    end
    return q
end

# function trajectory_optimization(model::AbstractRNNPlanner, s; num_updates = 10, lr = 0.1)
#     opt = Flux.AMSGrad(lr)
#     T = model.predict_window
#     N = num_actions(model)
#     a = randn(Float32, (N, T))
#     # a = reshape(vcat([[x, 1-x] for x in a]...), (N,T))
#     for i = 1:num_updates
#         # println(eval_plan(model, s, a))
#         grads = gradient(params(a)) do
#             # act = softmax(a)
#             act = get_mask(model, a)
#             z = eval_plan(model, s, act)
#             # println(z)
#             return z[1]
#         end
#         Flux.update!(opt, params(a), grads)
#     end
#     a = to_env_action(model, a)
#     # a = [x[1] for x in argmax(a, dims = 1)]
#     # a = 2*tanh.(a)
#     return a, get_mask(model, a)
# end


function trajectory_optimization(model::RNNPlanner, s; num_updates = 10, lr = 0.001)
    opt = Flux.AMSGrad(lr)
    T = model.predict_window
    N = num_actions(model)
    policy = Chain(Dense(32, 32, relu), Dense(32, 2), softmax) #|> gpu
    # a = reshape(vcat([[x, 1-x] for x in a]...), (N,T))
    println("Before training", eval_plan(model, s, policy))

    for i = 1:num_updates
        # println(eval_plan(model, s, a))
        grads = gradient(Flux.params(policy)) do
            # act = softmax(a)
            z = -eval_plan(model, s, policy)
            println(z)
            return sum(z)
        end
        Flux.update!(opt, Flux.params(policy), grads)
    end
    # a = to_env_action(model, a)
    # a = [x[1] for x in argmax(a, dims = 1)]
    # a = 2*tanh.(a)
    println("After training", eval_plan(model, s, policy))
    return policy
end
