mutable struct ContextualBandit{A,T,ACT,R<:AbstractRNG} <: AbstractEnv
    s_d::Int
    action_space::Base.OneTo{Int64}
    observation_space::Space{Array{Interval{:closed,:closed,Float32},1}}
    state_space::Base.OneTo{Int64}
    obs::AbstractArray{T,1}
    state::Int64
    action::ACT
    done::Bool
    t::Int
    reward::T
    rng::R
    n_actions::Int
    n_states::Int
    xs::AbstractArray{Array{Float32,1},1}
    ys::AbstractArray{Int64,1}
end

function ContextualBandit(
    dataset,
    T = Float32;
    corruption_rate = 0.0f0, # TODO corruption label
    n_actions = 10,
    test = false,
    rng = Random.GLOBAL_RNG,
    kwargs...,
)
    if dataset == "MNIST"
        if test
            xs, ys = MLDatasets.MNIST.testdata()
        else
            xs, ys = MLDatasets.MNIST.traindata()
        end
        L = length(ys)
        xs_flat = float.(reshape(xs, (:, L)))
        xs = [xs_flat[:, i] for i = 1:L]
    else
        error("Invalid dataset for ContextualBandit")
    end

    s_d = size(xs)[1]
    n_states = length(xs)
    action_space = Base.OneTo(n_actions)
    obs_space = Space([0.0f0..1.0f0 for i = 1:s_d])
    state_space = Base.OneTo(n_actions)

    for i = 1:L
        if rand(rng) < corruption_rate
            label = rand(rng, action_space) - 1
            ys[i] = label
        end
    end

    inds = [collect(1:L)[ys.==i-1] for i = 1:n_actions]

    A = rand(rng, action_space)
    R = typeof(rng)

    CBandit = ContextualBandit{Int,Float32,Int,R}(
        s_d::Int,
        action_space,
        obs_space,
        state_space,
        zeros(T, s_d)::AbstractArray,
        0::Int,
        rand(action_space)::Int,
        false::Bool,
        0::Int,
        T(0)::T,
        rng,
        n_actions::Int,
        n_states::Int,
        xs,
        ys,
    )
    reset!(CBandit)
    CBandit
end

Base.show(io::IO, t::MIME"text/plain", env::ContextualBandit) = begin
    println()
    println("---------------------------")
    name = "ContextualBandit"
    L = length(name)
    println(name)
    println(join(["-" for i = 1:L+1]))

    print(io, "  env.n_actions: ")
    println(io, env.n_actions)

    print(io, "  env.corruption_rate: ")
    println(io, env.corruption_rate)
    println("---------------------------")
end

function test_env(env::ContextualBandit)
    return ContextualBandit(
        "MNIST_TEST",
        n_actions = env.n_actions,
        corruption_rate = env.corruption_rate,
    )
end

function reset!(env::ContextualBandit)
    ind = rand(env.rng, 1:env.n_states)
    env.state = env.ys[ind] + 1
    env.obs = env.xs[ind]
    env.t = 0
    env.reward = 0.0f0
    env.done = false
    nothing
end

function get_obs(env::ContextualBandit)
    return env.obs
end

function get_state(env::ContextualBandit)
    return [env.state]
end

function get_reward(env::ContextualBandit)
    return env.reward
end

function get_actions(env::ContextualBandit)
    return env.action_space
end

function get_terminal(env::ContextualBandit)
    return env.done
end

function Random.seed!(env::ContextualBandit, seed) end

function (env::ContextualBandit)(a::Int)
    @assert a in env.action_space
    env.action = a
    _step!(env, a)
end

function _step!(env::ContextualBandit, a)
    if a == env.state
        env.reward = 1.0f0
    else
        env.reward = 0.0f0
    end
    # env.reward = -(a - env.ind)^2
    env.t += 1
    env.done = true
    nothing
end
