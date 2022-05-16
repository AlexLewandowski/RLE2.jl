import LinearAlgebra: transpose, I
import MatrixEquations: ared, lyapd

struct LQRParams{T}
    A::AbstractArray{T,2}
    B::AbstractArray{T,2}
    S::AbstractArray{T,2}
    R::Union{Array{T,2},Float32}
end

mutable struct LQREnv{A,T,ACT,R<:AbstractRNG} <: AbstractEnv
    params::LQRParams{T}
    a_d::Int
    s_d::Int
    action_space::A
    observation_space::MultiContinuousSpace
    state::AbstractArray{T,1}
    action::ACT
    done::Bool
    t::Int
    reward::T
    gamma::T
    rng::R
    max_steps::Int
    n_actions::Int
    bound::Float64
end

function LQREnv(
    A = [[1, 0] [1, 1]],
    B = transpose([0 1]),
    S = [[1, 0] [0, 0]],
    R = ones(1, 1);
    gamma = 1.0,
    T = Float32,
    n_actions = 0,
    max_steps = 10,
    bound = Inf,
)
    s_d = size(A, 2)
    a_d = size(B, 2)

    @assert size(A) == (s_d, s_d)
    @assert size(B) == (s_d, a_d)
    @assert size(S) == (s_d, s_d)
    @assert size(R) == (a_d, a_d)

    params = LQRParams(T.(A), T.(B), T.(S), T.(R))

    if n_actions == 0
        action_space = MultiContinuousSpace([-Inf for i = 1:a_d], [Inf for i = 1:a_d])
    elseif n_actions > 0
        @assert bound < Inf
        action_space = Base.OneTo(n_actions)
    end
    obs_space = MultiContinuousSpace([-Inf for i = 1:s_d], [Inf for i = 1:s_d])

    lqr = LQREnv(
        params,
        a_d,
        s_d,
        action_space,
        obs_space,
        zeros(T, s_d),
        rand(action_space),
        false,
        0,
        T(0),
        T(gamma),
        GLOBAL_RNG,
        max_steps,
        n_actions,
        bound,
    )
    reset!(lqr)
    lqr
end

function reset!(env::LQREnv)
    #env.state = T(0.1) * rand(env.rng, T, env.s_d) .- T(0.05) #TODO LQR Randomize reset
    #env.action = T(0.1) * rand(env.rng, T, env.s_d) .- T(0.05)
    #env.state = zeros(T, env.s_d)
    #env.action = zeros(T, env.a_d)
    env.state = [-1, 0]
    env.t = 0
    env.reward = 0.0f0
    env.done = false
    nothing
end

function Random.seed!(env::LQREnv, seed) end

function get_obs(env::LQREnv)
    return env.state
end

function get_state(env::LQREnv)
    return env.state
end

function get_reward(env::LQREnv)
    return env.reward
end

function get_actions(env::LQREnv)
    return env.action_space
end

function get_terminal(env::LQREnv)
    return env.done
end

function (env::LQREnv{<:MultiContinuousSpace})(a::AbstractArray{T,1}) where {T<:Number}
    env.action = a
    _step!(env, a)
end

function (env::LQREnv{<:Base.OneTo})(a::Int)
    @assert a in env.action_space
    env.action = a
    middle_action = (env.n_actions + 1) / 2
    if Base.mod(env.n_actions, 2) == 0
        lower = (env.n_actions) / 2
        if a < middle_action
            symmetric_a = -lower + a - 1
        elseif a > middle_action
            symmetric_a = a - lower
        end
    else
        lower = (env.n_actions - 1) / 2
        symmetric_a = -lower + a - 1
    end
    discretization = env.bound / lower
    float_a = [Float32.(discretization * symmetric_a)]
    _step!(env, float_a)
end

function _step!(env::LQREnv, a)
    env.reward =
        -copy(
            transpose(env.state) * env.params.S * env.state +
            transpose(a) * env.params.R * a,
        )
    env.state = env.params.A * env.state + env.params.B * a
    env.t += 1
    if env.t == env.max_steps
        env.done = true
    end
    nothing
end

##
## Model-based closed-form
##

function optimal_linear_policy(env::LQREnv)
    A = copy(sqrt(env.gamma) * env.params.A)
    B = copy(sqrt(env.gamma) * env.params.B)
    R = copy(env.params.R)
    S = copy(env.params.S)
    X, evals, f, z = ared(env.params.A, env.params.B, env.params.R, env.params.S)
    K = -inv(R + transpose(B) * X * B) * transpose(B) * X * A
end

function P_π(env::LQREnv, K)
    M = env.params.S + transpose(K) * env.params.R * K
    L = sqrt(env.gamma) * (env.params.A + env.params.B * K)
    E = I(env.s_d)
    P = lyapd(L, E, M)
end

function value_function(env::LQREnv, K)
    P = P_π(env, K)
    return x -> -transpose(x) * P * x
end

##
## Compat for LQR
##

function preprocess_action(env::LQREnv{<:MultiContinuousSpace}, a)
    a = clamp.(a, -env.bound, env.bound)
    return Float32.(a)
end
