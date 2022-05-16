abstract type AbstractValueEnv <: AbstractEnv end

mutable struct ValueEnv <: AbstractValueEnv
    env::AbstractEnv
    V
    last_s
end

Base.show(io::IO, env::AbstractValueEnv) = begin
    println("Value Env")
    Base.show(io, env.env)
end


Base.show(io::IO, t::MIME"text/markdown", env::AbstractValueEnv) = begin
    println("Value Env")
    Base.show(io, t, env.env)
end

function ValueEnv(env, V = :random)
    ns = length(get_obs(env))
    na = num_actions(env)
    if V == :random
        seed = 1
        hidden_size = 64
        drop_rate = 0.0
        num_layers = 2
        activation = tanh
        V = feed_forward(
            ns,
            1,
            hidden_size,
            drop_rate = drop_rate,
            num_hidden_layers = num_layers,
            σ = activation,
            seed = seed,
            initb = Flux.ones,
        )
    end
    a = random_action(env)
    last_s = get_state(env)
    env(a)
    return ValueEnv(env, V, last_s)
end

###
### Need the following for any env
###

function reset!(env::AbstractValueEnv)
    reset!(env.env)
    env.last_s = nothing
end

function reset!(env::AbstractValueEnv, s)
    reset!(env.env, s)
    env.last_s = nothing
end

function random_action(env::AbstractValueEnv)
    random_action(env.env)
end

function optimal_action(env::AbstractValueEnv, s)
    optimal_action(env.env, s)
end

function get_obs(env::AbstractValueEnv)
    get_obs(env.env)
end

function get_state(env::AbstractValueEnv)
    get_state(env.env)
end

function get_reward(env::AbstractValueEnv)
    V_sp = env.V(get_obs(env.env))
    V_s = env.V(env.last_s)
    reshape(V_sp - V_s, :)[1]
end

function get_reward(env, sp, s)
    # gamma = 0.99f0
    gamma = 1.0f0
    V_sp = env.V(sp)[1]
    V_s = env.V(s)[1]
    return -(gamma*V_sp - V_s)
end

function get_actions(env::AbstractValueEnv)
    get_actions(env.env)
end

function get_terminal(env::AbstractValueEnv)
    get_terminal(env.env)
end

function Random.seed!(env::AbstractValueEnv, seed)
    Random.seed!(env.env, seed)
end

function (env::AbstractValueEnv)(a)
    env.last_s = copy(get_state(env.env))
    env.env(a)
end

function transform_reward_buffer(env::AbstractValueEnv, buffer)
    num_eps = get_num_episodes(buffer)
    new_r_buffer = deepcopy(buffer)
    for i = 1:num_eps
        ep = buffer._episodes[i]
        length_ep = length(ep)
        for j = 1:length_ep
            transition = new_r_buffer._episodes[i][j]
            sp = transition.sp
            s = transition.s
            new_r = get_reward(env, sp, s)
            transition.r = new_r
        end
    end
    return new_r_buffer
end
