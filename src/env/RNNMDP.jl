abstract type AbstractRNNMDP <: AbstractEnv end

mutable struct RNNMDP <: AbstractRNNMDP
    rnn
    value
    policy
    action_space
    state
    reward
    done
    t
    gamma
    name
    rng
end

Base.show(io::IO, t::MIME"text/plain", env::AbstractRNNMDP) = begin
    println()
    println("---------------------------")
    name = "AbstractRNNMDP"
    L = length(name)
    println(name)
    println(join(["-" for i = 1:L+1]))

    print(io, "  env.name: ")
    println(io, env.name)

    println("---------------------------")
end

function RNNMDP(action_dim, state_dim; gamma = 1.0f0, rng = Random.GLOBAL_RNG, kwargs...)
    f = rnn(action_dim, state_dim, rng = rng, recurrence = RNN)
    #V = feed_forward(state_dim, action_dim, 32, rng = rng, output = (x) -> Flux.relu.(x))
    V = feed_forward(state_dim, action_dim, 32, rng = rng)
    action_space = Base.OneTo(action_dim)
    policy = [1/action_dim for _ = 1:action_dim]
    state = zeros(Float32, state_dim)
    reward = 0f0
    done = false
    t = 0
    name = "RNNMDP"
    env = RNNMDP(f, V, policy, action_space, state, reward, done, t, gamma, name, rng)
    reset!(env)
    return env
end

function step(env::AbstractRNNMDP, a::Int)
    A = num_actions(env)
    a = Float32.(discrete_action_mask(a, A))
    env.rnn(a)
end

function (env::AbstractRNNMDP)(a)
    step(env, a)
    env.t += 1
    env.done = env.t >= 100 ? true : false
    sp = env.rnn.layers[1].state
    # f = feed_forward(4, 1, 32, seed = 1)(sp)[1]
    # println(f)

    # if f > 0.95
    #     env.done = true
    # end

    if env.done
        r = env.value(env.state)[a]
    else
        Q_max = maximum(env.value(env.rnn.layers[1].state))
        r = env.value(env.state)[a] - env.gamma*Q_max
    end
    env.reward = r
    env.state = sp
    nothing
end

function optimal_action(env::AbstractRNNMDP, s)
    return argmax(env.value(s))[1]
end

function reset!(env::AbstractRNNMDP)
    Flux.reset!(env.rnn)
    env.state = env.rnn.layers[1].state
    env.t = 0
    env.reward = 0.0f0
    env.done = false
    nothing
end

function Random.seed!(env::AbstractRNNMDP, seed)
end

function get_state(env::AbstractRNNMDP)
    return env.state
end

function get_obs(env::AbstractRNNMDP)
    return env.state
end

function get_terminal(env::AbstractRNNMDP)
    return env.done
end

function get_reward(env::AbstractRNNMDP)
    return env.reward
end

function get_actions(env::AbstractRNNMDP)
    return env.action_space
end

function random_action(env::AbstractRNNMDP)
    return rand(env.rng, get_actions(env))
end
