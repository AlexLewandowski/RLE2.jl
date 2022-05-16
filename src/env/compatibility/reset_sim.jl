SimEnv = Union{CartPoleEnv, AcrobotEnv, PendulumEnv}

function reset!(env::CartPoleEnv, state::AbstractArray{Float32,1})
    #@assert state in env.observation_space
    env.state = copy(state)
    env.action = 2
    env.done = false
    env.t = 0
    nothing
end

function reset!(env::AcrobotEnv, state::AbstractArray{Float32,1})
    #@assert state in env.observation_space
    env.state = copy(state)
    env.action = 2
    env.done = false
    env.t = 0
    nothing
end

function reset!(env::PendulumEnv, state::AbstractArray{Float32,1})
    @assert state in env.observation_space
    env.state = [acos(state[1]), state[3]]
    env.reward = 0.0f0
    env.done = false
    env.t = 0
    nothing
end
