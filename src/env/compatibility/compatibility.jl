include("atari.jl")
include("gridworlds.jl")
# include("gym.jl")
# include("mujoco.jl")
include("reset_sim.jl")

function preprocess_action(env::AbstractEnv, a)
    a
end

function env_mode!(env::AbstractEnv; mode = :default)
    return env
end

function reset!(env::AbstractEnv)
    return ReinforcementLearningEnvironments.reset!(env)
end

function get_state(env::AbstractEnv)
    return deepcopy(ReinforcementLearningEnvironments.state(env))
end

function get_obs(env::AbstractEnv)
    return deepcopy(ReinforcementLearningEnvironments.state(env))
end

function get_actions(env::AbstractEnv)
    return ReinforcementLearningEnvironments.action_space(env)
end

function get_reward(env::AbstractEnv)
    return deepcopy(ReinforcementLearningEnvironments.reward(env))
end

function get_terminal(env::AbstractEnv)
    return ReinforcementLearningEnvironments.is_terminated(env)
end

function get_info(env::AbstractEnv)
end

function is_tabular(env::AbstractEnv)
    return typeof(get_obs(env)) == Vector{Bool}
end

function state_size(env::AbstractEnv)
    s = size(get_state(env))
    if s == ()
        s = (1,)
    end
    return s
end

function obs_size(env::AbstractEnv)
    s = size(get_obs(env))
    if s == ()
        s = (1,)
    end
    return s
end

function num_actions(env::AbstractEnv)
    space = get_actions(env)
    space_type = typeof(get_actions(env))
    if space_type <: Base.OneTo{Int64}
        return length(space)
    elseif space_type <: IntervalSets.Interval
        return length(space.left)
    elseif space_type <: ReinforcementLearningEnvironments.Space
        return length(space)
    end
end

function random_action(env::AbstractEnv)
    #TODO return probability of aciton as well
    # space = get_actions(env)
    space_type = typeof(get_actions(env))
    # if space_type <: Base.OneTo
    #     a = rand(env.rng, space)
    # elseif space_type <: Union{ContinuousSpace,MultiContinuousSpace}
    #     N = num_actions(env)
    #     if Inf in space.high || -Inf in space.low
    #         a = randn(env.rng, Float32, N)
    #     else
    #         a = Float32.(rand(env.rng, space))
    #     end
    # end
    if space_type <: IntervalSets.ClosedInterval
        a = Float32.(rand(env.rng, get_actions(env)))
    else
        a = rand(env.rng, get_actions(env))
    end
    return a
end

function action_type(env::AbstractEnv)
    typeof(random_action(env)[1])
end

# function (env::AbstractEnv)(action)
# end

# function reset!(env::AbstractEnv)
# end

# function Random.seed!(env::AbstractEnv, seed)
# end

# function get_state(env::AbstractEnv)
# end

# function get_obs(env::AbstractEnv)
# end

# function get_terminal(env::AbstractEnv)
# end

# function get_reward(env::AbstractEnv)
# end

# function get_actions(env::AbstractEnv)
# end

# function random_action(env::AbstractEnv)
# end
