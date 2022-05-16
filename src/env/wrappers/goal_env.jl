abstract type AbstractGoalEnv <: AbstractEnv end

mutable struct GoalEnv{E} <: AbstractGoalEnv
    env::E
    goal_space
    goal
end

Base.show(io::IO, t::MIME"text/plain", env::AbstractGoalEnv) = begin
    println()
    println("---------------------------")
    name = "GoalEnv"
    L = length(name)
    println(name)
    println(join(["-" for i = 1:L+1]))


    print(io, "  env.env: ")
    println(io, env.env)

    println("---------------------------")
    print(io, "  env.goal: ")
    println(io, env.goal)

    println("---------------------------")
end

function GoalEnv(base_env::AbstractEnv, goal_space; kwargs...)
    n_states = length(get_valid_states(base_env))
    goal = zeros(Float32, n_states)
    goal[end] = 1f0
    return GoalEnv(base_env, goal_space, goal)
end

function Random.seed!(env::AbstractGoalEnv, seed)
    Random.seed!(env.env, seed)
end

function env_mode!(env::AbstractGoalEnv; mode = :default)
    test_env = deepcopy(env)
    if mode !== :default
        test_env.goal_space = :eval
    else
        test_env.goal_space = nothing
    end
    return test_env
end


function reset!(env::AbstractGoalEnv; goal = nothing)
    if isnothing(goal)
        # if !isnothing(env.goal_space)
        #     goal = zeros(Float32, 11)
        #     goal[end] = 1f0
        #     env.goal = goal
        # else
        #     goal = zeros(Float32, 11)
        #     i = rand(env.env.rng, 1:10)
        #     goal[i] = 1f0
        #     env.goal = goal
        # end
    else
        env.goal = goal
    end
    return reset!(env.env)
end

function get_state(env::AbstractGoalEnv)
    return get_state(env.env)
end

function get_obs(env::AbstractGoalEnv)
    if !isnothing(env.goal)
        return vcat(get_obs(env.env), env.goal)
    else
        return get_obs(env.env)
    end
end

function get_actions(env::AbstractGoalEnv)
    return get_actions(env.env)
end

function get_reward(env::AbstractGoalEnv)
    s = get_obs(env.env)
    if s == env.goal
        # return 1f0
        return 0f0
    else
        # return 0f0
        return -1f0
    end
end

function get_terminal(env::AbstractGoalEnv)
    s = get_obs(env.env)
    return s == env.goal
end

function get_info(env::AbstractGoalEnv)
end

function random_action(env::AbstractGoalEnv)
    random_action(env.env)
end

function (env::AbstractGoalEnv)(a)
    env.env(a)
end

function optimal_action(env::AbstractGoalEnv, s)
    optimal_action(env.env, s)
end

function optimal_action(env::AbstractGoalEnv)
    optimal_action(env.env)
end
