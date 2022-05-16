

function default_action(env::AbstractCurriculumMDP)
    s = get_state(env)
    return default_action(env, s)
end

function default_action(env::AbstractCurriculumMDP, state)
    a = default_action(env, env.student_env)
    if env.action_type == :tandem
        a = 2
    end

    return a, 1f0
end


function teacher_action!(env, action)
    if env.action_type == :start_state
        set_start_state!(env, action)
    elseif env.action_type == :goal_state
        set_goal_state!(env, action)
    elseif env.action_type == :parameters
    elseif env.action_type == :tandem
        set_exploration_rate!(env, action)
    end
end

function set_exploration_rate!(env::UnionCurriculumMDP, teacher_action)
    eps = collect(1:num_actions(env))
    eps = [1.25f0^(-e + 1) for e in eps]
    env.student_agent.π_b.ϵ = eps[teacher_action]
end


##
## GridWorld
##

function calc_threshold(env::TabularGridWorld, G)
    thresh = score_threshold(env)
    score = mean(G[1] .> thresh)
    thresh = success_threshold(env)
    if score > thresh
        return true, score, thresh
    else
        return false, score, thresh
    end
end

function score_threshold(env::TabularGridWorld)
    return 0.99^15
end

function success_threshold(env::TabularGridWorld)
    return 0.95
end

function optimal_action(env::UnionCurriculumMDP{E}, s) where {E <: TabularGridWorld}
    return default_action(env.student_env)
end

function get_configurable_params(env::TabularGridWorld)
    if env.name == "FourRooms"
        # Base.OneTo(7)
        Base.OneTo(39)
    else
        error("Not a valid env.name for curriculum MDP")
    end
end

function default_action(env::AbstractCurriculumMDP, student_env::TabularGridWorld{E}) where {E <: AbstractGridWorldUndirected}
    return argmax(xy2int(student_env, (2,2)) .== get_valid_states(student_env))
end

function default_action(env::AbstractCurriculumMDP, student_env::TabularGridWorld{E}) where {E <: AbstractGridWorldDirected}
    return argmax(xy2int(student_env, (2,2,1)) .== get_valid_states(student_env))
end

function set_start_state!(env::UnionCurriculumMDP{E}, teacher_action) where {E<:TabularGridWorld}
    agent_pos = discrete_agent_pos(env.student_env, teacher_action)
    reset!(env.student_env, agent_pos = agent_pos)
end

function discrete_agent_pos(env, agent_pos)
    # list_of_states = [CartesianIndex((2,2)), # Top-left corner
    #     CartesianIndex((2,8)), # Bottom-right corner
    #     CartesianIndex((8,2)), # Top-right corner
    #     CartesianIndex((5,7)), # Right corridor
    #     CartesianIndex((7,5)), # Bottom corridor
    #     CartesianIndex((5,3)), # Left corridor
    #     CartesianIndex((3,5)), # Top corridor
    #                     ] # Bottom-right corner is goal
    list_of_states = get_valid_nonterminal_states(env)
    list_of_states = [int2xy(env, s) for s in list_of_states]
    return list_of_states[agent_pos]
    # if agent_pos == 1
    #     return CartesianIndex((2,2)) # Top-left corner
    # elseif agent_pos == 2
    #     return CartesianIndex((2,8)) # Bottom-right corner
    # elseif agent_pos == 3
    #     return CartesianIndex((8,2)) # Top-right corner
    # elseif agent_pos == 5
    #     return CartesianIndex((5,7)) # Right corridor
    # elseif agent_pos == 4
    #     return CartesianIndex((7,5)) # Bottom corridor
    # elseif agent_pos == 6
    #     return CartesianIndex((5,3)) # Left corridor
    # elseif agent_pos == 7
    #     return CartesianIndex((3,5)) # Top corridor
    # end
end

##
## AbstractMDP
##

function calc_threshold(env::AbstractMDP, G)
    thresh = success_threshold(env)
    score = all(G[1] .> thresh)
    if score
        return true, score, thresh
    else
        return false, score, thresh
    end
end

function score_threshold(env::AbstractMDP)
    return 11
end

function success_threshold(env::AbstractMDP)
    return 0.99^9
end

function optimal_action(env::UnionCurriculumMDP{E}, s) where {E<:AbstractMDP}
    return default_action(env.student_env)
end

function get_configurable_params(env::AbstractMDP)
    Base.OneTo(10)
end

function default_action(env::AbstractCurriculumMDP, student_env::AbstractMDP)
    return student_env.initial_state
end

function set_start_state!(env::UnionCurriculumMDP{E}, teacher_action) where {E<:AbstractMDP}
    reset!(env.student_env, agent_pos = teacher_action)
end

##
## CartPole
##

function calc_threshold(env::CartPoleEnv, G)
    thresh = score_threshold(env)
    score = mean(G[2] .>= thresh)
    # println(G[2])
    thresh = success_threshold(env)
    if score >= thresh
        return true, score, thresh
    else
        return false, score, thresh
    end
end

function score_threshold(env::CartPoleEnv)
    100f0
end

function success_threshold(env::CartPoleEnv)
    0.95
end

function optimal_action(env::UnionCurriculumMDP{E}, s) where {E<:CartPoleEnv}
    return default_action(env.student_env)
end

function get_configurable_params(env::CartPoleEnv)
    Base.OneTo(6)
end

function default_action(env::CartPoleEnv)
    return 3
end

function set_start_state!(env::UnionCurriculumMDP{E}, teacher_action) where {E<:CartPoleEnv}
    lrs = [0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001]
    teacher_action = lrs[teacher_action]
    env.student_agent.subagents[1].optimizer.eta = teacher_action

    env.student_env =
        CartPoleEnv(T = Float32, rng = env.rng)
end

function threshold(env::CartPoleEnv)
    75
end

##
## MountainCarEnv
##

function calc_threshold(env::MountainCarEnv, G)
    thresh = score_threshold(env)
    score = mean(G[2] .<= thresh)
    max_step = 199f0
    # score = -mean(G[2])/(max_step - thresh) + max_step/(max_step - thresh)
    thresh = success_threshold(env)
    if score >= thresh
        return true, score, thresh
    else
        return false, score, thresh
    end
end

function score_threshold(env::MountainCarEnv)
    140f0
end

function success_threshold(env::MountainCarEnv)
    0.95
end

function optimal_action(env::UnionCurriculumMDP{E}, s) where {E<:MountainCarEnv}
    return default_action(env.student_env)
end

function get_configurable_params(env::MountainCarEnv)
    Base.OneTo(20)
end

function default_action(env::MountainCarEnv)
    n_actions = length(get_configurable_params(env))
    return Int(floor(n_actions/2))
end

function set_start_state!(env::UnionCurriculumMDP{E}, teacher_action) where {E<:MountainCarEnv}
    n_actions = length(get_configurable_params(env.student_env))
    positions = collect(range(-1.2, 0.6, length = n_actions))
    teacher_action = positions[teacher_action]
    reset!(env.student_env, teacher_action)
end


function reset!(env::MountainCarEnv{A,T}, pos) where {A,T}
    env.state[1] = pos
    env.state[2] = 0.0
    env.done = false
    env.t = 0
    nothing
end


##
## GoalEnv{AbstractMDP}
##

function calc_threshold(env::GoalEnv{E}, G) where {E<:AbstractMDP}
    thresh = success_threshold(env)
    score = all(G[1] .>= thresh)
    if score
        return true, score, thresh
    else
        return false, score, thresh
    end
end

function score_threshold(env::GoalEnv{E}) where {E<:AbstractMDP}
    return 11
end

function optimal_episode_length(env::GoalEnv{E}) where {E<:AbstractMDP}
    10
end

function student_episode_length(env::GoalEnv{E}) where {E<:AbstractMDP}
    15
end

function optimal_episode_length(env::GoalEnv{E}) where {E<:TabularGridWorld}
    12
end

function student_episode_length(env::GoalEnv{E}) where {E<:TabularGridWorld}
    25
end

function success_threshold(env::GoalEnv{E}) where {E<:AbstractMDP}
    N = optimal_episode_length(env)
    return sum([-0.99f0^i for i = 0:(N-2)])
end

function optimal_action(env::UnionCurriculumMDP{GoalEnv{E}}, s) where {E<:AbstractMDP}
    return default_action(env.student_env)
end

function get_configurable_params(env::GoalEnv{E}) where {E<:AbstractMDP}
    n_states = get_valid_states(env.env)
    Base.OneTo(length(n_states))
end

function default_action(env::GoalEnv{E}) where {E<:AbstractMDP}
    target_goal = get_goal_states(env.env)[1]
    return target_goal
end

function set_start_state!(env::UnionCurriculumMDP{GoalEnv{E}}, teacher_action) where {E<:AbstractMDP}
    #TODO using 11 dim tabular state, annoyingly includes terminal state
    n_states = get_valid_states(env.student_env.env)
    goal = discrete_action_mask(teacher_action, length(n_states))
    reset!(env.student_env, goal = goal)
end


##
## GoalEnv{Acrobot}
##

function calc_threshold(env::GoalEnv{E}, G) where {E<:AcrobotEnv}
    thresh = success_threshold(env)
    score = all(G[1] .>= thresh)
    if score
        return true, score, thresh
    else
        return false, score, thresh
    end
end

function score_threshold(env::GoalEnv{E}) where {E<:AcrobotEnv}
    return 11
end

function optimal_episode_length(env::GoalEnv{E}) where {E<:AcrobotEnv}
    10
end

function student_episode_length(env::GoalEnv{E}) where {E<:AcrobotEnv}
    15
end

function success_threshold(env::GoalEnv{E}) where {E<:AcrobotEnv}
    N = optimal_episode_length(env)
    return sum([-0.99f0^i for i = 0:(N-2)])
end

function optimal_action(env::UnionCurriculumMDP{GoalEnv{E}}, s) where {E<:AcrobotEnv}
    return default_action(env.student_env)
end

function get_configurable_params(env::GoalEnv{E}) where {E<:AcrobotEnv}
    n_states = get_valid_states(env.env)
    Base.OneTo(length(n_states))
end

function default_action(env::GoalEnv{E}) where {E<:AcrobotEnv}
    target_goal = get_goal_states(env.env)[1]
    return target_goal
end

function set_goal_state!(env::UnionCurriculumMDP{GoalEnv{E}}, teacher_action) where {E<:AcrobotEnv}
    #TODO using 11 dim tabular state, annoyingly includes terminal state
    n_states = get_valid_states(env.student_env.env)
    goal = discrete_action_mask(teacher_action, length(n_states))
    reset!(env.student_env, goal = goal)
end
