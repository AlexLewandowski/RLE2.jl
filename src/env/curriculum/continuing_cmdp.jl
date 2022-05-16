
abstract type AbstractContinuingCurriculumMDP <: AbstractCurriculumMDP end

mutable struct ContinuingCurriculumMDP{E} <: AbstractContinuingCurriculumMDP
    student_env::E
    student_agent::AbstractAgent
    state_representation::Symbol
    state_update::Function
    state_encoder::Any
    student_action_encoder::Any
    teacher_action_encoder::Any
    state::Any
    action::Any
    reward::Any
    done::Any
    student_return::Any
    t::Int
    max_steps::Any
    configurable_params::Any
    rng::AbstractRNG
end

function reinitialize_state(env::AbstractContinuingCurriculumMDP)
    state_representation = env.state_representation
    if state_representation == :return
        env.state = [env.t, env.action, 0.0f0]
    elseif contains(String(state_representation), "trajectory_resevoir")
        encoding_dim = size(env.state_encoder.layers[1].cell.Wi)[1]
        env.state = zeros(Float32, encoding_dim + 2)
        Flux.reset!(env.state_encoder)
    elseif state_representation == :parameters
        env.state = vcat(0.0f0, reshape(env.student_agent.π_b.f.f, :))
    elseif state_representation == :last_state
        env.state = vcat(0.0f0, get_state(env.student_env), 0.0f0)
    end
end

function reset!(env::AbstractContinuingCurriculumMDP)
    reset!(env.student_env)
    env.student_agent = get_agent(env.student_env)
    env.reward = [0.0f0, 0.0f0]
    env.done = false
    env.action = Int(rand(env.rng, env.configurable_params))
    env.t = 0
    initialize_state(env)
    return nothing
end

function (env::AbstractContinuingCurriculumMDP)(action)
    @assert action in env.configurable_params
    env.t += 1
    G_state = rollout_student(env, action, learning = true, state_update = true)
    G, steps = rollout_student(env, learning = false, state_update = false, greedy = false)
    if env.t == env.max_steps
        env.done = true
    end
    env.reward[1] = env.reward[2]
    env.reward[2] = G
    env.state, env.done
end

function update_env_state!(env::AbstractContinuingCurriculumMDP, experience)
    env.state = vcat(env.t, env.state_update(env, experience))
end

function get_reward(env::AbstractContinuingCurriculumMDP)
    return env.reward[2] - env.reward[1]
end
