

import ReinforcementLearningEnvironments
import ReinforcementLearningEnvironments: CartPoleEnv, MountainCarEnv, PendulumEnv, AtariEnv
import ReinforcementLearningEnvironments: AcrobotEnv
import ReinforcementLearningEnvironments: Space

import Random
import Random: AbstractRNG, GLOBAL_RNG

# using PyCall
import ReinforcementLearningEnvironments: GymEnv

# import LyceumMuJoCo
# import LyceumMuJoCo: AbstractMuJoCoEnvironment, setaction!, actionspace
# import LyceumMuJoCo: getreward, getobs, isdone, getstate, step!, getaction
# import LyceumMuJoCo: HopperV2

using IntervalSets

abstract type AbstractContextualMDP{E} <: AbstractEnv end

# include("LQR.jl")
include("MDP.jl")
include("wrappers/wrappers.jl")
include("compatibility/compatibility.jl")
include("RNNMDP.jl")
include("contextualenvs.jl")
include("curriculum/curriculum_mdp.jl")
include("optimization_env.jl")

function get_env(EnvType; skip = 1, max_steps = 200, seed = nothing, value_reward = false, kwargs...)
    env_str = split(EnvType, "_")
    if seed === nothing
        rng = GLOBAL_RNG
    else
        rng = MersenneTwister(seed)
    end

    if :state_representation in keys(kwargs)
        state_representation = kwargs[:state_representation]
    else
        state_representation = :parameters
    end

    if :recurrent_action_dim in keys(kwargs)
        recurrent_action_dim = kwargs[:recurrent_action_dim]
    else
        recurrent_action_dim =  false
    end

    if :continuing in keys(kwargs)
        continuing =  kwargs[:continuing]
    else
        continuing = false
    end

    if EnvType == "Pendulum"
        max_env_steps = max_steps * skip
        env = PendulumEnv(;
            T = Float32,
            n_actions = 3,
            continuous = false,
            rng = rng,
            max_steps = max_env_steps,
        )
        action_decoder = x -> softmax(x)

    elseif EnvType == "Acrobot"
        max_env_steps = max_steps * skip
        env = AcrobotEnv(; T = Float32, rng = rng, max_steps = max_env_steps)
        action_decoder = x -> softmax(x)

    elseif EnvType == "PendulumCont"
        max_env_steps = max_steps * skip
        env = PendulumEnv(; T = Float32, rng = rng, max_steps = max_env_steps)
        action_decoder = x -> 2 * tanh.(x)

        # elseif EnvType == "LQRDiscrete"
        #     max_env_steps = max_steps*skip
        #     env = LQREnv(n_actions = 3, bound = 0.1, max_steps = max_env_steps, rng = rng)
        #     action_decoder = x -> softmax(x)

        # elseif EnvType == "LQRClip"
        #     max_env_steps = max_steps*skip
        #     env = LQREnv(bound = 0.1, max_steps = max_env_steps, rng = rng)
        #     action_decoder = x -> softmax(x)

    elseif EnvType == "CartPole"
        max_env_steps = max_steps * skip
        env =
            CartPoleEnv(T = Float32, max_steps = max_env_steps, dt = 0.02 / skip, rng = rng)
        action_decoder = x -> softmax(x)

    elseif EnvType == "CartPoleCont"
        max_env_steps = max_steps * skip
        env =
            CartPoleEnv(T = Float32, max_steps = max_env_steps, dt = 0.02 / skip, rng = rng, continuous = true)
        action_decoder = x -> softmax(x)

    elseif EnvType == "MountainCar"
        max_env_steps = 200# max_steps * skip
        env =
            MountainCarEnv(T = Float32, max_steps = max_env_steps, rng = rng)
        action_decoder = x -> softmax(x)

    elseif EnvType == "Hopper"
        max_env_steps = max_steps * skip
        env = HopperV2()
        action_decoder = x -> 2 * tanh.(x)

    elseif EnvType == "MNISTBANDIT"
        max_env_steps = 2
        env = ContextualBandit("MNIST"; kwargs..., rng = rng)
        action_decoder = x -> softmax(x)

    elseif EnvType == "MNISTMDP"
        max_env_steps = max_steps * skip
        env = ContextualMDP("MNIST"; kwargs..., rng = rng)
        action_decoder = x -> softmax(x)

    elseif EnvType == "CIFARMDP"
        max_env_steps = max_steps * skip
        env = ContextualMDP("CIFAR10"; kwargs..., rng = rng)
        action_decoder = x -> softmax(x)

    elseif EnvType == "CompassWorld"
        max_env_steps = max_steps * skip
        env = CompassWorld(rng = rng)
        action_decoder = x -> softmax(x)

    elseif contains(env_str[end], "OptEnv")
        sub_env_str = split(env_str[end], "-")
        max_env_steps = max_steps * skip
        env = OptEnv(string(state_representation), sub_env_str[2], sub_env_str[3], sub_env_str[4], max_steps = max_env_steps, rng = rng)
        action_decoder = x -> softmax(x)

    elseif contains(env_str[1], "RNN")
        max_env_steps = max_steps * skip
        rng = MersenneTwister(seed)
        action_dim = parse(Int, env_str[2])
        state_dim = parse(Int, env_str[3])
        env = RNNMDP(action_dim, state_dim, rng = rng; kwargs...)
        action_decoder = x -> softmax(x)

    elseif contains(env_str[end], "Goal")
        env_str_remains = join(env_str[1:end-1], "_")
        max_env_steps = max_steps * skip
        rng = MersenneTwister(seed)
        base_env, _, _ = get_env(env_str_remains, skip = skip, seed = seed, max_steps = max_env_steps; kwargs...)
        env = GoalEnv(base_env, nothing; kwargs...)
        action_decoder = x -> softmax(x)

    elseif contains(env_str[end], "GW")
        max_env_steps = max_steps * skip
        rng = MersenneTwister(seed)
        env = TabularGridWorld(EnvType[1:end-2], rng = rng; kwargs...)
        action_decoder = x -> softmax(x)

    elseif contains(env_str[end], "MDPFeaturized")
        L = length("MDPFeaturized")
        max_env_steps = max_steps * skip
        rng = MersenneTwister(seed)
        env = MDP(EnvType[1:end-L], rng = rng, tabular = false; kwargs...)
        action_decoder = x -> softmax(x)

    elseif contains(env_str[end], "MDP")
        max_env_steps = max_steps * skip
        rng = MersenneTwister(seed)
        env = MDP(EnvType[1:end-3], rng = rng; kwargs...)
        action_decoder = x -> softmax(x)

    elseif env_str[end] == "Meta"
        max_env_steps = max_steps * skip
        rng = MersenneTwister(seed)

        encode_dim = 128

        base_env, max_env_steps, action_decoder = get_env(env_str[1], skip = skip, seed = seed, max_steps = max_env_steps)
        env = MetaEnv(base_env, encode_dim)

        action_decoder = x -> softmax(x)


    elseif env_str[end] == "CDP"
        max_env_steps = max_steps * skip
        rng = MersenneTwister(seed)

        base_mdp, _, _ = get_env(env_str[1], skip = skip, seed = seed, max_steps = max_env_steps)
        observation_set = env_str[2]
        env = ContextualMDP(observation_set, base_mdp; kwargs..., rng = rng)

        action_decoder = x -> softmax(x)

    elseif contains(env_str[end], "Curriculum")
        if contains(env_str[end], "Tandem")
            action_type = :tandem
        else
            action_type = :start_state
        end

        env_str_remains = join(env_str[1:end-1], "_")
        max_env_steps = max_steps * skip
        rng = MersenneTwister(seed)

        student_env, _, _ = get_env(env_str_remains, skip = skip, seed = seed, max_steps = max_steps, negative_reward = false)
        env = CurriculumMDP(student_env, state_representation, max_env_steps, continuing; rng = rng, recurrent_teacher_action_encoder = recurrent_action_dim, action_type = action_type, kwargs...)

        action_decoder = x -> softmax(x)
    else
        error("Not a valid EnvType")
    end
    if value_reward
        env = ValueEnv(env)
    end

    Random.seed!(env, seed)
    max_agent_steps = max_env_steps - 1
    return env, max_agent_steps, action_decoder
end

function optimal_action(env::CartPoleEnv, s)
    if s[1] + s[2] < 0
        action = 2
    else
        action = 1
    end
end

function eval_plan(env::AbstractEnv, s, as::AbstractArray{T,2}) where {T<:Union{AbstractFloat,Int}}
    reset!(env, s)
    q = 0.0f0
    length_of_plan = size(as)[end]
    for t = 1:length_of_plan
        a = as[:, t]
        env(a)
        r = copy(get_reward(env))
        q += r
    end
    return q
end

function eval_plan(env, s, as::AbstractArray{T,3}) where {T<:Union{AbstractFloat,Int}}
    qs = []
    num_plans = size(as)[end]
    for i = 1:num_plans
        a_plan = as[:, :, i]
        q = eval_plan(env, s, a_plan)
        push!(qs, q)
    end
    return qs
end

function generate_experience(env, state, action, prob::AbstractFloat)
    action = preprocess_action(env, action)
    s = get_state(env)
    o = get_obs(env)
    env(action)

    sp = get_state(env)
    op = get_obs(env)
    r = get_reward(env)
    done = get_terminal(env)
    info = get_info(env)
    info = isnothing(info) ? nothing : info

    exp = Experience(s, o, action, Float32(prob), Float32(r), sp, op, done, info)
    return exp, done
end

function interact!(
    env,
    agent;
    greedy = false,
    policy = :agent,
    reward_only = false,
    buffer = nothing,
)

    state = get_obs(env)

    if policy == :agent
        policy = agent.π_b
    end

    if policy == :optimal
        state = get_state(env)
        action = optimal_action(env, state)
        prob = 1.0f0
    elseif policy == :random
        action = random_action(env)
        prob = 0.5f0
    elseif policy == :default
        action, prob = default_action(env, state)
    # elseif policy == :agent
    #     state = state |> agent.device  |> agent.state_encoder
    #     policy = agent.π_b
    #     action, prob = sample(policy, state, greedy = greedy, rng = agent.rng)
    elseif policy == :greedy_agent
        state = state |> agent.device  |> agent.state_encoder
        policy = agent.π_b
        action, prob = sample(policy, state |> agent.device, greedy = true, rng = agent.rng)
    elseif typeof(policy) <: AbstractModel
        state = state |> agent.device  |> agent.state_encoder
        action, prob = sample(policy, state |> agent.device, greedy = greedy, rng = agent.rng)
    elseif typeof(policy) <: Function
       action, prob = policy(env)
    elseif typeof(policy) <: Number
        # if greedy
            action = policy
            prob = 1f0
        # else
        #     if randn() > 0.1
        #         action = policy
        #         prob = 1f0
        #     else
        #         action = random_action(env)
        #         prob = 1f0
        #     end
        # end
    elseif policy == :grad_init
        # println("V: ", mean(agent.subagents[1](get_next_obs_with_f(env, env.f) |> agent.device)))
        action, prob = default_action(env, state)
    else
        action = policy(state)
        prob = 1.0f0
    end
    if reward_only
        # action = preprocess_action(env, action)
        # env(action)
        # ex = copy(get_reward(env))
        # done = copy(get_terminal(env))
    else
        ex, done = generate_experience(env, state, action, prob)
    end

    # if typeof(env) <: AbstractCurriculumMDP
    #     println("teacher action", action)
    # end
    if !isnothing(buffer)
        add_exp!(buffer, ex)
        return done
    else
        return ex, done
    end
end

function generate_episode(
    env,
    agent;
    policy,
    state = nothing,
    action = nothing,
    greedy = false,
    max_steps = nothing,
)
    if max_steps == nothing
        max_steps = agent.max_agent_steps
    end

    if typeof(agent) == AbstractAgent
        reset_model!(agent.π_b)
    end

    # start simulation
    if state == :dontreset
    elseif state === nothing
        reset!(env)
    else
        reset!(env, state)
    end

    if policy == :grad_init
        reset!(env, agent = agent)
    end

    S = typeof(get_state(env))
    O = typeof(get_obs(env))
    A = typeof(random_action(env))
    episode = Vector{Experience{S,O,A}}(undef, max_steps)


    done = get_terminal(env)

    step = 0
    if action !== nothing
        step += 1
        ex, done = generate_experience(env, state, action, 1.0f0)
        episode[step] = ex
    end

    while !done
        step += 1
        ex, done = interact!(env, agent, greedy = greedy, policy = policy)
        episode[step] = ex

        if step == max_steps
            break
        end
    end

    return episode[1:step]
end
