abstract type AbstractMDP <: AbstractEnv end

mutable struct MDP <: AbstractMDP
    state_transition_mat
    reward_mat
    initial_state
    terminal_states
    state
    reward
    done
    t
    V
    gamma
    name
    info
    tabular
    rng::AbstractRNG
end

Base.show(io::IO, t::MIME"text/plain", env::AbstractMDP) = begin
    println()
    println("---------------------------")
    name = "AbstractMDP"
    L = length(name)
    println(name)
    println(join(["-" for i = 1:L+1]))

    print(io, "  env.name: ")
    println(io, env.name)

    println("---------------------------")
end

function MDP(name; rng = Random.GLOBAL_RNG, negative_reward = true, tabular = true, kwargs...)
    info = Dict()
    terminal_states = []
    reward = 0f0
    t = 0
    gamma = 0.99f0
    if name == "Corridor"
        n_states = 10
        n_actions = 2
        final_reward = 1f0
        push!(terminal_states, 11)
        state_transition_mat = reshape(
            [zeros(Float32, n_states + 1) for i = 1:(n_states + 1)*n_actions],
            (n_states + 1, n_actions),
        )

        for (s, a) in collect(Iterators.product(1:n_states, 1:n_actions))
            if s == 1 && a == 1
                state_transition_mat[s, a][s] = 1.0f0
            elseif s == n_states + 1
                state_transition_mat[s, a][s] = 1.0f0
            elseif a == 1
                state_transition_mat[s, a][s - 1] = 1.0f0
            elseif a == 2
                state_transition_mat[s, a][s + 1] = 1.0f0
            end
        end
        reward_mat =
            Float32.(
                reshape(
                    -zeros(Float32, (n_states + 1) * n_actions),
                    (n_states + 1, n_actions),
                ),
            )/n_actions
        if negative_reward
            reward_mat .-= final_reward
        end
        reward_mat[n_states, n_actions] += final_reward

        initial_state = 1

    elseif name == "CausalCorridor"
        n_states = 10
        n_actions = 2
        loc_obstacle = Int(ceil(n_states/2))
        terminal_states = [n_states + 1, loc_obstacle]
        final_reward = 1f0
        state_transition_mat = reshape(
            [zeros(Float32, n_states + 1) for i = 1:(n_states + 1)*n_actions],
            (n_states + 1, n_actions),
        )

        reward_mat =
            Float32.(
                reshape(
                    -ones(Float32, (n_states + 1) * n_actions),
                    (n_states + 1, n_actions),
                ),
            )

        for (s, a) in collect(Iterators.product(1:n_states, 1:n_actions))
            if s in terminal_states
                state_transition_mat[s, a][s] = 1.0f0
            elseif s == n_states
                state_transition_mat[s, a][s+1] = 1.0f0
            elseif a == 1
                state_transition_mat[s, a][s + 2] = 1.0f0
                if s + 2 == loc_obstacle
                    reward_mat[s,a] = -10.0f0
                end
            elseif a == 2
                state_transition_mat[s, a][s + 1] = 1.0f0
                if s + 1 == loc_obstacle
                    reward_mat[s,a] = -10.0f0
                end
            end
        end


        for s in terminal_states
            reward_mat[s, :] .= 0
        end

        initial_state = 1
        info["loc_obstacle"] = loc_obstacle

    elseif name == "10Corridor"
        n_states = 10
        n_actions = 10
        push!(terminal_states, 11)

        state_transition_mat = reshape(
            [zeros(Float32, n_states + 1) for i = 1:(n_states + 1)*n_actions],
            (n_states + 1, n_actions),
        )

        for (s, a) in collect(Iterators.product(1:(n_states + 1), 1:n_actions))
            if s == a
                state_transition_mat[s, a][s+1] = 1.0f0
            elseif s == n_states + 1
                state_transition_mat[s, a][s] = 1.0f0
            elseif s > 1 && a == n_actions - s + 1
                state_transition_mat[s, a][s-1] = 1.0f0
            else
                state_transition_mat[s, a][s] = 1.0f0
            end
        end

        reward_mat =
            Float32.(
                reshape(
                    -zeros(Float32, (n_states + 1) * n_actions),
                    (n_states + 1, n_actions),
                ),
            )/n_actions
        # reward_mat -= reshape(rand(MersenneTwister(1), (n_states + 1)*n_actions), (n_states + 1, n_actions))
        final_reward = 1.0f0
        if negative_reward
            reward_mat .-= final_reward
        end
        reward_mat[n_states, n_actions] += final_reward

        bump_reward = -1.0f0
        reward_mat[1, n_actions] += bump_reward
        reward_mat[end, :] .= 0f0

        initial_state = 1
    else
        error("Not a valid MDP.")
    end
    V = get_optimal_V(state_transition_mat, reward_mat, gamma)

    mdp = MDP(state_transition_mat, reward_mat, initial_state, terminal_states, initial_state, reward, false, t, V, gamma, name, info, tabular, rng)
    reset!(mdp)
    return mdp
end

function get_goal_states(env::AbstractMDP)
    return env.terminal_states
end

function (env::AbstractMDP)(a)
    env.reward = env.reward_mat[env.state, a]
    env.state = StatsBase.sample(env.rng, StatsBase.Weights(env.state_transition_mat[env.state, a]))
    env.t += 1
    env.done = env.state in env.terminal_states
    nothing
end

function reset!(env::AbstractMDP; agent_pos = :default)
    if agent_pos == :default
        agent_pos = env.initial_state
    end
    env.state = agent_pos
    env.t = 0
    env.reward = 0.0f0
    env.done = false
    nothing
end

function Random.seed!(env::AbstractMDP, seed)
end

function get_info(env::AbstractMDP)
    nothing
end

function get_state(env::AbstractMDP)
    return env.state
end

function get_obs(env::AbstractMDP)
    s = zeros(Bool, size(env.reward_mat)[1])
    state = get_state(env)
    s[state] = 1
    if env.tabular
        return s[get_valid_states(env)]
    else
        return Float32.(s[get_valid_states(env)])
    end
end

function get_terminal(env::AbstractMDP)
    return env.done
end

function get_reward(env::AbstractMDP)
    return env.reward
end

function get_actions(env::AbstractMDP)
    return Base.OneTo(size(env.reward_mat)[2])
end

function random_action(env::AbstractMDP)
    return rand(env.rng, get_actions(env))
end

function optimal_action(env::AbstractMDP, s)
    # s = argmax(s)
    a = get_optimal_action(env, s)[1][1]
end

function optimal_action(env::AbstractMDP)
    # s = argmax(get_state(env))
    a = get_optimal_action(env, s)[1][1]
end


###
### Dynamic Programming
###

function get_optimal_action(env::AbstractMDP)
    s_list = get_all_states(env)
    get_optimal_action(env, s_list)
end

function get_optimal_action(env::AbstractEnv, s)
    Q_s = get_optimal_Q(env,s)
    max_Q = maximum(Q_s, dims = 2)
    as = Vector{Vector{Int64}}()

    n_states = length(s)

    for i  = 1:n_states
        Q = Q_s[i, :]
        a = findall(Q .== max_Q[i])
        push!(as, a)
    end
    return as
end

function get_all_states(env::AbstractMDP)
    R = env.reward_mat
    n_states = size(R)[1]
    return collect(1:n_states)
end

function get_valid_nonterminal_states(env::AbstractMDP)
    R = env.reward_mat
    n_states = size(R)[1]
    S = collect(1:(n_states))
    return setdiff(S, env.terminal_states)
end

function get_valid_states(env::AbstractMDP)
    R = env.reward_mat
    n_states = size(R)[1]
    return collect(1:(n_states))
end

function get_optimal_Q(env::AbstractEnv)
    s_list = get_all_states(env)
    get_optimal_Q(env, s_list)
end

function get_optimal_Q(env::AbstractEnv, s_list::AbstractArray{Int64})
    vcat([get_optimal_Q(env, s) for s in s_list]...)
end

function get_optimal_Q(env::AbstractEnv, s::Int64)
    T = env.state_transition_mat
    R = env.reward_mat
    V = env.V

    n_states = size(R)[1]
    n_actions = size(R)[2]

    Q_s = hcat([R[s, a] + env.gamma*sum(T[s, a].*V) for a in 1:n_actions]...)
    return Q_s
end

function get_optimal_V(env::AbstractEnv; tol = 1e-14)
    T = env.state_transition_mat
    R = env.reward_mat
    gamma = env.gamma
    get_optimal_V(T,R,gamma, tol = tol)
end

function get_optimal_V(T,R,gamma; tol = 1e-8)
    n_states = size(R)[1]
    n_actions = size(R)[2]

    V = zeros(Float32, n_states)
    Vnew = reshape(maximum(hcat([[transpose(T[s,a])*(R[s, a] .+ 0.99f0*V) for a = 1:n_actions] for s = 1:n_states]...), dims = 1), :)
    error = sum((V - Vnew).^2)
    V = copy(Vnew)

    step = 1
    while error > tol
        Vnew = reshape(maximum(hcat([[transpose(T[s,a])*(R[s, a] .+ 0.99f0*V) for a = 1:n_actions] for s = 1:n_states]...), dims = 1), :)
        error = sum((V - Vnew).^2)
        V = copy(Vnew)
        step += 1
    end
    # println("Step: ", step, " | Error: ", error)
    V
end

function int2xy(env::AbstractMDP, x; tuple = false)
    return CartesianIndex(1)
end
