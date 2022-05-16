
mutable struct ContextualMDP{M} <: AbstractContextualMDP{M}
    action_space::Base.OneTo{Int64}
    observation_space::Space{Array{Interval{:closed,:closed,Float32},1}}
    state_space::Base.OneTo{Int64}
    obs::AbstractArray{Float32,1}
    state::Int64
    action::Int64
    reward::Float32
    done::Bool
    t::Int
    n_actions::Int
    xs::AbstractArray
    ys::AbstractArray
    inds::AbstractArray{Array{Int64,1},1}
    train_size::Integer
    MDP::M
    eval_list
    corruption_rate::Float32
    dataset::String
    name::String
    mode::Symbol
    rng::AbstractRNG
end

function get_dataset(dataset, mode, rng, corruption_rate, train_size)
    if dataset == "MNIST"
        if mode == :test
            name = "MNIST_test"
            xs, ys = MLDatasets.MNIST.testdata()
            L = length(ys)
            L = Int(floor(L/2))
            inds = Random.shuffle(rng, 1:L)
            inds = 1:L
            xs = xs[:,:,inds]
            ys = ys[inds]

        elseif mode == :train
            name = "MNIST_train"
            xs, ys = MLDatasets.MNIST.traindata()
            L = length(ys)
            inds = Random.shuffle(rng, 1:L)
            inds = 1:L
            xs = xs[:, :, inds]
            ys = ys[inds]

        elseif mode == :validation
            name = "MNIST_validation"
            xs, ys = MLDatasets.MNIST.testdata()
            L_max = length(ys)
            L = Int(floor(L_max/2))
            inds = Random.shuffle(rng, L+1:L_max)
            inds = L+1:L_max
            xs = xs[:,:,inds]
            ys = ys[inds]
        end
        xs = reshape(xs, (:, L))

    elseif dataset == "MNISTColor"
        if mode == :test
            color_id = StatsBase.sample(rng, [1,2], StatsBase.weights([0.9, 0.1]))
            name = "MNIST_test"
            xs, ys = MLDatasets.MNIST.testdata()
            L = length(ys)
            L = Int(floor(L/2))
            inds = Random.shuffle(rng, 1:L)
            inds = 1:L
            xs = xs[:,:,inds]
            ys = ys[inds]

        elseif mode == :train
            color_id = StatsBase.sample(rng, [1,2], StatsBase.weights([0.1, 0.9]))
            name = "MNIST_train"
            xs, ys = MLDatasets.MNIST.traindata()
            L = length(ys)
            inds = Random.shuffle(rng, 1:L)
            inds = 1:L
            xs = xs[:, :, inds]
            ys = ys[inds]

        elseif mode == :validation
            color_id = StatsBase.sample(rng, [1,2], StatsBase.weights([0.25, 0.75]))
            name = "MNIST_validation"
            xs, ys = MLDatasets.MNIST.testdata()
            L_max = length(ys)
            L = Int(floor(L_max/2))
            inds = Random.shuffle(rng, L+1:L_max)
            inds = L+1:L_max
            xs = xs[:,:,inds]
            ys = ys[inds]
        end

        xs_r = zeros(Float32, (size(xs)[1:end-1]...,3,size(xs)[end]))
        xs_g = zeros(Float32, (size(xs)[1:end-1]...,3,size(xs)[end]))
        xs_b = zeros(Float32, (size(xs)[1:end-1]...,3,size(xs)[end]))
        xs_r[:,:,1,:] = xs
        xs_g[:,:,2,:] = xs
        xs_b[:,:,3,:] = xs
        xs_r = reshape(xs_r, (:, L))
        xs_g = reshape(xs_g, (:, L))
        xs_b = reshape(xs_b, (:, L))
        xs = hcat([[xs_r[:, i], xs_g[:,i], xs_b[:,i]] for i = 1:L]...)

    elseif dataset == "CIFAR10"
        if mode == :test
            name = "CIFAR_test"
            xs, ys = MLDatasets.CIFAR10.testdata()
            L = length(ys)
            L = Int(floor(L/2))
            inds = Random.shuffle(rng, 1:L)
            xs = xs[:,:,inds]
            ys = ys[inds]

        elseif mode == :train
            name = "CIFAR_train"
            xs, ys = MLDatasets.CIFAR10.traindata()
            L = length(ys)
            inds = Random.shuffle(rng, 1:L)
            xs = xs[:, :, inds]
            ys = ys[inds]
        elseif mode == :validation
            name = "CIFAR_validation"
            xs, ys = MLDatasets.CIFAR10.testdata()
            L = length(xs)
            L = Int(floor(L/2))
            inds = Random.shuffle(rng, L+1:L_max)
            xs = xs[:,:,inds]
            ys = ys[inds]
        end
        L = length(ys)
        xs = reshape(xs, (:, L))
    else
        error("Invalid dataset for ContextualMDP")
    end
    ys = ys .+ 1
    xs = Float32.(xs)

    n_classes = maximum(ys)
    for i = 1:L
        if rand(rng) < corruption_rate
            ys[i] = rand(rng, 1:n_classes)
        end
    end

    inds = [collect(1:L)[ys.==i] for i = 1:n_classes]

    xs = [xs[:, i] for i = 1:L]

    if mode == :train
        inds = [ind[1:train_size] for ind in inds]
    end
    return xs, ys, inds, name
end

function ContextualMDP(
    dataset,
    MDP;
    T = Float32,
    corruption_rate = 0.0f0,
    n_actions = 10,
    mode = :train,
    train_size = 10,
    rng = Random.GLOBAL_RNG,
    skip_eval = false,
    kwargs...,
)
    xs, ys, inds, name = get_dataset(dataset, mode, rng, corruption_rate, train_size)

    L = length(ys)

    dim_obs = length(xs[1])
    num_obs = length(xs)
    n_states = length(get_state(MDP))

    action_space = get_actions(MDP)
    n_actions = action_space.stop
    obs_space = Space([0.0f0..1.0f0 for i = 1:dim_obs])
    state_space = Base.OneTo(n_states)


    eval_list = []
    CMDP = ContextualMDP{typeof(MDP)}(
        action_space::Base.OneTo,
        obs_space,
        state_space,
        zeros(T, dim_obs)::AbstractArray,
        1::Int,
        1::Int,
        0.0f0::Float32,
        false::Bool,
        0::Int,
        n_actions::Int,
        xs,
        ys,
        inds,
        train_size,
        MDP,
        eval_list,
        corruption_rate::Float32,
        dataset,
        name,
        mode,
        rng,
    )

    if !skip_eval
        CMDP.eval_list = get_list_for_metric(CMDP)
    end

    reset!(CMDP)
    CMDP
end

Base.show(io::IO, t::MIME"text/plain", env::ContextualMDP) = begin
    println()
    println("---------------------------")
    name = "ContextualMDP"
    L = length(name)
    println(name)
    println(join(["-" for i = 1:L+1]))

    print(io, "  env.n_actions: ")
    println(io, env.n_actions)

    print(io, "  env.corruption_rate: ")
    println(io, env.corruption_rate)
    println("---------------------------")
end

function env_mode!(env::ContextualMDP; mode = :test)
    rng = MersenneTwister(env.rng.seed)

    if mode == :no_nest
        return ContextualMDP(
            env.dataset,
            env.MDP,
            n_actions = env.n_actions,
            corruption_rate = env.corruption_rate,
            mode = :test,
            rng = rng,
            skip_eval = true,
        )
    else
        return ContextualMDP(
            env.dataset,
            env_mode!(env.MDP, mode = mode),
            n_actions = env.n_actions,
            corruption_rate = env.corruption_rate,
            mode = mode,
            rng = rng,
            skip_eval = true,
        )
    end
end

function reset!(env::ContextualMDP, s::Int64)
    reset!(env.MDP, agent_pos = int2xy(env.MDP, s))
    env.state = findall(get_state(env.MDP).==1)[1]
    obs = sample_obs(env)
    env.obs = obs

    env.t = 0
    env.reward = 0.0f0
    env.done = false
    nothing
end

function reset!(env::ContextualMDP)
    reset!(env.MDP)
    env.state = get_state(env.MDP)
    # env.state = findall(get_state(env.MDP).==1)[1]
    obs = sample_obs(env)
    env.obs = obs
    env.t = 0
    env.reward = 0.0f0
    env.done = false
    nothing
end

function get_obs(env::ContextualMDP)
    obs = env.obs
    if env.MDP.name == "CompassWorld"
        MDP_obs = get_obs(env.MDP)[1:12]
        obs = vcat(MDP_obs, obs)
    end
    return obs
end

function get_state(env::ContextualMDP)
    return env.state
end

function get_reward(env::ContextualMDP)
    return env.reward
end

function get_actions(env::ContextualMDP)
    return env.action_space
end

function get_terminal(env::ContextualMDP)
    return env.done
end

function Random.seed!(env::ContextualMDP, seed) end

function (env::ContextualMDP)(a::Int)
    @assert a in env.action_space
    env.action = a
    env.MDP(a)
    env.reward = get_reward(env.MDP)
    env.state = get_state(env.MDP)
    env.t += 1
    env.done = get_terminal(env.MDP)
    if !env.done
        env.obs = sample_obs(env)
    end
    nothing
end

function optimal_action(env::ContextualMDP)
    a = optimal_action(env.MDP)
    return a
end

function optimal_action(env::ContextualMDP, s)
    s = get_state(env)
    a = optimal_action(env.MDP, s)
    return a
end

#
# Compat
#

function int2xy(env::ContextualMDP, x)
    int2xy(env.MDP, x)
end

function xy2int(env::ContextualMDP, x)
    xy2int(env.MDP, x)
end

#
# Metric helpers
#

function forbidden_action(env::ContextualMDP{M}, s) where {M <: AbstractMDP}
    if s == 1
        action = 10
    else
        action = s - 1
    end
    return action
end

function forbidden_action(A::Int, s::Int, num_actions::Int)
    if num_actions == 10
        if s == 1
            action = 10
        else
            action = s - 1
        end
        return action
    else
        rng = MersenneTwister(s)
        optimal_a = A
        all_as = collect(1:num_actions)
        non_optimal_as = setdiff(all_as, optimal_a)
        return StatsBase.sample(rng, non_optimal_as)
    end
end

function forbidden_action(env::ContextualMDP{E}, s) where {E <: TabularGridWorld}
    rng = MersenneTwister(s)
    optimal_a = get_optimal_action(env.MDP, s)[1][1]
    all_as = collect(get_actions(env))
    non_optimal_as = setdiff(all_as, optimal_a)
    return StatsBase.sample(rng, non_optimal_as)
end

function default_action(env::ContextualMDP, state = nothing)
    if isnothing(state)
        state = env.state
    end

    action = forbidden_action(env, state)
    a = rand(env.rng, collect(1:env.n_actions)[1:end .!== action])
    p = 1/(env.n_actions - 1)
    return a, p
end

function sample_obs(env::ContextualMDP{M}, s::Int; n = 1) where {M<:AbstractMDP}
    obs_mat = []
    for _ = 1:n
        ind = StatsBase.sample(env.rng, env.inds[s])
        if env.dataset == "MNISTColor"
            if env.mode == :test
                color_id = StatsBase.sample(env.rng, [1, 2], StatsBase.weights([0.9, 0.1]))
            else
                color_id = StatsBase.sample(env.rng, [1, 2], StatsBase.weights([0.1, 0.9]))
            end
            x =  env.xs[ind][color_id]
        else
            x = env.xs[ind]
        end
        push!(obs_mat, x)
    end
    return hcat(obs_mat...)
end

function sample_obs(env::ContextualMDP{M}, s_list::Vector{Int}; n = 1) where {M}
    return hcat([sample_obs(env, s, n = n) for s in s_list]...)
end

function sample_obs(env::ContextualMDP{M}) where {M <: AbstractMDP}
    reshape(sample_obs(env, env.state), :)
end

function sample_obs(env::ContextualMDP{M}) where {M <: AbstractGridWorld}
    A = get_tilemap(env.MDP.env)[1,:,:]
    loc = findall(A .== 1)[1]
    x,y = loc.I
    ind_1 = rand(env.rng, env.inds[x])
    ind_2 = rand(env.rng, env.inds[y])
    x = env.xs[ind_1]
    y = env.xs[ind_2]
    return vcat(x,y)
end

function sample_obs(env::ContextualMDP{M}, s::Int64; n = 1, deterministic = true) where {M <: AbstractGridWorld}
    A = get_tilemap(env.MDP.env)[1,:,:]
    obs_mat = []
    I = isqrt(n)
    J = isqrt(n)
    if deterministic
        @assert length(env.inds[1]) >= I
    end

    n_orientations = typeof(env.MDP.env) <: AbstractGridWorldDirected ? 4 : 1
    for i  = 1:I
        for j = 1:J
            X,Y = size(A)
            n_states = size(env.MDP.reward_mat)[1]
            # println(n_orientations)
            temp = zeros(Float32, n_states)
            temp[s] = 1f0
            temp = reshape(temp, (X,Y,n_orientations))
            loc = findall(temp .== 1)[1]
            x,y = loc.I
            if deterministic
                ind_1 = env.inds[x][i]
                ind_2 = env.inds[y][j]
            else
                ind_1 = rand(env.rng, env.inds[x])
                ind_2 = rand(env.rng, env.inds[y])
            end
            x = env.xs[ind_1]
            y = env.xs[ind_2]
            push!(obs_mat, vcat(x,y))
        end
    end
    return hcat(obs_mat...)
end

function get_min_samples(env::ContextualMDP{M}) where {M <: AbstractMDP}
    min_samples = 2500
    min_train_ind = minimum([length(ind) for ind in env.inds])
    min_len =  minimum([min_samples, min_train_ind])
    return min_len
end

function get_min_samples(env::ContextualMDP{M}) where {M <: AbstractGridWorld}
    min_samples = 10
    min_train_ind = minimum([length(ind) for ind in env.inds])
    # return minimum([min_samples, min_train_ind])^2
    return 0
end

function get_list_for_metric(env)
    valid_states = get_valid_nonterminal_states(env.MDP)

    train_env = env_mode!(env, mode = :train)
    N_train = get_min_samples(train_env)

    test_env = env_mode!(env, mode = :test)
    N_test = get_min_samples(test_env)

    val_env = env_mode!(env, mode = :validation)
    N_val = get_min_samples(val_env)

    train_xs = []
    train_ys = []

    val_xs = []
    val_ys = []

    test_xs = []
    test_ys = []

    for s in valid_states
        train_x = sample_obs(train_env, s, n = N_train)
        train_y = [s for _ = 1:N_train]
        push!(train_xs, train_x)
        push!(train_ys, train_y)

        test_x = sample_obs(test_env, s, n = N_test)
        test_y = [s for _ = 1:N_test]
        push!(test_xs, test_x)
        push!(test_ys, test_y)

        val_x = sample_obs(val_env, s, n = N_val)
        val_y = [s for _ = 1:N_val]
        push!(val_xs, val_x)
        push!(val_ys, val_y)
    end

    list = (train = (train_xs, train_ys), test = (test_xs, test_ys), validation = (val_xs, val_ys))
    extras = nothing
    if env.MDP.name == "FourRooms"
        train_xs = []
        train_ys = []

        val_xs = []
        val_ys = []

        test_xs = []
        test_ys = []
        valid_wall_states = get_valid_wall_states(env.MDP)
        for s in valid_wall_states
            train_x = sample_obs(train_env, s, n = N_train)
            train_y = [s for _ = 1:N_train]
            push!(train_xs, train_x)
            push!(train_ys, train_y)

            test_x = sample_obs(test_env, s, n = N_test)
            test_y = [s for _ = 1:N_test]
            push!(test_xs, test_x)
            push!(test_ys, test_y)

            val_x = sample_obs(val_env, s, n = N_val)
            val_y = [s for _ = 1:N_val]
            push!(val_xs, val_x)
            push!(val_ys, val_y)
        end
        extras = (wall_train = (train_xs, train_ys), wall_test = (test_xs, test_ys), wall_validation = (val_xs, val_ys))
    end
    return list, extras
end
