# global my_mujoco_rng = MersenneTwister(1)

function (env::AbstractMuJoCoEnvironment)(a)
    a = mujoco_action(env, a)
    setaction!(env, a)
    step!(env)
    nothing
end

function preprocess_action(env::AbstractMuJoCoEnvironment, a)
    a
end

function state_size(env::AbstractMuJoCoEnvironment)
    return size(env.obsspace)
end

function num_actions(env::AbstractMuJoCoEnvironment)
    # TODO
    length(LyceumMuJoCo.actionspace(env))^3
    # length(random_action(env))
end

function Random.seed!(env::AbstractMuJoCoEnvironment, seed)
    # global my_mujoco_rng = MersenneTwister(seed)
end

function action_embedding(env::AbstractMuJoCoEnvironment, a)
    temp_a = Int64.(a) .+ 2
    space = get_actions(env)
    N = length(space)
    one_dimarray = 1:N^3
    n_d = reshape(one_dimarray, ntuple(x -> 3, N))
    n_d[temp_a...]
end

function mujoco_action(env::AbstractMuJoCoEnvironment, a)
    space = get_actions(env)
    N = length(space)
    one_dimarray = 1:N^3
    N_dimarray = reshape(one_dimarray, ntuple(x -> N, 3))
    inds = findall(x -> x == a, N_dimarray)
    Float64.(getindex.(inds, 1:N)) .- 2
end

function get_obs(env::AbstractMuJoCoEnvironment)
    return Float32.(getobs(env))
end

function get_reward(env::AbstractMuJoCoEnvironment)
    return getreward(env)
end

function get_actions(env::AbstractMuJoCoEnvironment)
    LyceumMuJoCo.actionspace(env)
end

function get_terminal(env::AbstractMuJoCoEnvironment)
    return isdone(env)
end

function reset!(env::AbstractMuJoCoEnvironment)
    # TODO using their built in randreset.
    # Implement rng handling
    LyceumMuJoCo.randreset!(my_mujoco_rng, env)
end

function random_action(env::AbstractMuJoCoEnvironment)
    N = num_actions(env)
    rand(my_mujoco_rng, 1:N)
    #Continuous
    #2*(rand(get_actions(env)) .- 0.5)
    #TODO I think + need to manage rng
    # Float32.(rand(get_actions(env)))
end

function action_type(env::AbstractMuJoCoEnvironment)
    #typeof(getaction(env))
    Int64
    #Float64
end
