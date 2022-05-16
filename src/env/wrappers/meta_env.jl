abstract type AbstractMetaEnv <: AbstractEnv end

mutable struct MetaEnv{E} <: AbstractMetaEnv
    env::E
    buffer
    agent
    batch_size
    base_state
    meta_state
    encode_dim
    state_representation
    init
end

Base.show(io::IO, t::MIME"text/plain", env::AbstractMetaEnv) = begin
    println()
    println("---------------------------")
    name = "MetaEnv"
    L = length(name)
    println(name)
    println(join(["-" for i = 1:L+1]))
    print(io, "  env.env: ")
    println(io, env.env)
    println("---------------------------")
end

function MetaEnv(base_env::AbstractEnv, encode_dim; kwargs...)
    base_state = get_state(base_env)
    meta_state = zeros(Float32, encode_dim)
    batch_size = 128
    buffer = nothing
    agent = nothing
    init = true
    return MetaEnv{typeof(base_env)}(base_env, buffer, agent, batch_size, base_state, meta_state, encode_dim, "epbuffer", init)
end

function Random.seed!(env::AbstractMetaEnv, seed)
end

function env_mode!(env::AbstractMetaEnv; mode = :default)
end

function reset!(env::AbstractMetaEnv)
    @assert !isnothing(env.agent)
    @assert !isnothing(env.buffer)
    reset!(env.env)
end

function get_state(env::AbstractMetaEnv)
    get_state(env.env)
end

function get_obs(env::AbstractMetaEnv)

    base_state = get_obs(env.env)
    aux_dim = size(base_state)[1]
    in_dim = size(base_state)[1]
    out_dim = num_actions(env.env)
    t_dim = 0
    T = env.batch_size
    meta_info  = [base_state, aux_dim, in_dim, out_dim, t_dim, T]
    if env.init
        env2 = deepcopy(env.env)
        dummy_max_steps = 200
        ep = generate_episode(env2, nothing, policy = :random, max_steps = dummy_max_steps);
        while length(ep) < env.batch_size
            ep2 = generate_episode(env2, nothing, policy = :random, max_steps = dummy_max_steps);
            push!(ep, ep2...)
        end
        aux_only = hcat([e.o for e in ep[1:env.batch_size]]...)
        dummy_metastate = zeros(Float32, (env.encode_dim, env.batch_size))
        dummy_input = cat(aux_only, dummy_metastate, dims = 1)
        # output = env.agent.subagents[1](dummy_input)

        out_dim = num_actions(env.env)
        output = zeros(Float32, (out_dim, env.batch_size))
        metastate = cat(aux_only, output, dims = 1)
    else
        StatsBase.sample(env.buffer, env.batch_size)
        _, ob, _, _, _, _, _, _ = get_batch(env.buffer)
        aux_only = get_aux(ob[:,1,:])
        dummy_metastate = zeros(Float32, (env.encode_dim, env.batch_size))
        dummy_input = cat(aux_only, dummy_metastate, dims = 1)
        # dummy_input = reshape(dummy_input, :)
        # dummy_input = vcat(dummy_input, metainfo)
        output = env.agent.subagents[1].model.f(dummy_input |> env.agent.device) |> Flux.cpu
        metastate = cat(aux_only, output, dims = 1)
    end

    if !isnothing(env.agent) && !isnothing(env.buffer)
        if curr_size(env.buffer) > env.batch_size
            env.init = false
        end
    end

    metastate = reshape(metastate, :)
    return vcat(metastate, meta_info...)
    # return base_state
end

function get_actions(env::AbstractMetaEnv)
    return get_actions(env.env)
end

function get_reward(env::AbstractMetaEnv)
    return get_reward(env.env)
end

function get_terminal(env::AbstractMetaEnv)
    return get_terminal(env.env)
end

function get_info(env::AbstractMetaEnv)
end

function random_action(env::AbstractMetaEnv)
    return random_action(env.env)
end

function (env::AbstractMetaEnv)(a)
    env.env(a)
end

