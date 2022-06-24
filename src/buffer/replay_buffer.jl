import StatsBase: mean, std

abstract type AbstractBuffer{S,O,A} end
abstract type AbstractTransitionBuffer{S,O,A} <: AbstractBuffer{S,O,A} end

struct Experience{S,O,A}
    s::S
    o::O
    a::A
    p::Float32
    r::Float32
    sp::S
    op::O
    done::Bool
    info::Any
    env_t::Float32
    agent_t::Float32
end

function Experience(s,o,a,p,r,sp,op,done,info = nothing, env_t = 1f0, agent_t = 1f0)
    S = typeof(s)
    O = typeof(o)
    A = typeof(a)
    return Experience{S,O,A}(s,o,a,p,r,sp,op,done,info)
end

is_full(buffer::AbstractBuffer) = curr_size(buffer) == buffer.max_size

max_size(buffer::AbstractBuffer) = buffer.max_size

function curr_size(buffer::AbstractBuffer; online = false)
    if online # Don't count the episode being interacted with
        idx = buffer._buffer_idx
        if idx == 1
            return sum(buffer._episode_lengths[2:end])
 else
            lower = sum(buffer._episode_lengths[1:idx - 1])
            upper = sum(buffer._episode_lengths[idx + 1:end])
            return lower + upper
        end
    else
        return sum(buffer._episode_lengths)
    end
end

function add_exp!(buffer::AbstractBuffer{S,O,A}, exp::Experience) where {S,O,A}
    push!(buffer._episode, exp)
    return nothing
end

function populate_replay_buffer!(
    buffer::AbstractBuffer,
    env,
    agent;
    policy = :agent,
    num_episodes::Int,
    greedy = false,
    max_steps,
)
    if typeof(policy) <: Vector
        for p in policy
            num_eps = Int(floor(num_episodes/length(policy)))
            populate_replay_buffer!(buffer, env, agent; policy = p, num_episodes = num_eps, greedy = greedy, max_steps = max_steps)
        end
    else
        for _ = 1:num_episodes
            @assert buffer._episode_idx == 1
            episode = generate_episode(
                env,
                agent,
                policy = policy,
                greedy = greedy,
                max_steps = max_steps,
            )
            buffer._episode = episode
            buffer._episodes[buffer._buffer_idx] = episode
            buffer._episode_idx = length(episode) + 1
            buffer._episode_lengths[buffer._buffer_idx] = length(episode)
            finish_episode(buffer)
        end
    end

    while curr_size(buffer) < buffer.batch_size && get_num_episodes(buffer) !== buffer.max_num_episodes
        @assert buffer._episode_idx == 1
        episode = generate_episode(
            env,
            agent,
            policy = policy,
            greedy = greedy,
            max_steps = max_steps,
        )
        buffer._episode = episode
        buffer._episodes[buffer._buffer_idx] = episode
        buffer._episode_idx = length(episode) + 1
        buffer._episode_lengths[buffer._buffer_idx] = length(episode)
        finish_episode(buffer)
    end
    @assert curr_size(buffer) >= buffer.batch_size
    return nothing
end

function state_collisions(buffer::AbstractBuffer)
    idxs = buffer._episode_lengths .!== 0
    episodes = buffer._episodes[idxs]
    i = 1
    counts = []
    for episode in episodes
        j = 1
        for exp in episode
            first_hit = true
            o0 = copy(exp.o)
            count_collisions = 0
            j_next = j + 1
            for exp in episode[j+1:end]
                if o0 == exp.o
                    if first_hit
                        println("O0: ", i, ", ", j)
                        first_hit = false
                    end

                    count_collisions += 1
                    println("Collision at: ", i, ", ", j_next)
                end
                j_next += 1
            end

            i_next = 1
            for episode_next in episodes[i+1:end]
                j_next = 1
                for exp in episode_next
                    if o0 == exp.o
                        if first_hit
                            println("O0: ", i, ", ", j)
                            first_hit = false
                        end
                        count_collisions += 1
                        println("Collision at: ", i_next, ", ", j_next)
                    end
                    j_next += 1
                end
                i_next += 1
            end
            j += 1
            push!(counts, count_collisions)
        end
        i += 1
    end
    return counts
end

function montecarlo_episode(ep::Vector{Experience{S,O,A}}, gamma) where {S,O,A}
    new_ep = Experience{S,O,A}[]
    G = get_returns(ep, gamma)
    i = 1
    for g in G
        e = ep[i]
        exp = Experience(e.s, e.o, e.a, e.p, g, e.sp, e.op, e.done, e.info)
        push!(new_ep, exp)
        i += 1
    end
    return new_ep
end

function montecarlo_buffer(buffers::Vector{B}, gamma) where {B <: AbstractBuffer}
    mc_buffers = Vector{B}()
    for buffer in buffers
        push!(mc_buffers, montecarlo_buffer(buffer, gamma))
    end
    return mc_buffers
end

function montecarlo_buffer(buffer::AbstractBuffer, gamma)
    new_buffer = deepcopy(buffer)
    i = 1
    idxs = new_buffer._episode_lengths .!== 0
    episodes = new_buffer._episodes[idxs]
    for episode in episodes
        mc_ep = montecarlo_episode(episode, gamma)
        new_buffer._episodes[i] = mc_ep
        i += 1
    end
    return new_buffer
end

function get_batch(buffers::Vector{B}) where {B <: AbstractBuffer}
    batches = []
    for buffer in buffers
        b = get_batch(buffer)
        push!(batches, b)
    end
    num_entries = length(batches[1])
    num_buffers = length(buffers)
    return [cat([batches[i][j] for i = 1:num_buffers]..., dims = 3) for j = 1:num_entries]
end

function get_batch(buffer::AbstractBuffer)
    H = size(buffer._r_batch)[2]
    B = size(buffer._r_batch)[3]

    s = reshape(hcat(buffer._s_batch...), (:, H, B))
    o = reshape(hcat(buffer._o_batch...), (:, H, B))

    a = reshape(hcat(buffer._a_batch...), (:, H, B))
    p = buffer._p_batch
    G = buffer._r_batch

    sp = reshape(hcat(buffer._sp_batch...), (:, H, B))
    op = reshape(hcat(buffer._op_batch...), (:, H, B))
    ap = reshape(hcat(buffer._ap_batch...), (:, H, B))

    done = buffer._done_batch
    info = buffer._info_batch

    agent_t = buffer._agentt_batch
    env_t = buffer._envt_batch
    curr_t = buffer._currt_batch

    return s, o, a, p, G, sp, op, ap, done, info
end

function get_batch(buffer, subagent)
    action_encoder = subagent.submodels.action_encoder
    device = subagent.device
    gamma = subagent.gamma
    t = subagent.update_count

    s, o, a, p, r, sp, op, done, info, discounted_reward_sum, mask, ap, maskp = get_batch(buffer, gamma, device, action_encoder, t)

    return s, o, a, p, r, sp, op, done, info, discounted_reward_sum, mask, ap, maskp
end

function get_batch(buffer, gamma::Float32, device, action_encoder, t = 1)
    s, o, a, p, r, sp, op, ap, done, info = get_batch(buffer)
    discounted_reward_sum = nstep_returns(gamma, device, r, all = true)
    mask = action_encoder(a) |> device
    maskp = action_encoder(ap) |> device
    data = s, o, a, p, r, sp, op, done, info, discounted_reward_sum, mask, ap, maskp
    return data |> device
end

mutable struct TransitionReplayBuffer{S,O,A} <: AbstractBuffer{S,O,A}
    max_num_episodes::Int64
    max_episode_length::Int64
    batch_size::Int64

    _buffer_idx::Int64
    _episode_idx::Int64

    history_window::Int64
    predict_window::Int64

    _episodes::Vector{Vector{Experience{S,O,A}}}
    _episode::Vector{Experience{S,O,A}}
    _episode_lengths::Vector{Int64}

    _baseline::AbstractArray{Float32,2}
    _indicies::Vector{Tuple{Int64,Int64}}

    _s_batch::AbstractArray{S,2}
    _o_batch::AbstractArray{O,2}
    _a_batch::AbstractArray{A,2}
    _p_batch::AbstractArray{Float32,3}
    _r_batch::AbstractArray{Float32,3}
    _sp_batch::AbstractArray{S,2}
    _op_batch::AbstractArray{O,2}
    _ap_batch::AbstractArray{A,2}
    _done_batch::AbstractArray{Bool,3}
    _info_batch::AbstractArray{Any,3}
    _envt_batch::AbstractArray{Float32,3}
    _agentt_batch::AbstractArray{Float32,3}
    _currt_batch::AbstractArray{Float32,3}

    gamma::Float32

    overlap::Bool
    bootstrap::Bool
    rng::AbstractRNG
    name::String
end

Base.show(io::IO, buffer::TransitionReplayBuffer) = begin
    println()
    println("---------------------------")
    L = length(buffer.name)
    println(buffer.name)
    println(join(["-" for i = 1:L+1]))

    println("max_num_episodes: ", buffer.max_num_episodes)
    println("max_episode_length: ", buffer.max_episode_length)
    println("_buffer_idx: ", buffer._buffer_idx)
    println("_episode_idx: ", buffer._episode_idx)
    println("episode_lengths: ", buffer._episode_lengths)
    println("history_window: ", buffer.history_window)
    println("predict_window: ", buffer.predict_window)
    println("bootstrap: ", buffer.bootstrap)
    println("overlap: ", buffer.overlap)
    println("---------------------------")
end

function TransitionReplayBuffer(
    env,
    max_num_episodes::Int64,
    max_episode_length::Int64,
    batch_size::Int64,
    gamma;
    bootstrap::Bool = true,
    overlap::Bool = true,
    history_window::Int64 = 1,
    predict_window::Int64 = 0,
    rng::AbstractRNG = GLOBAL_RNG,
    name,
)
    dA = length(random_action(env))
    S = typeof(get_state(env))
    O = typeof(get_obs(env))
    A = typeof(random_action(env))

    @assert history_window > 0
    @assert predict_window >= 0

    H = history_window + predict_window

    _episodes = Vector{Vector{Experience{S,O,A}}}(undef, max_num_episodes)
    _episode = Vector{Experience{S,O,A}}(undef, max_episode_length)
    episode_lengths = zeros(Int, max_num_episodes)

    _baseline = zeros(Float32, H, 1) # [?] useless
    _indices = Vector{Tuple{Int64,Int64}}(undef, batch_size) # [?] useless

    _s_batch = Array{S}(undef, H, batch_size)
    _o_batch = Array{O}(undef, H, batch_size)
    _a_batch = Array{A}(undef, H, batch_size)
    _p_batch = Array{Float32}(undef, 1, H, batch_size)
    _r_batch = Array{Float32}(undef, 1, H, batch_size)
    _sp_batch = Array{S}(undef, H, batch_size)
    _op_batch = Array{O}(undef, H, batch_size)
    _ap_batch = Array{A}(undef, H, batch_size)
    _done_batch = Array{Bool}(undef, 1, H, batch_size)
    _info_batch = Array{Any}(undef, 1, H, batch_size)
    _agentt_batch = Array{Float32}(undef, 1, H, batch_size)
    _envt_batch = Array{Float32}(undef, 1, H, batch_size)
    _currt_batch = Array{Float32}(undef, 1, H, batch_size)

    buffer =  TransitionReplayBuffer{S,O,A}(
        max_num_episodes,
        max_episode_length,
        batch_size,
        1,
        1,
        history_window,
        predict_window,
        _episodes,
        _episode,
        episode_lengths,
        _baseline,
        _indices,
        _s_batch,
        _o_batch,
        _a_batch,
        _p_batch,
        _r_batch,
        _sp_batch,
        _op_batch,
        _ap_batch,
        _done_batch,
        _info_batch,
        _agentt_batch,
        _envt_batch,
        _currt_batch,
        gamma,
        overlap,
        bootstrap,
        rng,
        name,
    )

    if typeof(env) <: MetaEnv
        env.buffer = buffer
    end

    return buffer
end

function init_batches(buffer::TransitionReplayBuffer{S,O,A}, batch_size) where {S,O,A}

    H = buffer.history_window + buffer.predict_window

    buffer._s_batch = Array{S}(undef, H, batch_size)
    buffer._o_batch = Array{O}(undef, H, batch_size)
    buffer._a_batch = Array{A}(undef, H, batch_size)
    buffer._p_batch = Array{Float32}(undef, 1, H, batch_size)
    buffer._r_batch = Array{Float32}(undef, 1, H, batch_size)
    buffer._sp_batch = Array{S}(undef, H, batch_size)
    buffer._op_batch = Array{O}(undef, H, batch_size)
    buffer._ap_batch = Array{A}(undef, H, batch_size)
    buffer._done_batch = Array{Bool}(undef, 1, H, batch_size)
    buffer._info_batch = Array{Any}(undef, 1, H, batch_size)
    buffer._envt_batch = Array{Float32}(undef, 1, H, batch_size)
    buffer._agentt_batch = Array{Float32}(undef, 1, H, batch_size)
    buffer._currt_batch = Array{Float32}(undef, 1, H, batch_size)
end

function reset_experience!(buffer::TransitionReplayBuffer{S,O,A}) where {S,O,A}
    buffer._episode_idx = 1
    buffer._buffer_idx = 1
    buffer._episodes = Vector{Vector{Experience{S,O,A}}}(undef, buffer.max_num_episodes)
    buffer._episode = Vector{Experience{S,O,A}}(undef, buffer.max_episode_length)
    buffer._episodes[1] = buffer._episode
    buffer._episode_lengths = zeros(Int, buffer.max_num_episodes)
    init_batches(buffer, buffer.batch_size)
    return 0
end

function add_exp!(
    buffer::TransitionReplayBuffer{S,O,A},
    exp::Experience{S,O,A},
) where {S,O,A}

    if buffer._episode_idx == 1
        buffer._episodes[buffer._buffer_idx] = buffer._episode
        buffer._episode_lengths[buffer._buffer_idx] = 0
    end

    buffer._episode[buffer._episode_idx] = exp
    buffer._episode_lengths[buffer._buffer_idx] += 1
    buffer._episode_idx += 1
    return nothing
end

function finish_episode(
    buffer::TransitionReplayBuffer{S,O,A};
    bootstrap = true,
) where {S,O,A}
    buff_ep = buffer._episode
    bootstrap = buffer.bootstrap
    if !bootstrap
        buffer._episodes[buffer._buffer_idx] =
            montecarlo_episode(buff_ep[1:buffer._episode_idx-1], buffer.gamma)
    else
        buffer._episodes[buffer._buffer_idx] = buff_ep[1:buffer._episode_idx-1]
    end
    buffer._buffer_idx = mod1((buffer._buffer_idx + 1), buffer.max_num_episodes)
    buffer._episode = Vector{Experience{S,O,A}}(undef, buffer.max_episode_length)
    buffer._episode_idx = 1
    return nothing
end

function convert_sample_indices(buffer, sample_indices)
    H = buffer.history_window + buffer.predict_window - 1
    episode_inds = cumsum(max.(buffer._episode_lengths, 0))  #TODO This heavily skews the episodes sampled for large H
    sample_idxs = Vector{Tuple{Int64,Int64}}()
    for (i, idx) in enumerate(sample_indices)
        ep_idx = findfirst(x -> x >= idx, episode_inds)
        if ep_idx > 1
            idx = idx - episode_inds[ep_idx-1]
        end
        push!(sample_idxs, (ep_idx, idx))
    end
    return sample_idxs
end

function get_num_episodes(buffer::TransitionReplayBuffer)
    sum(buffer._episode_lengths .!== 0)
end

function sample_episodes(buffer::TransitionReplayBuffer)
    number_of_episodes = get_num_episodes(buffer)

    if number_of_episodes < buffer.batch_size
        batch_size = number_of_episodes
    else
        batch_size = buffer.batch_size
    end

    sample_indices =
        StatsBase.sample(buffer.rng, 1:number_of_episodes, batch_size, replace = false)
end

function sample_with_inds(buffer, sample_indices; bootstrap = true)
    init_batches(buffer, length(sample_indices))
    H = buffer.history_window + buffer.predict_window - 1
    for (i, idxs) in enumerate(sample_indices)
        ep_idx = idxs[1]
        idx = idxs[2]
        ep = buffer._episodes[ep_idx][idx:end]
        if !bootstrap
            # ep = montecarlo_episode(ep, 0.99f0)
        end

        if length(ep) < H + 2
            pad_episode!(ep, H + 2)
        else
            ep = ep[1:H+2]
        end

        b = zeros(Float32, 1, H + 1)
        if bootstrap
            fill_buffer!(buffer, ep, i)
        else
            fill_buffer!(buffer, ep, i, b)
        end
    end
end

function StatsBase.sample(buffer::Vector{B}; batch_size = nothing, bootstrap = true) where {B <: AbstractBuffer}
    L = length(buffer)
    for b in buffer
        batch_size = Int(floor(b.batch_size/L))
        StatsBase.sample(b, batch_size; bootstrap = bootstrap)
    end
end

function StatsBase.sample(buffer::TransitionReplayBuffer; batch_size = nothing, bootstrap = true)
    StatsBase.sample(buffer, batch_size; bootstrap = bootstrap)
end

function StatsBase.sample(
    buffer::TransitionReplayBuffer{S,O,A},
    batch_size;
    baseline = true,
    bootstrap = true,
    ind_range = nothing,
) where {S,O,A}

    bootstrap = buffer.bootstrap
    if batch_size == nothing
        batch_size = buffer.batch_size
    end

    if isnothing(ind_range)
        ind_range = 1:length(buffer._episode_lengths)
    end


    H = buffer.history_window + buffer.predict_window - 1

    b = max.(buffer._episode_lengths .- H, 0) #TODO This can skew the episodes sampled for large H
    bs = []
    starts = [0,  cumsum(buffer._episode_lengths)...]
    for i = 1:length(buffer._episode_lengths)
        ep_len = buffer._episode_lengths[i]
        start = starts[i] + 1
        end_ind = start + ep_len - H - 1
        if i == buffer._buffer_idx
            end_ind -= 1
        end
        if i == buffer._buffer_idx && !bootstrap
        else
            if i in ind_range
                push!(bs, collect(start:end_ind)...)
            end
        end
    end

    buffer_size = buffer._episode_lengths
    # list_of_p = collect(1:sum(buffer_size))
    list_of_p = bs
    if batch_size == -1 # -1 is full_batch
        # batch_size = sum(buffer_size)
        sample_indices_temp = list_of_p
        batch_size = length(sample_indices_temp)
    else
        sample_indices_temp =
            StatsBase.sample(buffer.rng, list_of_p, batch_size, replace = false)
    end


    sample_indices = sort(convert_sample_indices(buffer, sample_indices_temp), rev = true)

    sample_with_inds(buffer, sample_indices; bootstrap = bootstrap)

    buffer._indicies = sample_indices
    return buffer._s_batch,
    buffer._o_batch,
    buffer._a_batch,
    buffer._p_batch,
    buffer._r_batch,
    buffer._sp_batch,
    buffer._op_batch,
    buffer._ap_batch,
    buffer._done_batch,
    buffer._info_batch
end

function pad_episode!(ep, len)
    s = ep[end].sp
    o = ep[end].op
    r = 0.0f0
    as = [e.a for e in ep]
    a = rand(as)
    done = true
    p = 0.5f0
    while length(ep) < len
        push!(ep, Experience(s, o, a, p, r, s, o, done))
    end
    return ep
end

function fill_buffer!(buffer, epi, index, b = nothing)
    ep = epi[1:end-1]
    buffer._s_batch[:, index] = map(p -> p.s, ep)
    buffer._o_batch[:, index] = map(p -> p.o, ep)
    buffer._a_batch[:, index] = map(p -> p.a, ep)
    buffer._p_batch[:, :, index] = hcat(map(p -> p.p, ep)...)
    if b === nothing
        buffer._r_batch[:, :, index] = hcat(map(p -> p.r, ep)...)
    else
        buffer._r_batch[:, :, index] = hcat(map(p -> p.r, ep)...) - b
    end
    buffer._sp_batch[:, index] = map(p -> p.sp, ep)
    buffer._op_batch[:, index] = map(p -> p.op, ep)
    buffer._ap_batch[:, index] = map(p -> p.a, epi[2:end])
    buffer._done_batch[:, :, index] = hcat(map(p -> p.done, ep)...)
    buffer._info_batch[:, :, index] = hcat(map(p -> p.info, ep)...)
end
