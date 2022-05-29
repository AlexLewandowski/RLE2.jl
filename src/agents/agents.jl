import Flux: gpu, cpu
import BSON
import CodecZstd

include("subagents.jl")
include("targets.jl")

""" An Agent is a functional object that stores a collection of subagents for
learning, a behavior policy, buffers and the environment.

Each subagent has a model that is being learned in addition to submodels.

Examples of submodels can include a target network. """
mutable struct Agent <: AbstractAgent
    subagents::AbstractArray{Subagent,1}
    buffers::Union{NamedTuple,Nothing}
    state_encoder#::Union{AbstractModel,Symbol,Function}
    action_encoder#::Union{AbstractModel,Symbol,Function}
    π_b#::Union{AbstractModel,Symbol,Function}
    max_agent_steps::Int64
    measurement_freq::Int64
    measurement_count::Int64
    measurement_funcs::AbstractArray
    measurement_dict::Dict
    device::Union{typeof(gpu),typeof(cpu)}
    rng::AbstractRNG
    name::String
end

function to_device!(agent::AbstractAgent, device = :default)
    if device == :default
        device = agent.device
    end
    to_device!(agent.state_encoder, device)
    to_device!(agent.action_encoder, device)
    for subagent in agent.subagents
        to_device!(subagent, device)
    end
    agent.device = device
    nothing
end

function to_device!(f, device = :default)
    @debug "Not sending to device for device: "*string(typeof(f))
end

function remove_undefs(buffer::AbstractBuffer)
    num_buffer_eps = get_num_episodes(buffer)
    if num_buffer_eps < buffer.max_num_episodes
        buffer._episodes = buffer._episodes[1:num_buffer_eps]
    end
    buffer._episode = []
end

function remove_undefs(agent::AbstractAgent)
    for buffer in agent.buffers
        remove_undefs(buffer)
    end
end

function undo_remove_undefs(buffer::AbstractBuffer)
    num_buffer_eps = get_num_episodes(buffer)
    if num_buffer_eps < buffer.max_num_episodes
        episodes = Vector{Vector{Experience{typeof(buffer).parameters...}}}(
            undef,
            buffer.max_num_episodes,
        )
        _episode = Vector{Experience{typeof(buffer).parameters...}}(
            undef,
            buffer.max_episode_length,
        )
        episodes[1:num_buffer_eps] = buffer._episodes
        buffer._episodes = episodes
        buffer._episode = _episode
    end
end


function undo_remove_undefs(agent::AbstractAgent)
    for buffer in agent.buffers
        undo_remove_undefs(buffer)
    end
end

function save_agent(agent, path = "", name = "")
    remove_undefs(agent)
    to_device!(agent, Flux.cpu)
    my_save_dict = Dict(:agent => agent)
    open(joinpath(path, name * "agent.bson.zstd"), "w") do fd
        stream = CodecZstd.ZstdCompressorStream(fd)
        BSON.@save(stream, my_save_dict)
        close(stream)
    end
    undo_remove_undefs(agent)
end

function load_agent(path = "", name = "")
    load_dict = nothing
    open(joinpath(path, name * "agent.bson.zstd"), "r") do fd
        stream = CodecZstd.ZstdDecompressorStream(fd)
        load_dict = BSON.load(stream)
        close(stream)
    end
    println(load_dict)
    agent = load_dict[:my_save_dict][:agent]
    to_device!(agent)
    undo_remove_undefs(agent)
    return agent
end

Base.show(io::IO, a::Agent) = begin
    println()
    println("---------------------------")
    L = length(a.name)
    println(a.name)
    println(join(["-" for i = 1:L+1]))

    for subagent in a.subagents
        println(io, "subagent." * subagent.name * ": ")
        println(io, subagent)
    end
    print("Buffers: ")
    [println(io, buffer) for buffer in keys(a.buffers) if !isnothing(a.buffers[buffer])]
    println("π_b: ", typeof(a.π_b).name)
    println("measurement_freq: ", a.measurement_freq)
    println("measurement_count: ", a.measurement_count)
    println("max_agent_steps: ", a.max_agent_steps)
    println("---------------------------")
end

function train_subagents(agent; reg = false, buffer = nothing)
    if isnothing(buffer)
        buffer = agent.buffers.train_buffer
    end

    for subagent in agent.subagents
        train_subagent(
            subagent,
            buffer,
            agent.state_encoder,
            agent.action_encoder,
            reg = reg,
        )
        if mod(subagent.update_count, subagent.update_freq) == 0
            update_submodel!(subagent, "state_encoder"; new_ps = get_params(agent.state_encoder))
        end
    end
    return nothing
end

include("agent_zoo.jl")
include("default_agents/default_agents.jl")

###
###
