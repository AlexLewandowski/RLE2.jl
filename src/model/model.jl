using Flux: onehotbatch

abstract type AbstractModel{A} end

function (model::AbstractModel)(s; dropgrad = false, kwargs...)
    # _dropgrad(forward(model, s; kwargs...), dropgrad)
    if dropgrad
        stop_gradient() do
            forward(model, s; kwargs...)
        end
    else
        forward(model, s; kwargs...)
    end
end

function (model::AbstractModel)(s, mask; dropgrad = false, kwargs...)
    # _dropgrad(forward(model, s, mask; kwargs...), dropgrad)
    if dropgrad
        stop_gradient() do
            forward(model, s, mask; kwargs...)
        end
    else
        forward(model, s, mask; kwargs...)
    end
end

function forward(model::AbstractModel, s; kwargs...)
    model.f(s; kwargs...)
end

function forward(model::AbstractModel, s, mask; kwargs...)
    sum(model.f(s) .* mask, dims = 1; kwargs...)
end

include("value.jl")
include("action_value.jl")
# include("policy_action_value.jl")
include("policy.jl")
# include("persist.jl")
# include("rnn.jl")
# include("rnnplanner.jl")
# include("meta_state_encoder.jl")
include("model_zoo.jl")

Base.show(io::IO, m::AbstractModel) = begin
    # print(io, string(typeof(m).name)[10:end-1])
    print(io, typeof(m))
    # println(io,  m.f)
end

function to_device!(model::AbstractModel, device)
    model.f.f = model.f.f |> device
end

function representation(model::AbstractModel, s)
    representation(model.f, s)
    # list_of_layers = layers(subagent)
    # Chain(list_of_layers[1:end-1]...)(s)
end

function get_out_dim(model::AbstractModel)
    return size(layers(model)[end].W)[1]
end

function get_out_dim(model::AbstractModel{E}) where {E <: Tabular}
    return size(model.f.f)[2]
end

function layers(model::AbstractModel)
    return layers(model.f)
end

function get_params_vector(m::AbstractModel)
    get_params_vector(m.f)
end

function get_params(m::AbstractModel)
    get_params(m.f)
end

function num_states(m::AbstractModel)
    size(m.f.layers[1].W)[2]
end


function discrete_action_mask(Q::AbstractArray{Float32}, num_actions::Int)
    a = argmax(Q, dims = 1)

    mask = onehotbatch(a, 1:num_actions)
    return Int64.(mask)
end

function discrete_action_mask(a::Int, num_actions::Int)
    mask = onehotbatch(a, 1:num_actions)
    return Int64.(mask)
end

function discrete_action_mask(a::AbstractArray{Int}, num_actions::Int)
    if size(a)[1] == 1
        a = dropdims(a, dims = 1)
    end
    mask = onehotbatch(a, 1:num_actions)
    # return (1 .- mask) - mask
    return Int64.(mask)
end

function range_mask(a::Int, num_actions::Int)
    z = zeros((num_actions, 1))
    z[1:a, 1] .= 1
    return Int64.(z)
end

function range_mask(a::AbstractArray{Int}, num_actions::Int)
    L = length(a)
    z = range_mask(a[1], num_actions)
    for i = 2:L
        z = cat(z, range_mask(a[i], num_actions), dims = 2)
    end
    return z
end

function to_env_action(π::AbstractModel, a)
    return a
end

function to_env_action(π::AbstractModel, a::AbstractArray{Float32,2})
    return [act[1] for act in argmax(a, dims = 1)]
end

function positional_encoding(
    pos::AbstractArray{T, 1},
    d = 4,
) where {T <: Number}
    x = []
    for e in pos
        push!(x, positional_encoding(e, d))
    end
    return reshape(hcat(x...), :)
end

function positional_encoding(
    pos::AbstractArray{T, 2},
    d = 4,
) where {T <: Number}
    x = []
    M = size(pos)[end]
    for i = 1:M
        push!(x, reshape(positional_encoding(pos[:, i], d), (:,1)))
    end
    return reshape(hcat(x...), (:, M))
end

function positional_encoding(
    pos::Number,
    d = 4,
)
    i = 0
    pe_list = zeros(Float32, (1, 2*d))
    pe_list[1, 1:2] = [sin.(pos / (10000.0f0^(i / d))), cos.(pos / (10000.0f0^(i / d)))]

    for i = 1:d-1
        pe = [sin.(pos / (10000.0f0^(i / d))), cos.(pos / (10000.0f0^(i / d)))]
        pe_list[1, 2*i+1 : 2*i + 2] = pe
    end
    return pe_list
end
