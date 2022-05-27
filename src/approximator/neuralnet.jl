import Flux: Chain

abstract type AbstractNeuralNetwork <: AbstractApproximator end

mutable struct NeuralNetwork{A} <: AbstractNeuralNetwork
    f
    params
end

function NeuralNetwork(f::Union{Chain, NamedTuple})
    ps = get_params(f)
    if typeof(f) <: Chain
        type_f = AbstractChain
    else
        type_f = AbstractNamed
    end
    return NeuralNetwork{type_f}(f, ps)
end

Base.show(f::NeuralNetwork{A}) where {A<:AbstractNamed} = begin
    keys(f)
    println()
    for key in keys(f)
        print("    ", key)
        head = getfield(f, key)
        print("    ")
        print(head)
    end
end

Base.show(f::NeuralNetwork{A}) where {A<:AbstractChain} = begin
    keys(f)
    println()
    for key in keys(f)
        print("    ", key)
        head = getfield(f, key)
        print("    ")
        print(head)
    end
end

(f::AbstractNeuralNetwork)(x; kwargs...) = forward(f, x; kwargs...)

forward(f::NeuralNetwork{<:AbstractChain}, x; kwargs...) = f.f(x)
forward(f::NeuralNetwork{<:AbstractNamed}, x, head::Symbol; kwargs...) = getfield(f.f, head)(x)
forward(f::NeuralNetwork{<:AbstractNamed}, head::Symbol; kwargs...) = getfield(f.f, head)

function to_device!(NN::AbstractNeuralNetwork, device)
    NN.f = NN.f |> device
    NN.params = Flux.params(NN.f)
end

function layers(NN::NeuralNetwork)
    return NN.f.layers
end

function representation(NN::NeuralNetwork, s)
    list_of_layers = layers(NN)
    Chain(list_of_layers[1:end-1]...)(s)
end

function get_params_vector(NN::AbstractNeuralNetwork)
    return Flux.destructure(NN.f)[1]
end

function get_params(NN::AbstractNeuralNetwork)
    return Flux.params(NN.f)
end

function get_greedypolicy_vector(NN::AbstractNeuralNetwork)
    return get_greedypolicy(NN)
end

function get_greedypolicy(NN::AbstractNeuralNetwork)
    return []
end

function get_params(m::Chain)
    return Flux.params(m)
end

function get_params(m::NamedTuple)
    ps = []
    for key in keys(m)
        push!(ps, Flux.params(m[key]))
    end
    # return Flux.params(ps)
    return ps
end

