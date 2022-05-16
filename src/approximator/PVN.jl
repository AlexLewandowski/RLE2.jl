##
##  Policy Evaluation Networks
##

abstract type AbstractPVN <: AbstractNeuralNetwork end

mutable struct PVN{A} <: AbstractPVN
    f
    re
    inputs
end

Flux.@functor PVN

function forward(f::PVN, x::AbstractArray{Float32, 3})
    T = size(x)[2]
    M = size(x)[3]
    @assert T == 1
    reshape(forward(f, x[:,1,:]), (:, T, M))
end

function forward(f::PVN, x::AbstractArray{Float32, 2})
    M = size(x)[2]
    outs = []
    for i = 1:M
        outs = vcat(outs, [forward(f, x[:, i])])
    end
    return hcat(outs...)
end

function forward(f::PVN, x::AbstractArray{Float32, 1})
    g = f.re(x)
    outs = g(f.inputs)
    outs = reshape(outs, :)
    return f.f(outs)
end

function PVN(f::Union{Chain, NamedTuple}, re, input_dim; init = glorot_uniform, num_inputs = 10)
    if typeof(f) <: Chain
        type_f = AbstractChain
    else
        type_f = AbstractNamed
    end
    inputs = init(input_dim, num_inputs)
    return PVN{Chain}(f, re, inputs)
end

function to_device(f::AbstractPVN, device)
    f.inputs = f.inputs |> device
    f.f = f.f |> device
end
