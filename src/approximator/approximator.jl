import Flux: Chain

"""
An approximator is a functional object to estimate a state.
Two typical kinds of approximators are
[`AbstractVApproximator`](@ref) and [`AbstractQApproximator`](@ref).
"""

abstract type AbstractApproximator end
abstract type AbstractChain end
abstract type AbstractNamed end

include("neuralnet.jl")
include("tabular.jl")
