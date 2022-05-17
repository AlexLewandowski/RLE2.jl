"""
An approximator is a functional object that produces an output given an input.
The main backend for neural networks is Flux/Zygote.
Tabular is simply a lookup table.
"""

abstract type AbstractApproximator end
abstract type AbstractChain end
abstract type AbstractNamed end

include("tabular.jl")
include("neuralnet.jl")

##
## Specialized NN Modules
##

include("PVN.jl")
include("attention.jl")
include("PE.jl")

##
## Neural Network Architecture Constructors
##

include("architectures.jl")
