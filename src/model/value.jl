
abstract type AbstractValue{A} <: AbstractModel{A} end

mutable struct Value{A<:AbstractApproximator} <: AbstractValue{A}
    f::A
end
