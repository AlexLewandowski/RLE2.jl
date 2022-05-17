##
## Attention and Multi-Head Attention
##

function attention(q,k,v)
    d = Float32(size(k)[2])

    score = Flux.batched_mul(q, permutedims(k, (2,1,3)))
    soft_score = softmax(score/sqrt(d), dims = 2)
    attention_mask = Flux.batched_mul(soft_score, v)
    return attention_mask
end

mutable struct Attention{W<:AbstractMatrix} <: AbstractNeuralNetwork
    WQ::W
    WK::W
    WV::W
end

Flux.@functor Attention

 function Base.show(io::IO, l::Attention)
     print(io, "Attention(",
           "WK: ", size(l.WK, 2), ", ", size(l.WK, 1), " | ",
           "WQ: ", size(l.WQ, 2), ", ", size(l.WQ, 1), " | ",
           "WV: ", size(l.WV, 2), ", ", size(l.WV, 1),
           )
  print(io, ")") end

function Attention(in::Integer, h_dim::Integer; num_heads::Integer = 4, output_a = identity,
               init = glorot_uniform, seed = nothing, y_in = nothing, bias=true)
    if isnothing(y_in)
        y_in = in
    end

    d = h_dim
    WQ = init(d, in)
    WK = init(d, y_in)
    WV = init(d, y_in)
    A = Attention(WQ, WK, WV)

    return A
end

mutable struct SetAttention{W<:AbstractMatrix} <: AbstractNeuralNetwork
    Q::W
    WK::W
    WV::W
end

function to_device(NN::SetAttention, device)
    NN.Q = NN.Q |> device
    NN.WK = NN.WK |> device
    NN.WV = NN.WV |> device
end

Flux.@functor SetAttention

 function Base.show(io::IO, l::SetAttention)
     print(io, "SetAttention(",
           "WK: ", size(l.WK, 2), ", ", size(l.WK, 1), " | ",
           "Q: ", size(l.Q, 2), ", ", size(l.Q, 1), " | ",
           "WV: ", size(l.WV, 2), ", ", size(l.WV, 1),
           )
  print(io, ")") end

function SetAttention(in::Integer, h_dim::Integer;
               init = glorot_uniform, seed = nothing, y_in = nothing, bias=true, output_set_size = 1)
    if isnothing(y_in)
        y_in = in
    end

    d = h_dim

    Q = init(output_set_size, d)
    WK = init(d, y_in)
    WV = init(d, y_in)

    A = SetAttention(Q, WK, WV)
    return A
end

function forward(f::SetAttention, x::AbstractArray{Float32, 1}; kwargs...)
    error()
end

function forward(f::SetAttention, x::AbstractArray{Float32, 2}; kwargs...)
    D = size(x)[1]
    T = size(x)[2]

    forward(f, reshape(x, (D, T, 1)))
end

function forward(f::SetAttention, x::AbstractArray{Float32, 3}; kwargs...)
    Q = f.Q
    K = Flux.batched_mul(f.WK, x)
    K = permutedims(K, (2,1,3))
    V = Flux.batched_mul(f.WV, x)
    V = permutedims(V, (2,1,3))
    return permutedims(attention(Q,K,V), (2,1,3))
end

mutable struct MultiHeadAttention <: AbstractNeuralNetwork
    As::Vector{Union{Attention, SetAttention}}
    WO::Flux.Dense
end

function to_device(NN::MultiHeadAttention, device)
    for A in NN.As
        A = A |> device
    end
    NN.WO = NN.WO |> device
    nothing
end

function MultiHeadAttention(in::Integer, out::Integer, h_dim::Integer; num_heads::Integer = 4, output_a = identity,
               init = glorot_uniform, seed = nothing, y_in = nothing, bias=true, set_attention = false, WO = nothing)
    if isnothing(y_in)
        y_in = in
    end

    if set_attention
        AttentionType = SetAttention
    else
        AttentionType = Attention
    end

    d = Int(floor(in/num_heads))
    As = Vector{AttentionType}()
    for _ = 1:num_heads
        A = AttentionType(in, d, y_in = y_in)
        push!(As, A)
    end

    WO = Dense(num_heads*d, out, output_a, initW = init, bias = false)

    return MultiHeadAttention(As, WO)
end

Flux.@functor MultiHeadAttention

 function Base.show(io::IO, x::MultiHeadAttention)
  print(io, "MultiHeadAttention(")
  print(io, "num_heads = ", length(x.As),     " | ")
  print(io, "in_dim = ", size(x.As[1].WK)[1], " | ")
  print(io, "h_dim = ", size(x.As[1].WK)[2],  " | ")
  print(io, "out_dim = ", size(x.WO.W)[1],    " | ")
  print(io, ")")
 end

(m::MultiHeadAttention)(xs::AbstractArray) = forward(m ,xs)

function forward(f::MultiHeadAttention, x::AbstractArray{Float32, 1}, y = nothing; kwargs...)
    error()
end

function forward(f::MultiHeadAttention, x::AbstractArray{Float32, 2}, y = nothing; kwargs...)
    D = size(x)[1]
    T = size(x)[2]

    forward(f, reshape(x, (D, T, 1)))
end

function forward(f::MultiHeadAttention, x::AbstractArray{Float32, 3}, y = nothing; kwargs...)
    if isnothing(y)
        y = x
    end
    num_heads = length(f.As)
    M = size(x)[end]
    head_out = cat([forward(f.As[i], x) for i = 1:num_heads]..., dims = 1)
    return f.WO(head_out)
end
