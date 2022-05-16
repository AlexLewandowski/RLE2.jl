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

abstract type AbstractSequenceNeuralNetwork <: AbstractNeuralNetwork end

mutable struct MultiSequenceNeuralNetwork <: AbstractSequenceNeuralNetwork
    f
end

function get_params(f::MultiSequenceNeuralNetwork)
   Flux.params([f_.f for f_ in f.f]...)
end

function to_device(NN::MultiSequenceNeuralNetwork, device)
    for f_ in NN.f
        to_device(f_, device)
    end
end

function forward(f::MultiSequenceNeuralNetwork, x::AbstractArray{Float32, 2});
    inputs, aux_input, s_dim, a_dim, L, T = get_input(x)

    outs = f.f[1](x, ind = 1, with_aux = false)
    i = 2
    for input in inputs[2:end]
        outs = cat(f.f[i](x, ind = i, with_aux = false), outs, dims = 1)
        i += 1
    end
    outs = cat(outs, aux_input, dims = 1)
    return outs
end


mutable struct SequenceNeuralNetwork{A} <: AbstractSequenceNeuralNetwork
    f
    params
end

function Flux.params(f::AbstractSequenceNeuralNetwork)
    Flux.params(f.f)
end

Base.show(f::AbstractSequenceNeuralNetwork) = begin
    println(typeof(f))
end

Base.display(f::AbstractSequenceNeuralNetwork) = begin
    println(typeof(f))
end

function SequenceNeuralNetwork(f::Union{Chain, NamedTuple}; recurs = true)
    ps = get_params(f)
    if typeof(f) <: Chain
        type_f = AbstractChain
    else
        type_f = AbstractNamed
    end
    return SequenceNeuralNetwork{type_f}(f, ps)
end

function get_aux(x; target = false)
    M = size(x)[end]
    L = size(x)[1]
    aux_dim = Int(sum(mean(x[end-4,:], dims = 1)))
    aux = reshape(x[end-5-aux_dim+1:end-5,:], (:,M))
    return aux
end

function get_input(x; target = false)
    if length(size(x)) == 1
        M = 1
    else
        M = size(x)[end]
    end

    S = size(x)[1]

    L = Int(sum(mean(x[end-1,:], dims = 1)))
    T = Int(sum(mean(x[end,:], dims = 1)))
    a_dim = Int(sum(mean(x[end-2,:], dims = 1)))
    s_dim = Int(sum(mean(x[end-3,:], dims = 1)))
    aux_dim = Int(sum(mean(x[end-4,:], dims = 1)))

    D = s_dim + a_dim

    input = reshape(x[1:D*T*L, :], (D*L,T,M))
    aux = reshape(x[end-5-aux_dim+1:end-5,:], (:,M))

    if aux_dim == 0
        aux_dim = 1
    end

    full_batch_size = Int((size(x)[1] - aux_dim - 5) / D)
    full_batch_factor = Int(full_batch_size / T)

    if full_batch_factor > 1
        input = [input, reshape(x[D*T*L + 1:2*D*T*L, :], (D*L,T,M))]
    else
        input = [input]
    end

    return input, aux, s_dim, a_dim, L, T
end


function reset_model!(m, mode = :default)
end

function reset_model!(m::AbstractSequenceNeuralNetwork, mode = :default)
    # if mode == :zeros
    #
    # if typeof(m.f[1][1]) <: Flux.Recur
    #     M = size(m.f[1][1].state)[1]
    #     m.f[1][1].state = zeros(Float32, (M,1))
    # end
        # m.old_f.f[1][1].state = zeros(Float32, (M,1))
    # else
    #     Flux.reset!(m.f)
    # end
end

function forward(f::AbstractSequenceNeuralNetwork, x::AbstractArray{Float32, 3}; kwargs...)
    T = size(x)[2]
    M = size(x)[end]
    out = reshape(forward(f,x[:,1,:];kwargs...), (:, 1, M))
    for t = 2:T
        out = cat(out, reshape(forward(f,x[:,t,:]; kwargs...), (:, 1, M)), dims = 2)
    end
    return out
end

function forward(f::AbstractSequenceNeuralNetwork, x::AbstractArray{Float32, 1}; kwargs...)
    return reshape(forward(f, reshape(x, (:,1,1)); kwargs...), :)
end

function forward(f::AbstractSequenceNeuralNetwork, x::AbstractArray{Float32, 2}; target = false, ind = 1, with_aux = true)
    M = size(x)[end]

    input, aux_input, s_dim, a_dim, L, T = get_input(x)
    input = input[ind]

    D = s_dim + a_dim

    l = 1
    offset = (l-1)*D + 1

    s_inds = offset:(s_dim + offset - 1)
    a_inds = (s_dim + offset):(offset - 1 + s_dim + a_dim)
    input_s = input[s_inds, :, :]
    input_a = input[a_inds, :, :]

    # input_s = reshape(input_s, (:, T*M))
    input_s = f.f[1][l][1](input_s)
    # input_s = reshape(input_s, (:, T, M))

    if !isempty(input_a)
        # input_a = reshape(input_a, (:, T*M))
        # input_a = f.f[1][l][2](input_a)
        # input_a = reshape(input_a, (:, T, M))
        # input_joint = cat(input_a, input_s, dims = 1)
    else
        input_joint = input_s
    end


    inputs = f.f[1][l][3](input_joint)
    # inputs = input_joint

    # for l = 2:L
    #     offset = (l-1)*D + 1

    #     s_inds = offset:(s_dim + offset - 1)
    #     a_inds = (s_dim + offset):(offset - 1 + s_dim + a_dim)
    #     input_s = input[s_inds, :, :]
    #     input_a = input[a_inds, :, :]

    #     input_s = f.f[1][l][1](input_s)
    #     input_a = reshape(input_a, (:, T*M))
    #     input_a = f.f[1][l][2](input_a)
    #     input_a = reshape(input_a, (:, T, M))
    #     input_joint = cat(input_a, input_s, dims = 1)
    #     input_joint = f.f[1][l][3](input_joint)
    #     inputs = cat(inputs, input_joint, dims = 1)
    # end
    input = inputs

    res = dropdims(f.f[2](input), dims = 2)
    res = f.f[3](res)

    D = size(res)[1]

    if with_aux
        res = cat(res, aux_input, dims = 1)
    end

    return res
end

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
    # return head_out
    return f.WO(head_out)
end

abstract type AbstractPEN <: AbstractNeuralNetwork end

mutable struct PEN{A} <: AbstractPEN
    f
    re
    inputs
end

Flux.@functor PEN

function forward(f::PEN, x::AbstractArray{Float32, 3})
    T = size(x)[2]
    M = size(x)[3]
    @assert T == 1
    reshape(forward(f, x[:,1,:]), (:, T, M))
end

function forward(f::PEN, x::AbstractArray{Float32, 2})
    M = size(x)[2]
    outs = []
    for i = 1:M
        outs = vcat(outs, [forward(f, x[:, i])])
    end
    return hcat(outs...)
end

function forward(f::PEN, x::AbstractArray{Float32, 1})
    g = f.re(x)
    outs = g(f.inputs)
    outs = reshape(outs, :)
    return f.f(outs)
end

function PEN(f::Union{Chain, NamedTuple}, re, input_dim; init = glorot_uniform, num_inputs = 10)
    if typeof(f) <: Chain
        type_f = AbstractChain
    else
        type_f = AbstractNamed
    end
    inputs = init(input_dim, num_inputs)
    return PEN{Chain}(f, re, inputs)
end

function to_device(f::AbstractPEN, device)
    f.inputs = f.inputs |> device
    f.f = f.f |> device
end
