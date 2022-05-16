##
## Parametric Behavior Embedder
##

abstract type AbstractSequenceNeuralNetwork <: AbstractNeuralNetwork end

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
