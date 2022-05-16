mutable struct Tabular <: AbstractApproximator
    f::Union{Vector, Array}
end

function Tabular(dim::Tuple)
    f = zeros(Float32, dim)
    return Tabular(f)
end

function layers(NN::Tabular)
    return [size(NN.f)[1], size(NN.f)[2]]
end

function (f::Tabular)(x::Union{Vector,Array})
    if length(size(x)) == 1
        return reshape(sum(f.f.*x, dims = 1), (:, 1))
    else
        B = size(x)[end]
        xs = [x[:,i] for i = 1:B]
        return reshape(hcat([sum(f.f.*x, dims = 1) for x in xs]...), (:,B))
    end
end

function (f::Tabular)(dim::Tuple, location::Tuple)
    s = zeros(dim)
    s[CartesianIndex(location)] = 1
    s = reshape(s, :)
    f(s)
end

function update!(f::Tabular, s, a, L, lr)
    if length(size(f.f)) == 1
        f.f[argmax(s)] += lr * (L)
    else
        f.f[argmax(s), a] += lr * (L)
    end
end

function get_params(f::Tabular)
    return f.f
end

function get_greedypolicy(f::Tabular)
    Q = f.f
    m = maximum(Q, dims = 2)
    policy = Q .== m
    Z = sum(policy, dims = 2)
    return policy./Z
end

function get_greedypolicy(Q::Matrix{Float32})
    m = maximum(Q, dims = 1)
    policy = Q .== m
    Z = sum(policy, dims = 1)
    return policy./Z
end

function get_greedypolicy(Q::Vector{Float32})
    m = maximum(Q)
    policy = Q .== m
    Z = sum(policy)
    return policy./Z
end

function get_greedypolicy_vector(Q::Vector{Float32})
    return reshape(get_greedypolicy(Q), :)
end

function get_params_vector(f::Tabular)
    return reshape(f.f, :)
end

function get_greedypolicy_vector(f::Tabular)
    p = get_greedypolicy(f)
    return reshape(p, :)
end

# struct TileCode #TODO Tile-coding and tabular
# end
