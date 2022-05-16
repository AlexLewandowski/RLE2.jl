reversesumscan_(x::AbstractArray; dims::Integer) =
    reverse(cumsum(reverse(x, dims = dims), dims = dims), dims = dims)

reverseprodscan_(x::AbstractArray; dims::Integer) =
    reverse(cumprod(reverse(x, dims = dims), dims = dims), dims = dims)

function get_returns(r, p_b, p_a)
    ρ_step = reverseprodscan_(p_a ./ p_b, dims = 1)
    ρ_step = p_a ./ p_b
    weighted = r .* ρ_step
    G = reversesumscan_(weighted, dims = 1)
    #ρ_traj = reverseprodscan_(p_a ./ p_b, dims = 1)

    #ρ_traj =  cumprod(p_a ./ p_b, dims = 1)
    #G = reversesumscan_(r, dims = 1)
    #G =  G .* ρ_traj
    return G
end

function get_returns(r::AbstractArray{<:Number})
    return reversesumscan_(r, dims = 1)
end

function get_returns(ep::AbstractArray{<:Experience}, gamma)
    r = map(p -> p.r, ep)
    T = size(r)[1]
    gamma_mat = reshape(Float32.([gamma^n for n = 0:(T-1)]), T)
    r = r .* gamma_mat
    g = reversesumscan_(r, dims = 1)
    return g ./ gamma_mat
end

function kldivergence(p::AbstractArray, q::AbstractArray)
    KL = p .* (log.(p) .- log.(q))
    return sum(KL)
end

function crossent_logits(p::AbstractArray, q_logits::AbstractArray)
    max = maximum(q_logits, dims = 1)
    # println(size(p))
    # println(size(q_logits))
    q_scale = q_logits .- max
    # println(q_logits)
    Z = sum(exp.(q_scale), dims = 1)
    # Z = Float32(0.0)
    # println(p.*(q_logits .- log.(Z)))
    return sum(sum(p .* (q_scale .- log.(Z)), dims = 1))
end


function doublyrobust(Q, G, p_b, mask)
    T = length(p_b)
    G_r = reshape(G, (1, T))
    P_r = reshape(p_b, (1, T))
    imputation = (G_r .- Q) ./ P_r
    imputation = imputation .* mask
    DR = Q .+ imputation
    return DR
end
