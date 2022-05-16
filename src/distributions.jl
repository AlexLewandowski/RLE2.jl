function ϵgreedy(q; ϵ = 0.10f0)
    num_actions = length(q)
    p = zeros(size(q)) .+ ϵ / num_actions
    action = argmax(q)
    p[action] = 1.0f0 - ϵ + ϵ / num_actions
    return p
end
