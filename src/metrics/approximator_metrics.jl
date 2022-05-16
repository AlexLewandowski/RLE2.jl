function layer_rank(layer::Dropout)
    return nothing, "dropout"
end

function layer_rank(layer::Dense)
    λ = svd(layer.W).S .^ 2
    return sum(λ) / maximum(λ), "Dense"
end

function layer_rank(layer::Recur)
    λ = svd(layer.cell.Wh).S .^ 2
    return sum(λ) / maximum(λ), "recurrent"
end

function stable_rank(agent::AbstractAgent, env::AbstractEnv)
    return apply_to_list(stable_rank, agent.subagents, env)
end

function stable_rank(subagent::AbstractSubagent, env)
    list_of_layers = layers(subagent)
    SRs = []
    names = []
    for l in list_of_layers
        SR, layer_name = layer_rank(l)
        if SR !== nothing
            name = "stable_rank_$layer_name" * string(l) * "_" * subagent.name
            push!(SRs, SR)
            push!(names, name)
        end
    end
    return SRs, names
end

function sum_weights(layer::typeof(identity))
    return 0
end

function sum_weights(layer::Dropout)
    return 0
end

function sum_weights(layer::Dense)
    return sum(layer.W .^ 2) + sum(layer.b .^2)
end

function sum_weights(layer::Recur)
    l = layer.cell
    return sum(l.Wh .^ 2)# + sum(l.Wi) + sum(l.b)
end

function num_weights(layer::typeof(identity))
    return 1
end

function num_weights(layer::Dropout)
    return 1
end

function num_weights(layer::Dense)
    return prod(size(layer.W)) + prod(size(layer.b))
end

function num_weights(layer::Recur)
    l = layer.cell
    return prod(size(l.Wh))# + prod(size(l.Wi)) + prod(size(l.b))
end

function mean_weights(agent::AbstractAgent, env::AbstractEnv)
    return apply_to_list(mean_weights, agent.subagents, env, state_encoder = agent.state_encoder)
end

function mean_weights(subagent::AbstractSubagent, env; state_encoder)
    if typeof(state_encoder) <: AbstractNeuralNetwork
        list_of_layers = vcat(collect(layers(subagent)), state_encoder.f.layers...)
    else
        list_of_layers = layers(subagent)
    end
    W = 0.0f0
    N = 0 + 1
    for l in list_of_layers
        W += sum_weights(l)
        N += num_weights(l)
    end
    return [W / N], [subagent.name * "-mean_weights"]
end


function num_weights(agent::AbstractAgent, env::AbstractEnv)
    return apply_to_list(num_weights, agent.subagents, env)
end

function num_weights(subagent::AbstractSubagent, env)
    list_of_layers = layers(subagent)
    N = 0
    for l in list_of_layers
        N += num_weights(l)
    end
    return [N], [subagent.name * "-num_weights"]
end
