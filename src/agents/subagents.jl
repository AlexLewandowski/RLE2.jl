""" An Subagent is a functional object that stores a collection of subagents (or
function approximators).
"""

mutable struct Subagent{P<:AbstractModel} <: AbstractSubagent
    model::P
    submodels::NamedTuple
    target_submodels::NamedTuple
    target::NamedTuple
    gamma::Float32
    num_grad_steps::Int
    train_freq::Int64
    update_freq::Int64
    update_count::Int64
    callbacks::AbstractArray
    loss::Function
    training_proc::Function
    optimizer::Any
    params::Any
    regularizer::Any
    device::Function
    name::String
end

function Subagent(
    model::AbstractModel,
    submodels::NamedTuple,
    gamma::Float32,
    update_freq::Int64,
    num_grad_steps::Int64,
    loss::Any,
    opt::Any,
    name::String;
    target = (s_network = nothing, sp_network = nothing, func = max_target),
    training_proc = minibatch_loss,
    callbacks = [],
    device = Flux.cpu,
    other_params = [nothing],
    train_freq = 1,
    target_submodels = nothing
)
    if isnothing(target_submodels)
        target_submodels = submodels
    end
    regularizer = nothing
    params = Flux.params(model.f.f, other_params...)
    return Subagent{typeof(model)}(
        model,
        submodels,
        target_submodels,
        target,
        gamma,
        num_grad_steps,
        train_freq,
        update_freq,
        0,
        callbacks,
        loss,
        training_proc,
        opt,
        params,
        regularizer,
        device,
        name,
    )
end

function update!(subagent::Subagent, gs)
    Flux.update!(subagent.optimizer, subagent.params, gs)
end

Base.show(io::IO, s::Subagent) = begin
    print(io, "  model: ")
    println(io, s.model)

    print("  submodels: ")
    [print(io, submodel, " | ") for submodel in keys(s.submodels)]
    println()

    print("  target: ")
    targets = s.target
    [
        print(io, target, "=", targets[target], " | ") for
        target in keys(targets) if !isnothing(targets[target])
    ]
    println()

    print("  num_grad_steps: ")
    println(s.num_grad_steps)

    print("  gamma: ")
    println(s.gamma)

    print("  loss: ")
    println(s.loss)
end

function (subagent::AbstractSubagent)(s; dropgrad = false)
    subagent.model(s |> subagent.submodels.state_encoder, dropgrad = dropgrad)
end

(subagent::Subagent{A})(s; dropgrad = false) where {A<:QRActionValue} = dropdims(
    mean(
        subagent.model(s |> subagent.submodels.state_encoder, dropgrad = dropgrad),
        dims = 1,
    ),
    dims = 1,
)

(subagent::AbstractSubagent)(s, mask; dropgrad = false) =
    subagent.model(s |> subagent.submodels.state_encoder, mask, dropgrad = dropgrad)

(subagent::Subagent{A})(s, mask; dropgrad = false) where {A<:QRActionValue} = dropdims(
    mean(
        subagent.model(s |> subagent.submodels.state_encoder, mask, dropgrad = dropgrad),
        dims = 1,
    ),
    dims = 1,
)

function get_out_dim(subagent::AbstractSubagent)
    return size(layers(subagent)[end].W)[1]
end

function representation(subagent::AbstractSubagent, s)
    s = subagent.submodels.state_encoder(s)
    representation(subagent.model, s)
    # list_of_layers = layers(subagent)
    # Chain(list_of_layers[1:end-1]...)(s)
end

function get_params_vector(subagent::AbstractSubagent)
    get_params_vector(subagent.model)
end

function get_params(subagent::AbstractSubagent)
    get_params(subagent.model)
end

function layers(subagent::AbstractSubagent)
    return layers(subagent.model)
end

function get_oldparams(subagent::AbstractSubagent, name)
    submodel_keys = keys(subagent.submodels)
    if Symbol(name) in submodel_keys
        if subagent.submodels[Symbol(name)] == nothing
            return nothing
        else
            return get_params(subagent.target_submodels[Symbol(name)])
        end
    else
        return nothing
    end
end

function update_submodel!(subagent::AbstractSubagent, name; new_ps = nothing)
    if new_ps === nothing
        new_ps = subagent.params
    end
    old_ps = get_oldparams(subagent, name)
    if !isnothing(old_ps)
        p = 0.995f0
        for (old_p, new_p) in zip(old_ps, new_ps)
            # old_p .= p .* old_p + (1 - p) .* new_p
            old_p .= new_p
        end
    end
end

function callback(subagent::AbstractSubagent)
    if mod(subagent.update_count, subagent.update_freq) == 0
        update_submodel!(subagent, subagent.name)
        for cb in subagent.callbacks
            cb(subagent)
        end
    end
end


function nstep_returns(gamma, device, rs::AbstractArray{Float32,3}; all = false)
    if all == true
        return nstep_returns(gamma, rs, true) |> device
    else
        return nstep_returns(gamma, rs) |> device
    end
end

function nstep_returns(subagent::AbstractSubagent, rs::AbstractArray{Float32,3}; all = false)
    if all == true
        return nstep_returns(subagent.gamma, rs, true) |> subagent.device
    else
        return nstep_returns(subagent.gamma, rs) |> subagent.device
    end
end

function nstep_returns(gamma::Float32, rs::AbstractArray{Float32,3})
    T = size(rs)[2]
    M = size(rs)[end]
    @assert size(rs)[1] == 1
    gamma_mat =
        reshape(Float32.(hcat([[gamma^n for n = 0:(T-1)] for _ = 1:M]...)), (1, T, M))
    discounted_reward_sum = reshape(sum(rs .* gamma_mat, dims = 2), :)
    @assert length(discounted_reward_sum) == M
    return discounted_reward_sum
end


function nstep_returns(gamma::Float32, rs::AbstractArray{Float32,3}, all)
    T = size(rs)[2]
    M = size(rs)[end]
    discounted_reward_sum = zeros(Float32, (1, T, M))
    for t = 1:T
        dsm = nstep_returns(gamma, rs[:, t:end, :])
        discounted_reward_sum[:, t, :] = dsm
    end
    return discounted_reward_sum
end

function to_device!(subagent::AbstractSubagent, device = :default)
    if device == :default
        device = subagent.device
    end
    old_ps = subagent.params
    to_device!(subagent.model, device)
    for submodel in subagent.submodels
        to_device!(submodel, device)
    end
    for submodel in subagent.target_submodels
        to_device!(submodel, device)
    end
    new_ps = Flux.params(get_params(subagent.model)..., Flux.params(subagent.submodels.state_encoder)...) #TODO: How to incoporate other pararms

    subagent.params = new_ps

    old_opt = subagent.optimizer
    if typeof(old_opt) <: Flux.ADAM
    if !isempty(old_opt.state)
        old_opt.state = IdDict(new_p => (device(old_opt.state[old_p][1]), device(old_opt.state[old_p][2]), old_opt.state[old_p][3])  for (new_p, old_p) in zip(new_ps, old_ps))
    end
    end

    subagent.device = device
    nothing
end
