using Parameters
@with_kw mutable struct ModelParams
    input_dim::Int64
    output_dim::Int64
    discrete::Bool
    predict_window::Int64
    history_window::Int64
    num_layers::Int64
    hidden_size::Int64
    activation::Function
    device::Any
    drop_rate::Float32
    seed::Union{Int,Nothing}
    tabular::Bool
    model_local::NamedTuple
end

function (model_params::ModelParams)(xs::Vector)
    new = deepcopy(model_params)
    for x in xs
        if x[1] == :model_local
            setfield!(new, x[1], merge(new.model_local, x[2]))
        else
            setfield!(new, x[1], x[2])
        end
    end
    return new
end


# TODO maybe kwargs?
function get_model(
    ModelType::String,
    model_params,
    # ns,
    # na,
    # discrete,
    # predict_window,
    # history_window,
    # num_layers,
    # hidden_size,
    # activation,
    # device,
    # drop_rate = 0.0,
    # seed = nothing,
    # tabular = true,
    # model_local::NamedTuple,
)
    @unpack_ModelParams model_params

    memory_size = model_local.memory_size
    planning_algo = model_local.planning_algo

    if ModelType == "ActionValue"
        if !tabular
            A = feed_forward(
                input_dim,
                output_dim,
                hidden_size,
                drop_rate = drop_rate,
                num_hidden_layers = num_layers,
                σ = activation,
                seed = seed,
            )

            layers = collect(A.layers)

            if :input_encoder in keys(model_local)
                pushfirst!(layers, model_local.input_encoder)
            end

            if :output_encoder in keys(model_local)
                push!(layers, model_local.output_encoder)
            end

            net = NeuralNetwork(Chain(layers...) |> device)

        else
            net = Tabular((input_dim, output_dim))
        end
        model = ActionValue(net)

    elseif ModelType == "ContinuousActionValue"
        @assert !tabular
        A = feed_forward(
            input_dim + output_dim,
            1,
            hidden_size,
            drop_rate = drop_rate,
            num_hidden_layers = num_layers,
            σ = activation,
            seed = seed,
        )

        layers = collect(A.layers)

        if :input_encoder in keys(model_local)
            pushfirst!(layers, model_local.input_encoder)
        end

        if :output_encoder in keys(model_local)
            push!(layers, model_local.output_encoder)
        end

        net = NeuralNetwork(Chain(layers...) |> device)

        model = ContinuousActionValue(net)

    elseif ModelType == "EnvValue"
        rng = MersenneTwister(seed)
        model = EnvValue(model_local.env, model_local.Q, model_local.state_encoder, device, rng)

    elseif ModelType == "PolicyActionValue"
        if !tabular
            embedding_dim = 10

            A = feed_forward(
                input_dim + embedding_dim,
                output_dim,
                hidden_size,
                drop_rate = drop_rate,
                num_hidden_layers = num_layers,
                σ = activation,
                seed = seed,
            )
            layers = collect(A.layers)

            if :input_encoder in keys(model_local)
                pushfirst!(layers, model_local.input_encoder)
            end

            if :output_encoder in keys(model_local)
                push!(layers, model_local.output_encoder)
            end

            net = NeuralNetwork(Chain(layers...) |> device)
        else
            net = Tabular((input_dim, output_dim))
        end
        model = PolicyActionValue(net)

    elseif ModelType == "QRActionValue"
        N_quantiles = model_params.model_local.num_heads
        net = NeuralNetwork(
            feed_forward(
                input_dim,
                output_dim * N_quantiles,
                hidden_size,
                drop_rate = drop_rate,
                num_hidden_layers = num_layers,
                σ = activation,
                seed = seed,
            ) |> device,
        )
        # println(typeof(net.f.layers[1].W))
        model = QRActionValue(net, N = N_quantiles)

    elseif ModelType == "Value"
        if !tabular
            net = NeuralNetwork(
                feed_forward(
                    input_dim,
                    1,
                    hidden_size,
                    num_hidden_layers = num_layers,
                    σ = activation,
                    seed = seed,
                ) |> device,
            )
        else
            net = Tabular((input_dim,1))
        end
        model = Value(net)

    elseif ModelType == "Policy"
        if !tabular
            A = feed_forward(
                input_dim,
                output_dim,
                hidden_size,
                drop_rate = drop_rate,
                num_hidden_layers = num_layers,
                σ = activation,
                seed = seed,
                # output_a = Flux.tanh,
                output_a = Flux.sigmoid,
            )

            layers = collect(A.layers)

            if :input_encoder in keys(model_local)
                pushfirst!(layers, model_local.input_encoder)
            end

            if :output_encoder in keys(model_local)
                push!(layers, model_local.output_encoder)
            end

            net = NeuralNetwork(Chain(layers...) |> device)

        else
            net = Tabular((input_dim, output_dim))
        end
        model = Policy(net, discrete = discrete)

    elseif ModelType == "Persist"
        net = NeuralNetwork(
            feed_forward(
                input_dim,
                output_dim,
                hidden_size,
                num_hidden_layers = num_layers,
                σ = activation,
                seed = seed,
            ) |> device,
        )
        model =
            PersistActionValue(net, predict_window = predict_window, discrete = discrete)

    elseif ModelType == "RNNOpenLoop"
        net = NeuralNetwork(
            rnn_model(input_dim, output_dim, hidden_size, σ = activation) |> device,
        )
        model = RNNActionValue(
            net,
            history_window = history_window,
            predict_window = predict_window,
            open_loop = true,
            discrete = discrete,
        )

    elseif ModelType == "RNNClosedLoop"
        net = NeuralNetwork(rnn(input_dim) |> device)
        model = RNNActionValue(
            net,
            history_window = history_window,
            predict_window = predict_window,
            open_loop = false,
            discrete = discrete,
        )

    elseif ModelType == "RNNPlanner"
        @assert planning_algo !== nothing

        memory_dim = Int(ceil(hidden_size / 2))
        recurrence = RNN
        if recurrence == LSTM
            mem_factor = 2
        else
            mem_factor = 1
        end

        state_encoder =
            feed_forward(input_dim, mem_factor * memory_dim, hidden_size, σ = activation) |>
            device
        state_dynamics = rnn(hidden_size, memory_dim, recurrence = recurrence) |> device
        action_encoder =
            feed_forward(output_dim, hidden_size, hidden_size, σ = activation) |> device
        reward_head = feed_forward(memory_dim, 1, hidden_size, σ = activation) |> device

        net = NeuralNetwork((
            reward_head = reward_head,
            state_encoder = state_encoder,
            state_dynamics = state_dynamics,
            action_encoder = action_encoder,
        ))

        model = RNNPlanner(
            net,
            history_window = history_window,
            predict_window = predict_window,
            open_loop = false,
            discrete = discrete,
            action_dim = output_dim,
            planning_algo = planning_algo,
        )
    else
        error("Not a valid ModelType")
    end
    return model
end
