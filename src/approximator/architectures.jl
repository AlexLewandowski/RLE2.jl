import Flux: RNN, LSTM, GRU, Dropout, Conv, LayerNorm, SkipConnection, BatchNorm
import Flux: Dense, Chain, relu, softmax, Recur, logsoftmax
import Flux: glorot_uniform, glorot_normal

"Helper function for constructing feed-forward neural-network"
#TODO repdim?
function feed_forward(
    in_dim,
    out_dim,
    h_dim;
    num_hidden_layers = 0,
    σ = relu,
    output = nothing,
    output_a = identity,
    seed = nothing,
    rng = nothing,
    drop_rate = 0.0f0,
    initb = nothing,
    rep_dim = nothing,
    output_bias = true,
    input_batchnorm = false,
    skip_connection = false,
    layernorm = false,
)
    if isnothing(rep_dim)
        rep_dim = h_dim
    end

    if skip_connection == true
        if h_dim == out_dim == in_dim
            skip_connection = :only
        elseif h_dim == out_dim
            skip_connection = :external
        elseif h_dim == in_dim
            skip_connection = :input
        else
            skip_connection = :internal
        end
    end

    if layernorm == true
        if h_dim == out_dim == in_dim
            layernorm = :only
        elseif h_dim == out_dim
            layernorm = :external
        elseif h_dim == in_dim
            layernorm = :input
        else
            layernorm  = :internal
        end
    end


    initbf = (i) -> Flux.zeros32
    if !isnothing(seed)
        initW = (i) -> (x, y) -> glorot_uniform(MersenneTwister(seed + i), x, y)
        if !isnothing(initb)
            initbf = (i) -> (x) -> initb(MersenneTwister(seed + i), x)
        end
    elseif !isnothing(rng)
        initW = (i) -> (x, y) -> glorot_uniform(rng, x, y)
        if !isnothing(initb)
            initbf = (i) -> (x) -> initb(rng, x)
        end
    else
        initW = (i) -> glorot_uniform
    end

    layers = []
    i = 0

    if num_hidden_layers == 0
        out_h_dim = rep_dim
    else
        out_h_dim = h_dim
    end

    if input_batchnorm
        push!(layers, BatchNorm(in_dim))
    end
    layer = []

    if skip_connection == :only || skip_connection == :input
        layer_temp = Dense(in_dim, out_h_dim, σ, initW = initW(i), initb = initbf(i))
        push!(layer, SkipConnection(layer_temp, +))
    else
        push!(layer, Dense(in_dim, out_h_dim, σ, initW = initW(i), initb = initbf(i)))
    end

    if layernorm == :only || layernorm == :input
        push!(layer, LayerNorm(out_h_dim))
    end

    push!(layers, layer...)
    for i = 1:num_hidden_layers
        if i == num_hidden_layers
            out_h_dim = rep_dim
        else
            out_h_dim = h_dim
        end

        if drop_rate > 0
            push!(layers, Dropout(drop_rate))
        end

        layer = []

        if skip_connection == :only || skip_connection == :external || skip_connection == :internal || skip_connection == :input
            layer_temp =  Dense(h_dim, out_h_dim, σ, initW = initW(i), initb = initbf(i))
            push!(layer, SkipConnection(layer_temp, +))
        else
            push!(layer, Dense(h_dim, out_h_dim, σ, initW = initW(i), initb = initbf(i)))
        end

        if layernorm == :only || layernorm == :external || layernorm == :internal || layernorm == :input
            push!(layer, LayerNorm(out_h_dim))
        end

        push!(layers, layer...)
    end

    layer = []
    layer_temp = Dense(rep_dim, out_dim, output_a, initW = initW(i), initb = initbf(i), bias = output_bias)
    if skip_connection == :only || skip_connection == :external
        push!(layer, SkipConnection(layer_temp, +))
    else
        push!(layer, layer_temp)
    end

    layers = push!(
        layers,
        layer...
    )

    if layernorm == :only || layernorm == :external
        push!(layers, LayerNorm(out_h_dim))
    end

    if num_hidden_layers == -1
        i = 0
        layers = [Dense(in_dim, out_dim, output_a, initW = initW(i), initb = initbf(i), bias = output_bias)]
    end

    if output !== nothing
        push!(layers, output)
    end

    m = Chain(layers...)
    Flux.testmode!(m, true)
    return m
end

function LeNet5(; imgsize=(28,28,1), nclasses=10, output_activation = softmax)

    out_conv_size = (imgsize[1]÷4 - 3, imgsize[2]÷4 - 3, 16)
    reshape_layer = x -> reshape(x, (imgsize..., size(x)[end]))

    layer_l = [
            Conv((5, 5), imgsize[end]=>6, relu),
            Flux.MaxPool((2, 2)),
            Conv((5, 5), 6=>16, relu),
            Flux.MaxPool((2, 2)),
            Flux.flatten,
            Dense(prod(out_conv_size), 120, relu),
            Dense(120, 84, relu),
        Dense(84, nclasses),
        output_activation,
          ]
    return Chain(reshape_layer, layer_l...)
    end

function cnn(
    in_dim,
    out_dim,
    h_dim;
    num_hidden_layers = 3,
    σ = relu,
    output = nothing,
    seed = nothing,
    rng = nothing,
    drop_rate = 0.0f0,
    output_a = identity,
)
    if !isnothing(seed)
        initW = (i) -> (w, x, y, z) -> glorot_uniform(MersenneTwister(seed + i), w, x, y, z)
    elseif !isnothing(rng)
        initW = (i) -> (w, x, y, z) -> glorot_uniform(rng, w, x, y, z)
        # if !isnothing(initb)
        #     initbf = (i) -> (x) -> initb(rng, x)
        # end
    else
        initW = (i) -> glorot_uniform
    end

    layers = []
    i = 0
    L = 28
    W = L
    S = 3
    push!(layers, x -> reshape(x, (28, 28, in_dim, size(x)[end])))
    push!(layers, Conv((S, S), in_dim => h_dim, σ, init = initW(i)))
    push!(layers, Dropout(drop_rate))
    for i = 1:num_hidden_layers
        h_dim_out = Int(h_dim / 2)
        push!(layers, Conv((S, S), h_dim => h_dim_out, σ, init = initW(i)))
        push!(layers, Dropout(drop_rate))
        h_dim = h_dim_out
    end
    h_dim_out = Int(h_dim / 2)
    push!(layers, Conv((S, S), h_dim => h_dim_out, σ, init = initW(i)))
    h_dim = h_dim_out
    push!(layers, x -> reshape(x, (:, size(x)[end])))
    L = L - (S-1)*(num_hidden_layers + 2)
    push!(layers, Dense(L*L*h_dim, out_dim, output_a))

    if output !== nothing
        push!(layers, output)
    end

    m = Chain(layers...)
    Flux.testmode!(m, true)
    return m
end

"Helper function for constructing recurrent neural-network for use in model-based RL"
function rnn(
    in_dim,
    mem_dim = 32;
    num_recurrent_layers = 0,
    recurrence = RNN,
    σ = relu,
    seed = nothing,
    rng = nothing,
    initb = glorot_uniform,
)
    if !isnothing(seed)
        initW = (i) -> (x, y) -> glorot_uniform(MersenneTwister(seed + i), x, y)
    elseif !isnothing(rng)
        initW = (i) -> (x, y) -> glorot_normal(rng, x, y)
    else
        initW = (i) -> glorot_uniform
    end

    layers = []
    push!(layers, recurrence(in_dim, mem_dim, init = initW(0), initb = initb))
    for i = 1:num_recurrent_layers
        push!(layers, recurrence(mem_dim, mem_dim, init = initW(i), initb = initb))
    end
    m = Chain(layers...)
    Flux.testmode!(m, true)
    return m
end

function mha(
    in_dim,
    out_dim,
    h_dim;
    num_hidden_layers = 1,
    num_heads = 4,
    σ = relu,
    output = nothing,
    output_a = identity,
    seed = nothing,
    rng = nothing,
    drop_rate = 0.0f0,
    initb = nothing,
)

    heads = []
    for _ = 1:num_heads
        WQ = Dense(in_dim, h_dim)
        WK = Dense(in_dim, h_dim)
        WV = Dense(in_dim, h_dim)
        head = Chain(WQ, WK, WV)
        push!(heads, head)
    end

    output = Dense(num_heads * h_dim, out_dim)

    return MultiHeadAttention(Chain(output, heads))
end

function transformer_encoder(
    in_dim,
    out_dim,
    h_dim;
    num_hidden_layers = 0,
    num_heads = 4,
    σ = relu,
    output = nothing,
    seed = nothing,
    rng = nothing,
    drop_rate = 0.0f0,
    initb = nothing,
    output_layer = false,
    input_layer = true,
)
    if input_layer
        project_in = Chain(Dense(in_dim, h_dim, σ), LayerNorm(h_dim))
        # project_in = Chain(Dense(in_dim, h_dim), LayerNorm(h_dim))
        # project_in = Dense(in_dim, h_dim)
        latent_in_dim = h_dim
    else
        project_in = identity
        latent_in_dim = in_dim
    end
    at = MultiHeadAttention(latent_in_dim, latent_in_dim, h_dim, num_heads = num_heads)
    project = Dense(latent_in_dim, latent_in_dim, σ)
    layers = [
        project_in,
        SkipConnection(at, +),
        LayerNorm(latent_in_dim),
        SkipConnection(project, +),
        LayerNorm(latent_in_dim),
    ]

    for i = 1:num_hidden_layers
        at = MultiHeadAttention(latent_in_dim, latent_in_dim, h_dim, num_heads = num_heads)
        project = Dense(latent_in_dim, latent_in_dim, σ)

        sub_layers = [
            SkipConnection(at, +),
            LayerNorm(latent_in_dim),
            SkipConnection(project, +),
            LayerNorm(latent_in_dim),
        ]
        push!(
            layers,
            sub_layers...
        )
    end
    if output_layer
        output = Dense(latent_in_dim, out_dim)
        push!(layers, output)
    end

    return Chain(layers...)
end

function transformer_decoder(
    in_dim,
    out_dim,
    h_dim;
    num_hidden_layers = 0,
    num_heads = 4,
    σ = relu,
    output = nothing,
    seed = nothing,
    rng = nothing,
    drop_rate = 0.0f0,
    initb = nothing,
    output_layer = true,
    input_layer = true,
)
    if input_layer
        project_in = Chain(Dense(in_dim, h_dim, σ), LayerNorm(h_dim))
        # project_in = Chain(Dense(in_dim, h_dim), LayerNorm(h_dim))
        # project_in = Dense(in_dim, h_dim)
        latent_in_dim = h_dim
    else
        project_in = identity
        latent_in_dim = in_dim
    end

    at1 = MultiHeadAttention(latent_in_dim, latent_in_dim, h_dim, num_heads = num_heads)
    at2 = MultiHeadAttention(latent_in_dim, latent_in_dim, h_dim, num_heads = num_heads)
    project = Dense(latent_in_dim, latent_in_dim, σ)
    layers = [
        project_in,
        SkipConnection(at1, +),
        LayerNorm(latent_in_dim),
        SkipConnection(at2, +),
        LayerNorm(latent_in_dim),
        SkipConnection(project, +),
        LayerNorm(latent_in_dim),
    ]

    for i = 1:num_hidden_layers
        at1 = MultiHeadAttention(latent_in_dim, latent_in_dim, h_dim, num_heads = num_heads)
        at2 = MultiHeadAttention(latent_in_dim, latent_in_dim, h_dim, num_heads = num_heads)
        project = Dense(latent_in_dim, latent_in_dim, σ)

        sub_layers = [
            SkipConnection(at1, +),
            LayerNorm(latent_in_dim),
            SkipConnection(at2, +),
            LayerNorm(latent_in_dim),
            SkipConnection(project, +),
            LayerNorm(latent_in_dim),
        ]
        push!(
            layers,
            sub_layers...
        )
    end

    if output_layer
        output = Dense(latent_in_dim, out_dim)
        push!(layers, output)
    end

    return Chain(layers...)
end

function meta_state_encoder(ins, outs, embed_dim, seed; aux_dim = 0, aux_embed_dim = 32, σ = Flux.relu, pooling_f = :max)
    @assert length(ins) == length(outs)
    L = length(ins)
    input_modules = []
    num_layers = -1
    skip_connection = false
    layernorm = false
    for l = 1:L
        A_in = feed_forward(
            ins[l],
            Int(embed_dim / L),
            Int(embed_dim / L),
            seed = seed,
            output_a = σ,
            σ = σ,
            num_hidden_layers = num_layers,
            layernorm = layernorm,
            skip_connection = skip_connection,
        )
        # A_in = cnn(
        #     1,
        #     Int(input_embed_dim / L),
        #     input_embed_dim,
        #     seed = seed,
        #     output_a = Flux.relu,
        # )

        if outs[1] > 0
        A_out = feed_forward(
            outs[l],
            Int(embed_dim / L),
            2*Int(embed_dim / L),
            seed = seed,
            output_a = σ,
            σ = σ,
            num_hidden_layers = num_layers,
            layernorm = layernorm,
            skip_connection = skip_connection,
        )
        else
            A_out = identity
        end

        A_joint = feed_forward(
            Int(embed_dim / L),
            Int(embed_dim / L),
            Int(embed_dim / L),
            seed = seed,
            output_a =σ,
            σ = σ,
            num_hidden_layers = num_layers,
            layernorm = layernorm,
            skip_connection = skip_connection,
        )
        push!(input_modules, [A_in, A_out, A_joint])
    end

    post = true

    if post
    A_joint_post_pool = feed_forward(
        embed_dim,
        embed_dim,
        embed_dim,
        seed = seed,
        output_a = σ,
        output_bias = true, #TODO not for self-teaching RL
        σ = σ,
        num_hidden_layers = num_layers,
        layernorm = layernorm,
        skip_connection = skip_connection,)
    else
        A_joint_post_pool = identity
    end


    if pooling_f == :attention
        pooling = MultiHeadAttention(embed_dim, embed_dim, embed_dim, set_attention = true, num_heads = 4, output_a = relu)
    elseif pooling_f == :max
        pooling = x -> maximum(x, dims = 2)
    elseif pooling_f == :mean
        pooling = x -> mean(x, dims = 2)
    end

    return SequenceNeuralNetwork(Chain(input_modules, pooling, A_joint_post_pool))
end

function set_dropout!(f::Chain, drop_rate)
    for layer in f
        if typeof(layer) <: Flux.Dropout
            layer.p = drop_rate
        end
    end
end
