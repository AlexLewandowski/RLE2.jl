abstract type AbstractOptEnv <: AbstractEnv end

mutable struct OptEnv <: AbstractOptEnv
    state_representation::Any
    reward_function::Any
    obs::Any
    J::Any
    alpha::Any
    f::Any
    init_f::Any
    init_f2::Any
    old_f::Any
    fstar::Any
    h_dim::Any
    num_layers::Any
    activation::Any
    n_tasks::Any
    x::Any
    y::Any
    x_test::Any
    y_test::Any
    ind_range::Any
    batch_size::Any
    task_batch_size::Any
    inds::Any
    task_inds::Any
    opt::Any
    internal_opt::Any
    action_space::Any
    done::Any
    reward::Any
    termination_condition::Any
    success_condition::Any
    t::Any
    horizon::Any
    acc::Any
    in_dim::Any
    out_dim::Any
    data_size::Any
    dataset_name::Any
    max_steps::Any
    device::Any
    rng::Any
end

function OptEnv(
    state_representation,
    reward_function = "logLP",
    dataset_name = "synthetic",
    optimizer = "ADAM";
    continuous = false,
    in_dim = 10,
    out_dim = 10,
    h_dim = 128,
    num_layers = 1, #TODO Reduced num lauers
    activation = Flux.relu,
    batch_size = 64,
    data_size = 1000,
    max_steps = 200,
    device = Flux.gpu,
    rng
)
    internal_opt1 = Flux.ADAM()
    internal_opt2 = Flux.ADAM()
    internal_opt = [internal_opt1, internal_opt2]

    if optimizer == "ADAM"
        opt = Flux.ADAM()
    elseif optimizer == "SGD"
        opt = Flux.Descent()
    else
        error("Not a valid optimizer.")
    end

    n_tasks = 1
    J =
        (f, x, y) ->
            Flux.Losses.logitcrossentropy(log.(f(x) .+ eps(Float32)), y, agg = mean)
    if dataset_name == "syntheticClusterEasy"
        in_dim = 10
        out_dim = 10
        batch_size = 64
        data_size = 128
        termination_condition = 1 #TODO 1 sgd step per action
        success_condition = 0.5
        if optimizer == "ADAM"
            success_condition = 0.95
        end
        if contains(state_representation, "hardmax")
            success_condition = 0.95
        end

    elseif dataset_name == "sinWave"
        n_tasks = 32
        in_dim = 1
        out_dim = 1
        h_dim = 40
        num_layers = 1
        data_size = 20
        batch_size = data_size
        termination_condition = 1 #TODO 1 sgd step per action
        J = (f, x, y) -> mean((f(x) .- y) .^ 2)
        # J = (f, x, y) -> mean((f(x) .- y).^2)
        success_condition = 0.5
        if optimizer == "ADAM"
            success_condition = 0.95
        end
        if contains(state_representation, "hardmax")
            success_condition = 0.95
        end

    elseif contains(dataset_name, "syntheticCluster")
        in_dim = 10
        out_dim = 10
        batch_size = 64
        termination_condition = 1 #TODO 1 sgd step per action
        success_condition = 0.95
        if optimizer == "ADAM"
            # success_condition = 0.99 TODO
            success_condition = 0.95
        end
        if contains(state_representation, "hardmax")
            success_condition = 0.95
        end

    elseif dataset_name == "syntheticNN"
        n_tasks = 2
        in_dim = 784
        out_dim = 10
        data_size = 4000
        batch_size = 128
        h_dim = 128
        termination_condition = 1
        success_condition = 0.75

    elseif dataset_name == "mnist"
        in_dim = 784
        out_dim = 10
        data_size = 60000
        batch_size = 64
        h_dim = 128
        termination_condition = 1
        success_condition = 0.90

    elseif dataset_name == "mnistCNN"
        in_dim = 1
        out_dim = 10
        data_size = 60000
        batch_size = 64
        termination_condition = 1
        success_condition = 0.99

    elseif dataset_name == "fashion"
        in_dim = 784
        out_dim = 10
        data_size = 60000
        batch_size = 256
        h_dim = 128
        termination_condition = 1
        success_condition = 0.99

    elseif dataset_name == "fashionCNN"
        in_dim = 1
        out_dim = 10
        data_size = 60000
        batch_size = 256
        termination_condition = 1
        success_condition = 0.99

    elseif dataset_name == "cifar10"
        in_dim = 3
        out_dim = 10
        h_dim = 8
        data_size = 50000
        batch_size = 128
        termination_condition = 1
        success_condition = 0.40

    else
        error("Not a valid dataset name for OptEnv.")
    end

    task_batch_size = minimum([4, n_tasks])

    if contains(reward_function, "FiniteHorizon")
        println(reward_function)
        if length(reward_function) == 13
            success_condition = 10
        else
            success_condition = parse(Int, reward_function[14:end])
            max_steps = success_condition
        end

        reward_function = reward_function[1:13]
    end

    x = nothing
    y = nothing
    x_test = nothing
    y_test = nothing
    fstar = nothing
    f = init_nn(in_dim, out_dim, h_dim, activation, num_layers, rng, device, dataset_name)
    to_device!(f, device)

    if continuous
        action_space =
            ReinforcementLearningEnvironments.IntervalSets.ClosedInterval{Float32}(
                0.0f0,
                1.0f0,
            )
        # action_space = ReinforcementLearningEnvironments.Space([
        #     ReinforcementLearningEnvironments.IntervalSets.ClosedInterval{Float32}(
        #         0.0f0,
        #         1.0f0,
        #     )
        #     for i = 1:2
        # ])
    else
        action_set = collect(1:3) #TODO larger action set?
        action_space = Base.OneTo(length(action_set))
    end

    inds = nothing
    task_inds = nothing
    horizon = 0 #TODO remove, redundant with max_steps
    t = max_steps
    acc = [0.0f0, 0.0f0, 0.0f0]

    # opt = opt()
    alpha = 0.0f0

    p, re = Flux.destructure(f.f)
    old_f = NeuralNetwork(re(p))
    init_f = NeuralNetwork(re(p))
    init_f2 = NeuralNetwork(re(p))

    ind_range = 1:data_size

    env = OptEnv(
        state_representation,
        reward_function,
        nothing,
        J,
        alpha,
        f,
        init_f,
        init_f2,
        old_f,
        fstar,
        h_dim,
        num_layers,
        activation,
        n_tasks,
        x,
        y,
        x_test,
        y_test,
        ind_range,
        batch_size,
        task_batch_size,
        inds,
        task_inds,
        opt,
        internal_opt,
        action_space,
        false,
        0,
        termination_condition,
        success_condition,
        t,
        horizon,
        acc,
        in_dim,
        out_dim,
        data_size,
        dataset_name,
        max_steps,
        device,
        rng,
    )
    reset!(env, reinit = true)
    return env
end

Base.show(io::IO, t::MIME"text/plain", env::OptEnv) = begin
    println()
    println("---------------------------")
    name = "OptEnv"
    L = length(name)
    println(name)
    println(join(["-" for i = 1:L+1]))

    print(io, "  env.dataset_name: ")
    println(io, env.dataset_name)

    print(io, "  env.opt: ")
    println(io, typeof(env.opt))
    println("---------------------------")
end

function to_device!(env::AbstractOptEnv, device)
    env.device = device

    env.x = env.x |> env.device
    env.y = env.y |> env.device
    env.x_test = env.x_test |> env.device
    env.y_test = env.y_test |> env.device

    to_device!(env.f, env.device)
end

function init_nn(env)
    in_dim = env.in_dim
    out_dim = env.out_dim
    h_dim = env.h_dim
    activation = env.activation
    num_layers = env.num_layers
    rng = env.rng
    device = env.device
    dataset_name = env.dataset_name
    return init_nn(
        in_dim,
        out_dim,
        h_dim,
        activation,
        num_layers,
        rng,
        device,
        dataset_name,
    )
end


function init_nn(
    in_dim,
    out_dim,
    h_dim,
    σ,
    num_hidden_layers,
    rng,
    device,
    dataset_name;
    drop_rate = 0.0f0
)
    if contains(dataset_name, "CNN")
        f = NeuralNetwork(LeNet5())
    elseif dataset_name == "cifar10"
        f = NeuralNetwork(LeNet5(imgsize = (32, 32, 3)))
    else
        # h_dim = rand(env.rng, [16, 32, 64, 128, 256])
        # h_dim = 16
        output_act = softmax
        if dataset_name == "sinWave"
            function clamp_sin(x)
                return clamp.(x, -5.0f0, 5.0f0)
                # return 5f0*Flux.tanh.(x)
            end

            # output_act = clamp_sin
            output_act = identity
        end

        f = NeuralNetwork(
            feed_forward(
                in_dim,
                out_dim,
                h_dim,
                σ = Flux.relu,
                num_hidden_layers = num_hidden_layers,
                rng = rng,
                drop_rate = 0.0f0,
                output = output_act,
            ),
        )
    end
    # f = NeuralNetwork(LeNet5())
    to_device!(f, device)
    return f
end

function generate_sin_targets(x, task_id)
    rng = MersenneTwister(task_id)

    a = (5.0f0 - 0.1f0) * rand(rng, Float32) + 0.1f0
    b = Base.π * rand(rng, Float32)

    return a .* sin.(x .+ b)
end


function get_data(
    dataset_name,
    data_size,
    in_dim,
    out_dim,
    h_dim,
    num_layers,
    n_tasks;
    rng,
    device,
    task_id = nothing
)

    if isnothing(task_id)
        task_id = StatsBase.sample(rng, 1:n_tasks)
    end

    fstar = nothing
    if dataset_name == "syntheticNN"

        fstar = NeuralNetwork(
            feed_forward(
                in_dim,
                out_dim,
                4 * h_dim,
                num_hidden_layers = num_layers,
                σ = tanh,
                # rng = rng,
                seed = task_id,
                drop_rate = 0.0f0,
                output_bias = false,
            ),
        )

        to_device!(fstar, device)
        if device == Flux.gpu
            x = CUDA.randn(Float32, (in_dim, data_size))
        else
            x = randn(rng, Float32, (in_dim, data_size))
        end

        y = fstar(x) |> Flux.cpu
        x = x |> Flux.cpu

        test_data_size = data_size

        if device == Flux.gpu
            x_test = CUDA.randn(Float32, (in_dim, test_data_size))
        else
            x_test = randn(rng, Float32, (in_dim, test_data_size))
        end
        y_test = fstar(x_test) |> Flux.cpu

        y = Flux.onehotbatch(
            reshape([ind.I[1] for ind in argmax(y, dims = 1)], :),
            1:out_dim,
        )
        y_test = Flux.onehotbatch(
            reshape([ind.I[1] for ind in argmax(y_test, dims = 1)], :),
            1:10,
        )

        fstar = fstar

    elseif contains(dataset_name, "sinWave")

        x = Float32.(reshape((5 - (-5)) .* rand(rng, data_size) .- 5.0f0, (1, data_size)))
        x_test =
            Float32.(reshape((5 - (-5)) .* rand(rng, data_size) .- 5.0f0, (1, data_size)))

        y = generate_sin_targets(x, task_id)
        y_test = generate_sin_targets(x_test, task_id)

    elseif contains(dataset_name, "syntheticCluster")
        x = randn(rng, Float32, (in_dim, data_size))
        y = []
        inds = []
        for j = 1:data_size
            i = argmax(x[:, j])
            z = zeros(Float32, in_dim)
            z[i] = 1.0f0
            push!(inds, i)
            push!(y, z)
        end

        ind_sort = sortperm(inds)
        x = x[:, ind_sort]
        y = y[ind_sort]

        y = hcat(y...)

        test_data_size = data_size
        x_test = randn(rng, Float32, (in_dim, test_data_size))
        y_test = []
        inds_test = []
        for j = 1:test_data_size
            i = argmax(x_test[:, j])
            z = zeros(Float32, in_dim)
            z[i] = 1.0f0
            push!(y_test, z)
            push!(inds_test, i)
        end

        ind_test_sort = sortperm(inds_test)
        x_test = x_test[:, ind_test_sort]
        y_test = y_test[ind_test_sort]

        y_test = hcat(y_test...)

        y =
            Float32.(
                Flux.onehotbatch(
                    reshape([ind.I[1] for ind in argmax(y, dims = 1)], :),
                    1:out_dim,
                ),
            )
        y_test =
            Float32.(
                Flux.onehotbatch(
                    reshape([ind.I[1] for ind in argmax(y_test, dims = 1)], :),
                    1:10,
                ),
            )

        fstar = nothing

    elseif contains(dataset_name, "mnist")
        train_data_size = 10000
        out_dim = 10
        xs, ys = MLDatasets.MNIST.traindata()
        ys .+= 1
        y =
            Flux.onehotbatch(reshape([y for y in ys[1:train_data_size]], :), 1:out_dim) |> device
        x = Float32.(reshape(xs[:, :, 1:train_data_size], (:, train_data_size)))

        test_data_size = 1000
        xs, ys = MLDatasets.MNIST.testdata()
        ys .+= 1
        y_test = Flux.onehotbatch(reshape([y for y in ys[1:test_data_size]], :), 1:out_dim)
        x_test = Float32.(reshape(xs[:, :, 1:test_data_size], (:, test_data_size)))

    elseif contains(dataset_name, "fashion")
        train_data_size = 10000
        out_dim = 10
        xs, ys = MLDatasets.FashionMNIST.traindata()
        ys .+= 1
        y =
            Flux.onehotbatch(reshape([y for y in ys[1:train_data_size]], :), 1:out_dim) |> device
        x = Float32.(reshape(xs[:, :, 1:train_data_size], (:, train_data_size)))

        xs, ys = MLDatasets.FashionMNIST.testdata()
        ys .+= 1
        test_data_size = 1000
        y_test = Flux.onehotbatch(reshape([y for y in ys[1:test_data_size]], :), 1:out_dim)
        x_test = Float32.(reshape(xs[:, :, 1:test_data_size], (:, test_data_size)))

    elseif contains(dataset_name, "cifar10")
        train_data_size = 10000
        out_dim = 10
        xs, ys = MLDatasets.CIFAR10.traindata()
        ys .+= 1
        y =
            Flux.onehotbatch(reshape([y for y in ys[1:train_data_size]], :), 1:out_dim) |> device
        x = Float32.(reshape(xs[:, :, :, 1:train_data_size], (:, train_data_size)))

        xs, ys = MLDatasets.CIFAR10.testdata()
        ys .+= 1
        test_data_size = 1000
        y_test = Flux.onehotbatch(reshape([y for y in ys[1:test_data_size]], :), 1:out_dim)
        x_test = Float32.(reshape(xs[:, :, :, 1:test_data_size], (:, test_data_size)))

    else
        error("Not a valid dataset name")
    end

    return x, y, x_test, y_test, fstar
end


function init_data!(env, data_size, in_dim, out_dim, h_dim, num_layers; rng, device)
    dataset_name = env.dataset_name
    env.fstar = nothing

    # x, y, x_test, y_test, fstar = get_data(dataset_name, data_size, in_dim,
    #                                        out_dim, h_dim, num_layers; rng, device)

    n_tasks = env.n_tasks
    xs = []
    ys = []
    x_tests = []
    y_tests = []
    fstar = nothing
    for task_id = 1:n_tasks
        x, y, x_test, y_test, fstar = get_data(
            env.dataset_name,
            env.data_size,
            env.in_dim,
            env.out_dim,
            env.h_dim,
            env.num_layers,
            env.n_tasks;
            rng = env.rng,
            device = env.device,
            task_id = task_id
        )
        push!(xs, x)
        push!(x_tests, x_test)
        push!(ys, y)
        push!(y_tests, y_test)
    end

    x = cat(xs..., dims = 2)
    y = cat(ys..., dims = 2)
    x_test = cat(x_tests..., dims = 2)
    y_test = cat(y_tests..., dims = 2)

    env.fstar = fstar
    env.x = x |> env.device
    env.y = y |> env.device
    env.x_test = x_test |> env.device
    env.y_test = y_test |> env.device

    return nothing
end


function env_mode!(env::AbstractOptEnv; mode = :wide)
    if mode == :wide
        h_dim = 2 * env.h_dim
        num_layers = Int(floor(env.num_layers / 2))
        activation = Flux.relu
        activation = tanh
    elseif mode == :tanh
        h_dim = env.h_dim
        num_layers = env.num_layers
        activation = tanh
    elseif mode == :narrow
        h_dim = Int(env.h_dim / 2)
        num_layers = 2 * env.num_layers
        activation = Flux.relu
        activation = tanh
    else
        error("Not a valid env mode for OptEnv")
    end

    env = OptEnv(
        env.state_representation,
        env.reward_function,
        env.dataset_name,
        h_dim = h_dim,
        num_layers = num_layers,
        activation = activation,
        rng = env.rng,
    )

    return env
end

function loss(env::AbstractOptEnv)
    f = env.f
    J = env.J
    x = env.x
    y = env.y
    J(f, x, y)
end

function retrieve_with_inds(env, arr, inds; eval = false)
    return arr[:, inds]
    # return arr[fill(:, length(size(arr)) - 1)..., inds]
end

function calc_norm(env::AbstractOptEnv; mode = :test)
    f = env.f
    N = size(env.x)[end]
    x = env.x
    N = size(x)[end]
    M = 1000
    if N < M
        inds = collect(1:N)
    else
        inds = rand(env.rng, 1:N, M) # Subsample
    end
    x = retrieve_with_inds(env, env.x, inds, eval = true)# |> env.device
    y = retrieve_with_inds(env, env.y, inds, eval = true)# |> env.device
    preds = f(x) |> Flux.cpu
    return mean(sum(preds .* log.(preds .+ eps(Float32)), dims = 1))
end

# function calc_transfer(env::AbstractOptEnv; mode = :test)
#     env_copy = deepcopy(env)
#     reset!(env_copy)
#     train_loop(env_copy)
#     return calc_performance(env_copy; mode = :test)
# end

function pretrain_f(env::AbstractOptEnv; task_id = nothing)

    if isnothing(task_id)
        reset!(env, saved_f = false)
        N = size(env.x)[end]
        env.ind_range = 1:N
    else
        reset!(env, saved_f = false, task_id = task_id)
    end

    og_opt = env.opt
    env.opt = Flux.ADAM()

    env.inds = StatsBase.sample(env.rng, env.ind_range, env.batch_size, replace = false)

    println("INIT: ", calc_performance(env, mode = :train, for_reward = false))
    for i = 1:10000
        if mod(i, 1000) == 0
            # println(env.inds)
            println(
                "PRETRAINING: ",
                calc_performance(env, mode = :train, for_reward = false),
            )
        end

        train_loop(env)
    end
    env.opt = og_opt

    println("PRETRAINED: ", calc_performance(env, mode = :train, for_reward = false))

end


function calc_performance(
    env::AbstractOptEnv;
    mode = :test,
    f = nothing,
    all_data = false,
    for_reward = true
)
    if isnothing(f)
        f = env.f
    end
    N = size(env.x)[end]
    if mode == :test
        N = size(env.x_test)[end]
        M = 1024
        ind_range = env.ind_range
        if all_data
            ind_range = 1:N
        end

        inds = rand(env.rng, ind_range, M) # Subsample
        inds = collect(ind_range)
        x = retrieve_with_inds(env, env.x_test, inds, eval = true)# |> env.device
        y = retrieve_with_inds(env, env.y_test, inds, eval = true)# |> env.device

    elseif mode == :train
        x = env.x
        N = size(x)[end]
        M = 1024
        ind_range = env.ind_range
        if all_data
            ind_range = 1:N
        end
        if N < M
            inds = collect(ind_range)
        else
            inds = rand(env.rng, ind_range, M) # Subsample
        end
        x = retrieve_with_inds(env, env.x, inds, eval = true)# |> env.device
        y = retrieve_with_inds(env, env.y, inds, eval = true)# |> env.device
    end
    if env.dataset_name == "sinWave"
        if for_reward
            scale = 10
            r = env.J(f, x, y)

            # return -minimum([r, scale])/scale
            reward = -1 / (1 / r + 1)
            # reward = -r
            # return reward

            # println(reward)
            # return reward
        else
            return env.J(f, x, y)
        end
    else
        preds = f(x) |> Flux.cpu
        y = y |> Flux.cpu
        preds = argmax(preds, dims = 1)
        y = argmax(y, dims = 1)
        return mean(preds .== y)
    end

end


function update_f(env, opt, J, f, x, y)
    p, re = Flux.destructure(f)
    env.old_f = NeuralNetwork(re(p))
    gs = gradient(Flux.params(f)) do
        J(f, x, y)# + env.alpha*sum(LinearAlgebra.norm, env.f.params)
    end
    Flux.Optimise.update!(opt, Flux.params(f), gs)
end

function loglp(env, acc_new, acc_old; normalized = true, gamma = 1.0f0)
    if normalized
        neg_loglp =
            (
                gamma * log(1.0f0 - acc_new + eps(Float32)) -
                log(1.0f0 - acc_old + eps(Float32))
            ) / log(1.0f0 - env.success_condition + eps(Float32))
    else
        neg_loglp = -(
            gamma * log(1.0f0 - acc_new + eps(Float32)) -
            log(1.0f0 - acc_old + eps(Float32))
        )
    end
    return neg_loglp
end

function train_loop(env::AbstractOptEnv)
    N = size(env.x)[end]
    for i = 1:env.termination_condition
        x = retrieve_with_inds(env, env.x, env.inds)# |> env.device
        y = retrieve_with_inds(env, env.y, env.inds)# |> env.device

        update_f(env, env.opt, env.J, env.f.f, x, y)
        env.acc[2] = env.J(env.f, x, y)
        env.inds = StatsBase.sample(env.rng, env.ind_range, env.batch_size, replace = false)
    end
end

function (env::AbstractOptEnv)(a)

    if typeof(env.action_space) <: Base.OneTo
        # action_set = collect(1:5)
        # action_set = collect(1:9)
        # action_set = collect(5:9)
        # a = 2f0^-(action_set[a])
    else
        a = clamp.(a, 0.0f0, 1.0f0)
        env.opt.eta = 0.999f0 * a
    end

    f = env.f
    J = env.J

    if typeof(env.opt) <: Flux.ADAM
        # L1 = 7/4f0
        L1 = 2.0f0
        L2 = 5.0
        if env.dataset_name == "cifar10"
            L1 = 1.5
        end
        if env.dataset_name == "syntheticCluster" &&
           typeof(env.opt) <: Flux.ADAM &&
           env.max_steps < 201
            L1 = 1.5
        end

    elseif typeof(env.opt) <: Flux.Descent
        L1 = 2.0
        L2 = 5.0
    end

    if a == 1
        env.opt.eta = (env.opt.eta) / L1
    elseif a == 2
        env.opt.eta = (env.opt.eta) * L1
    elseif a == 3
    end
    # if a == 1
    #     env.opt.eta = (env.opt.eta) * L2
    # elseif a  == 2
    #     env.opt.eta = (env.opt.eta) * L1
    # elseif a == 3
    # elseif a == 4
    #     env.opt.eta = (env.opt.eta) / L1
    # elseif a == 5
    #     env.opt.eta = (env.opt.eta) / L2
    # end

    env.opt.eta = maximum([env.opt.eta, eps(Float32)])
    env.opt.eta = minimum([env.opt.eta, 0.5f0])
    # env.opt.eta = maximum([env.opt.eta, 10.0f0^-10])

    acc_old = env.acc[1]
    acc_old_test = env.acc[3]

    # for _ = 1:(env.success_condition - 1)
    train_loop(env)
    env.t -= 1
    # end


    acc_new = calc_performance(env, mode = :train)
    acc_new_test = calc_performance(env, mode = :test)

    env.obs = get_next_obs(env)
    env.acc[1] = acc_new
    env.acc[3] = acc_new_test

    # if !contains(env.dataset_name, "syntheticCluster") && !contains(env.dataset_name, "mnist")
    # # if !contains(env.dataset_name, "syntheticCluster") && !contains(env.dataset_name, "mnist") && !contains(env.dataset_name, "syntheticNN")
    #     acc_new = acc_new_test
    #     acc_old = acc_old_test
    # end

    if env.reward_function == "FiniteHorizon"
        if env.t == 0
            env.done = true
        end
    else
        if acc_new >= env.success_condition
            env.done = true
        end
    end


    beta = 1.0f0

    if env.reward_function == "NoLP"
        env.reward = -1.0f0
    elseif env.reward_function == "Acc"
        env.reward = -1.0f0 + acc_new
    elseif env.reward_function == "logAcc"
        log_acc = log(1 - acc_new) / log(1 - env.success_condition)
        env.reward = -1.0f0 + log_acc
    elseif env.reward_function == "LP"
        LP = 0.99f0 * acc_new - acc_old
        LP = beta * LP
        # env.reward = -1.0f0/10 + LP
        env.reward = -1.0f0 + LP
        # env.reward = LP
    elseif env.reward_function == "logLP"
        LP = loglp(env, acc_new, acc_old, gamma = 0.99f0)
        LP = beta * LP
        env.reward = -1.0f0 + LP
    elseif env.reward_function == "BinaryLP"
        LP = acc_new - acc_old > 0 ? 1.0f0 : -1.0f0
        env.reward = -1.0f0 + LP
    elseif env.reward_function == "logAccOnly"
        env.reward = -log(1 - acc_new)
    elseif env.reward_function == "AccOnly"
        env.reward = acc_new
    elseif env.reward_function == "LPOnly"
        LP = acc_new - acc_old
        env.reward = LP
    elseif env.reward_function == "logLPOnly"
        LP = loglp(env, acc_new, acc_old, normalized = true)
        env.reward = LP
    elseif env.reward_function == "BinaryLPOnly"
        LP = acc_new - acc_old > 0 ? 1.0f0 : -1.0f0
        env.reward = LP
    elseif env.reward_function == "FiniteHorizon"
        # LP = 0.99*acc_new - acc_old

        env.reward = acc_new
        # env.reward = acc_new_test
        # env.reward = 0f0
        # env.reward = LP
    else
        error("Not a valid reward function")
    end

    if env.done
        # env.reward += 1.0f0
        env.reward = 0.0f0

        if env.reward_function == "FiniteHorizon"
            env.reward = acc_new
            # env.reward = acc_new_test
        end
    end

    # if any(isnan.(get_params_vector(env.f)))
    #     env.done = true
    #     env.reward = -200f0
    # elseif any(isinf.(get_params_vector(env.f)))
    #     env.done = true
    #     env.reward = -200f0
    # elseif any(isinf.(env.obs))
    #     env.done = true
    #     env.reward = -200f0
    # elseif any(isnan.(env.obs))
    #     env.done = true
    #     env.reward = -200f0
end

function reset!(
    env::OptEnv;
    reinit = false,
    agent = nothing,
    saved_f = false,
    task_id = nothing,
    t = nothing
)
    in_dim = env.in_dim
    out_dim = env.out_dim
    device = env.device
    h_dim = env.h_dim
    num_layers = env.num_layers
    eta = env.opt.eta
    env.opt = typeof(env.opt)()

    if env.dataset_name == "sinWave" && typeof(env.opt) <: Flux.Descent
        env.opt.eta = 0.01
    end

    env.alpha = 0.0f0
    env.done = false
    data_size = env.data_size
    rng = env.rng

    if reinit == true
        # env.dataset_name == "syntheticNN" ||
        # env.dataset_name == "sinWave" ||
        # env.dataset_name == "syntheticCluster" ||
        init_data!(
            env,
            env.data_size,
            env.in_dim,
            env.out_dim,
            env.h_dim,
            env.num_layers;
            rng = env.rng,
            device = env.device
        )
    end

    N = size(env.x)[end]
    if env.dataset_name == "sinWave" || env.dataset_name == "syntheticNN"
        n_tasks = env.n_tasks
        if isnothing(task_id)
            env.task_inds = StatsBase.sample(env.rng, 1:env.n_tasks, env.task_batch_size, replace = false)
            task_id = StatsBase.sample(env.rng, env.task_inds)
        end
        # println(task_id)
        N2 = Int(N / n_tasks)
        ind_range = (1+N2*(task_id-1)):N2*task_id
    else
        ind_range = 1:N
    end

    env.ind_range = ind_range

    env.inds = StatsBase.sample(env.rng, env.ind_range, env.batch_size, replace = false)

    if saved_f == :new
        p, re = Flux.destructure(env.init_f.f)
        f = NeuralNetwork(re(p))

    elseif saved_f == :old
        p, re = Flux.destructure(env.init_f2.f)
        f = NeuralNetwork(re(p))

    else
        f = init_nn(env)
    end

    env.f = f
    to_device!(env.f, device)

    env.acc = [0.0f0, 0.0f0, 0.0f0]
    env.acc[1] = calc_performance(env, mode = :train)
    env.acc[3] = calc_performance(env, mode = :test)
    if isnothing(t)
        env.t = env.max_steps
    else
        env.t = t
    end

    env.obs = get_next_obs(env)
    return nothing
end

function optimal_action(env::AbstractOptEnv, s)
    return 4
end

function Random.seed!(env::AbstractOptEnv, seed) end

function optimize_student_metrics(
    agent,
    en::AbstractOptEnv;
    n_steps = 1000,
    return_gs = false,
    f = :new,
    value = false
)
    adapted_performance = []
    performance = []

    adapted_test_performance = []
    test_performance = []
    env = deepcopy(en)

    p, re = Flux.destructure(env.init_f.f)
    env.init_f = NeuralNetwork(re(p))


    # for i in env.task_inds
    for i in 1:4
        time_lim = env.success_condition #TODO
        # time_lim = 10
        reset!(env, task_id = i, saved_f = false)
        ep = generate_episode(
            env,
            agent,
            policy = :default,
            max_steps = time_lim,
            state = :dontreset,
        )

        if value
            train_perf = sum([e.r for e in ep])
            test_perf = train_perf
        else
            train_perf = calc_performance(env, mode = :train, f = env.f, for_reward = false)
            test_perf = calc_performance(env, mode = :test, f = env.f, for_reward = false)
        end

        push!(performance, train_perf)
        push!(test_performance, test_perf)


        reset!(env, task_id = i, saved_f = f)
        ep = generate_episode(
            env,
            agent,
            policy = :default,
            max_steps = time_lim,
            state = :dontreset,
        )
        if value
            train_perf = sum([e.r for e in ep])
            test_perf = train_perf
        else
            train_perf = calc_performance(env, mode = :train, f = env.f, for_reward = false)
            test_perf = calc_performance(env, mode = :test, f = env.f, for_reward = false)
        end
        push!(adapted_performance, train_perf)
        push!(adapted_test_performance, test_perf)
    end
    rs = [
        mean(adapted_performance),
        mean(adapted_test_performance),
        mean(performance),
        mean(test_performance),
    ]
    names = [
        "adapted_performance",
        "adapted_test_performance",
        "nonadapted_performance",
        "nonadapted_test_performance",
    ]
    return rs, names
end


function fomaml_student(
    env::AbstractOptEnv,
    buffer;
    n_steps = 200,
    return_gs = false,
    greedy = false
)
    in_dim = env.in_dim
    out_dim = env.out_dim
    device = env.device
    h_dim = env.h_dim
    num_layers = env.num_layers
    env.opt = typeof(env.opt)()
    env.alpha = 0.0f0
    env.done = false
    data_size = env.data_size
    rng = env.rng

    @assert env.state_representation == "parameters"

    opt = env.internal_opt[1]

    last_ep_idx = buffer._buffer_idx - 1
    if last_ep_idx == 0
        last_ep_idx = buffer.max_num_episodes
    end

    ps = buffer._episodes[last_ep_idx][end].op[1:end-1]
    _, re = Flux.destructure(env.init_f.f)
    temp_f = re(ps) |> env.device

    xs = []
    ys = []
    N = size(env.x)[end]
    for task_id in env.task_inds

        reset!(env, task_id = task_id)
        N2 = Int(N / env.n_tasks)
        ind_range = (1+N2*(task_id-1)):N2*task_id
        inds = stop_gradient() do
            StatsBase.sample(env.rng, ind_range, env.batch_size, replace = false)
        end

        x = retrieve_with_inds(env, env.x, inds)# |> env.device
        y = retrieve_with_inds(env, env.y, inds)# |> env.device

        push!(xs, x)
        push!(ys, y)
    end

    xs = cat(xs..., dims = 2)
    ys = cat(ys..., dims = 2)

    gs = Flux.gradient(Flux.params(temp_f)) do
        env.J(temp_f, xs, ys)# + env.alpha*sum(LinearAlgebra.norm, env.f.params)
    end
    println("FOMAML LOSS: ", env.J(temp_f, xs, ys))

    new_gs = Zygote.Grads(IdDict(), Flux.params(env.init_f.f))
    num_ps = length(Flux.params(env.init_f.f))

    for i = 1:num_ps
        old_f_p = env.init_f.params[i]
        temp_f_p = Flux.params(temp_f)[i]

        new_gs.grads[old_f_p] = gs.grads[temp_f_p]
    end

    Flux.Optimise.update!(opt, Flux.params(env.init_f.f), new_gs)

    env.f = deepcopy(env.init_f)

    env.obs = get_next_obs(env)

    if return_gs
        return new_gs
    end
end

function get_v(env, agent, f, task_id; t = nothing)
    if isnothing(t)
        t = env.max_steps
    end
    s = nothing
    if env.state_representation == "parameters"
        s = vcat(s_p, Float32.(log(env.opt.eta)))
    else
        s = get_next_obs_with_f(env, f; task_id = task_id, t = t)
    end

    V = agent.subagents[1](s |> agent.device)
    @assert length(V) == 1
    return V
end

function get_mean_v(env, agent, f; deterministic = false, t = nothing)
    if isnothing(t)
        t = env.max_steps
    end
    J = 0.0f0
    n_task_sampled = minimum([4, env.n_tasks])

    tasks = stop_gradient() do
        # StatsBase.sample(env.rng, env.task_inds, 4)
        StatsBase.sample(env.rng, 1:env.n_tasks, n_task_sampled, replace = false)
    end
    # tasks = env.task_inds

    #     tasks = 1:env.n_tasks
    if deterministic
        tasks = 1:env.n_tasks
    end

    # println(tasks)
    for task_id in tasks
        J += sum(get_v(env, agent, f, task_id, t = t))
    end
    return J / length(tasks)
end

function get_sum_grad(env, agent, f, ps; deterministic = false, t = nothing)
    if isnothing(t)
        t = env.max_steps
    end

    gs = Flux.gradient(
        () -> begin
            -get_mean_v(env, agent, f, deterministic = deterministic, t = t)
        end,
        ps,
    )
    return gs
end

using Optim, FluxOptTools
function optimize_student(
    agent,
    env::AbstractOptEnv;
    n_steps = 10,
    return_gs = false,
    greedy = false,
    cold_start = false,
    all_data = true,
    t = nothing
)
    device = agent.device
    data_size = env.data_size
    rng = env.rng

    # @assert env.t == env.max_steps
    t = env.max_steps

    # if !isnothing(t)
    #     env.t = t
    # end
    # env.t = 8
    # env.t = StatsBase.sample(env.rng, 1:10)

    # env.internal_opt = typeof(env.internal_opt)()
    # env.internal_opt.eta /= 10

    default_f = init_nn(env)

    if cold_start
        println("Cold")
        f = default_f
        env.init_f = f
        opt = typeof(env.internal_opt[1])()
        env.internal_opt[1] = opt
    else
        f = env.init_f #TODO cold-start coniditons?
        v_init = sum(agent.subagents[1](get_next_obs_with_f(env, f, t = t) |> agent.device))
        # opt = typeof(env.internal_opt)()
        # env.internal_opt = opt
        # opt = env.internal_opt[1]
    end

    # println(i)
    ps = nothing
    s_p = nothing
    if env.state_representation == "parameters"
        s_p, re = Flux.destructure(f.f)
        ps = Flux.params(s_p)
    else
        ps = f.params
    end


    # gs = get_sum_grad(env, agent, f, ps)
    # G_norm_init = sum(LinearAlgebra.norm, gs)

    # v_init = get_mean_v(env, agent, env.init_f, deterministic = true)
    # v_init_old = get_mean_v(env, agent, env.init_f2, deterministic = true)
    # # v_default = get_mean_v(env, agent, default_f, deterministic = true)

    # rs, names = optimize_student_metrics(agent, env, f = :new, value = true)
    # mc_v_init, _, _, _ = rs

    # rs, names = optimize_student_metrics(agent, env, f = :old, value = true)
    # mc_v_init_old, _, _, _ = rs

    # println("MC V Init: ", mc_v_init)
    # println("MC V Init Old: ", mc_v_init_old)

    # println("V Init: ", v_init)
    # println("V Init Old: ", v_init_old)

    # error()

    if !cold_start
        # if mc_v_init_old > mc_v_init# && G_norm_init < 1e-5 # TODO these reset conditions arer worse more often than not
        # if mc_v_init_old - mc_v_init > 0.5f0# && G_norm_init < 1e-5 # TODO these reset conditions arer worse more often than not
        if false
            v_init = v_init_old
            println("ROLLBACK")
            p, re = Flux.destructure(env.init_f2.f)
            env.init_f = NeuralNetwork(re(p))
            f = env.init_f

            env.internal_opt[1] = deepcopy(env.internal_opt[2])

            opt = env.internal_opt[1]
            old_opt = env.internal_opt[2]

            old_ps = env.init_f2.params
            new_ps = env.init_f.params

            if typeof(opt) <: Flux.ADAM
                if !isempty(opt.state)
                    opt.state = IdDict(new_p => (deepcopy(old_opt.state[old_p][1]), deepcopy(old_opt.state[old_p][2]), deepcopy(old_opt.state[old_p][3])) for (new_p, old_p) in zip(new_ps, old_ps))
                end
            elseif typeof(opt) <: Flux.RMSProp
                if !isempty(opt.acc)
                    opt.acc = IdDict(new_p => deepcopy(old_opt.acc[old_p]) for (new_p, old_p) in zip(new_ps, old_ps))
                end
            end

            ps = nothing
            s_p = nothing
            if env.state_representation == "parameters"
                s_p, re = Flux.destructure(f.f)
                ps = Flux.params(s_p)
            else
                ps = f.params
            end
        else
            p, re = Flux.destructure(env.init_f.f)
            env.init_f2 = NeuralNetwork(re(p))
            env.internal_opt[2] = deepcopy(env.internal_opt[1])
            opt = env.internal_opt[2]
            old_opt = env.internal_opt[1]

            old_ps = env.init_f.params
            new_ps = env.init_f2.params

            if typeof(opt) <: Flux.ADAM
                if !isempty(opt.state)
                    opt.state = IdDict(new_p => (deepcopy(old_opt.state[old_p][1]), deepcopy(old_opt.state[old_p][2]), deepcopy(old_opt.state[old_p][3])) for (new_p, old_p) in zip(new_ps, old_ps))
                end
            elseif typeof(opt) <: Flux.RMSProp
                if !isempty(opt.acc)
                    opt.acc = IdDict(new_p => deepcopy(old_opt.acc[old_p]) for (new_p, old_p) in zip(new_ps, old_ps))
                end
            end

            opt = env.internal_opt[1]
        end
    end
    # opt = typeof(env.internal_opt[1])()

    # println("V init: ", v_init)
    # println("V default: ", v_default)

    # gs = get_sum_grad(env, agent, f, ps)
    # G_norm_init = sum(LinearAlgebra.norm, gs)
    # println("G_norm init: ", G_norm_init)

    gs = nothing
    num_reports = 10
    if n_steps > 10
        report_freq = Int(n_steps / num_reports)
    else
        report_freq = 10
    end

    ###
    ### LBFGS
    ###
    # old_env_device = env.device
    # old_agent_device = agent.device
    # to_device(env, Flux.cpu)
    # to_device(agent, Flux.cpu)

    # init_data!(
    #     env,
    #     data_size,
    #     in_dim,
    #     out_dim,
    #     h_dim,
    #     num_layers;
    #     rng = rng,
    #     device = env.device,
    # )

    # loss() =  -sum(agent.subagents[1](RLExperiments.get_next_obs_with_f(env, f) |> agent.device))
    # pars = Flux.params(f.f)
    # lossfun, gradfun, fg!, p0 = optfuns(loss, pars)
    # res = Optim.optimize(Optim.only_fg!(fg!), p0, LBFGS(), Optim.Options(iterations=10000, store_trace=true))
    # println(res)
    # to_device(env, old_env_device)
    # to_device(agent, old_agent_device)
    ###
    ### LBFGS
    ###

    for i = 1:n_steps

        if mod(i, report_freq) == 0

            #     if !isempty(env.internal_opt[2].state)
            # ss = env.internal_opt[2].state[env.init_f2.params[1]]
            # println([sum(s) for s in ss])
            #     end

            # V = agent.subagents[1](get_next_obs_with_f(env, env.f) |> agent.device)
            # G_norm = sum(LinearAlgebra.norm, gs)
            # env_copy = deepcopy(env)
            # ep = generate_episode(env_copy, agent, policy = 3, state = :dontreset)
            # println("Init Acc: ", ep[1].info[1][1])
            # println("Final Acc: ", ep[end].info[1][1])
            # p rintln("i: ", string(i), " | V : ", V, " | G_norm : ", G_norm)
            # println("ACC: ", calc_performance(env, f = f, mode = :train))
            # println("V: ", get_mean_v(env, agent, f, deterministic = true))
            # println("ACC_ALL: ", calc_performance(env, mode = :train, f = f, all_data = true))
            # println("V ALL: ", agent.subagents[1](get_next_obs_with_f(env, f, t = t, all_data = true) |> agent.device))
        end

        # ps = nothing
        # s_p = nothing
        # if env.state_representation == "parameters"
        #     s_p, re = Flux.destructure(f.f)
        #     ps = Flux.params(s_p)
        # else
        #     ps = f.params
        # end


        gs = get_sum_grad(env, agent, f, ps, t = t)

        Flux.Optimise.update!(opt, ps, gs)
        # if env.state_representation == "parameters"
        #     f.f = re(ps[1])
        #     f.params = ps
        # end
    end

    # p, re = Flux.destructure(f.f)
    # env.f = NeuralNetwork(re(p))
    # env.obs = get_next_obs(env)
    # v_post = get_mean_v(env, agent, f, deterministic = true)
    # println("V post: ", v_post)




    # G_norm_post = sum(LinearAlgebra.norm, gs)
    # println("G_norm post: ", G_norm_post)
    # println("ACC: ", calc_performance(env, mode = :train, f = f))
    # println("ACC_ALL: ", calc_performance(env, mode = :train, f = f, all_data = true))

    if return_gs
        return gs
    end
end

function get_next_obs(env::AbstractOptEnv)
    return get_next_obs_with_f(env, env.f) |> Flux.cpu
end

function get_local_improvement(env::AbstractOptEnv, M, f, x)
    # inds_local = StatsBase.sample(env.rng, env.ind_range, M, replace = false)
    # x_local = retrieve_with_inds(env, env.x, env.inds)# |> env.device
    # y_local = retrieve_with_inds(env, env.y, env.inds)# |> env.device

    # opt2 = deepcopy(env.opt)
    # f2 = deepcopy(f)

    # # println(length(keys(opt2.state)))

    # # new_ps = f2.params
    # # old_ps = f.params

    # # if typeof(opt2) <: Flux.ADAM
    # #     if !isempty(opt2.state)
    # #         opt2.state = IdDict(
    # #             new_p => (
    # #                 deepcopy(env.opt.state[old_p][1]),
    # #                 deepcopy(env.opt.state[old_p][2]),
    # #                 deepcopy(env.opt.state[old_p][3]),
    # #             ) for (new_p, old_p) in zip(new_ps, old_ps)
    # #                 )
    # #     end
    # # end

    # update_f(env, opt2, env.J, f2.f, x_local, y_local)
    # println(length(keys(opt2.state)))
    # return f2(x)

    # g_vec = vcat([reshape(gs[p], :) for p in env.f.params]...)
    p, re = Flux.destructure(f.f)
    if isempty(env.opt.state)
        Z = zeros(Float32, size(f(x)))
        return Z, Z
    else
        mts = x -> env.opt.state[env.f.params[x]][1]
        vts = x -> env.opt.state[env.f.params[x]][2]
    end
    # g_vec = vcat([reshape(gs[p], :) for p in env.f.params]...)
    m_vec = vcat([reshape(mts(i), :) for i = 1:length(env.f.params)]...) |> env.device
    v_vec = vcat([reshape(vts(i), :) for i = 1:length(env.f.params)]...) |> env.device
    f_m = re(m_vec)
    f_v = re(v_vec)
    return f_m(x), f_v(x)
end


function get_next_obs_with_f(
    env::AbstractOptEnv,
    f;
    M = nothing,
    task_id = nothing,
    xy = nothing,
    t = nothing,
)
    if isnothing(t)
        t = env.t
    end

    N = size(env.x)[end]

    state_rep_str = stop_gradient() do
        split(env.state_representation, "_")
    end

    if state_rep_str[1] == "parameters"
        # Can be differentiated
        P, re = Flux.destructure(f.f)
        aux = Float32.(log(env.opt.eta))

        return vcat(P, aux) |> env.device

    elseif state_rep_str[1] == "PVN"
        # Can be differentiated
        P, re = Flux.destructure(f.f)
        aux = Float32.(log(env.opt.eta))
        aux = []
        return P

    elseif state_rep_str[1] == "oblivious"
        acc = copy(log(-log(env.acc[1])))
        acc_test = copy(log(-log(env.acc[3])))
        aux = Float32.([log(env.opt.eta), t, acc, acc_test])
        return aux

    elseif contains(state_rep_str[1], "PD")
        if isnothing(task_id)
            N2 = Int(N / env.n_tasks)
            ind_range = env.ind_range
            task_id = Int(collect(ind_range)[end] / N2)
        else
            N2 = Int(N / env.n_tasks)
            ind_range = (1+N2*(task_id-1)):N2*task_id
        end

        if isnothing(M)
            if length(state_rep_str) == 1
                M = stop_gradient() do
                    minimum([256, env.data_size])
                end
            else
                M = stop_gradient() do
                    parse(Int, state_rep_str[2])
                end
            end
        elseif M == :all
            M = N
        elseif typeof(M) <: Int
            M = M
        else
            error()
        end

        inds = stop_gradient() do
            StatsBase.sample(env.rng, ind_range, M, replace = false)
        end
        # inds = collect(1:N)

        if isnothing(xy)
            x_ = retrieve_with_inds(env, env.x, inds)# |> env.device
            y = retrieve_with_inds(env, env.y, inds)# |> env.device
        else
            x_ = xy[1] |> agent.device
            y = xy[2] |> agent.device
        end

        y_ = f(x_)

        if env.dataset_name == "sinWave"
            scale = 10f0
            y_ = clamp.(y_, -scale, scale)/scale
        end

        # y_ = (y_ .- y).^2

        if env.reward_function == "FiniteHorizon"
            t = stop_gradient() do
                positional_encoding([t])
            end
            # aux = stop_gradient() do
            #     Float32.([log(env.opt.eta), t])
            # end

            aux = stop_gradient() do
                Float32.([log(env.opt.eta), t...])
            end
        else
            aux = Float32.([log(env.opt.eta)])
        end

        aux_dim = length(aux)

        meta_data_list = []

        if state_rep_str[1] == "PD-0-grad"
            for i = 1:1
                inds_local = StatsBase.sample(env.rng, 1:N, M, replace = false)
                x_local = retrieve_with_inds(env, env.x, inds_local)# |> env.device
                y_local = retrieve_with_inds(env, env.y, inds_local)# |> env.device

                opt2 = deepcopy(env.opt)
                f2 = deepcopy(f)

                new_ps = f2.params
                old_ps = f.params

                if typeof(opt2) <: Flux.ADAM
                    if !isempty(opt2.state)
                        opt2.state = IdDict(
                            new_p => (
                                deepcopy(env.opt.state[old_p][1]),
                                deepcopy(env.opt.state[old_p][2]),
                                deepcopy(env.opt.state[old_p][3]),
                            ) for (new_p, old_p) in zip(new_ps, old_ps)
                        )
                    end
                end

                update_f(env, opt2, env.J, f2.f, x_local, y_local)

                z_ = f2(x_)
                z_ = Z_ - f(x_)

                aux_stack = reshape(repeat(aux, M), (aux_dim, M))
                z_ = cat(z_, aux_stack, dims = 1)

                push!(meta_data_list, z_)
            end

        elseif state_rep_str[1] == "PD-x-grad"
            meta_data_list = [x_]
            for i = 1:1
                inds_local = StatsBase.sample(env.rng, 1:N, M, replace = false)
                x_local = retrieve_with_inds(env, env.x, inds_local)# |> env.device
                y_local = retrieve_with_inds(env, env.y, inds_local)# |> env.device

                opt2 = deepcopy(env.opt)
                f2 = deepcopy(f)

                new_ps = f2.params
                old_ps = f.params

                if typeof(opt2) <: Flux.ADAM
                    if !isempty(opt2.state)
                        opt2.state = IdDict(
                            new_p => (
                                deepcopy(env.opt.state[old_p][1]),
                                deepcopy(env.opt.state[old_p][2]),
                                deepcopy(env.opt.state[old_p][3]),
                            ) for (new_p, old_p) in zip(new_ps, old_ps)
                        )
                    end
                end

                update_f(env, opt2, env.J, f2.f, x_local, y_local)

                z_ = f2(x_)
                z_ = z_ - f(x_)

                aux_stack = reshape(repeat(aux, M), (aux_dim, M))
                z_ = cat(z_, aux_stack, dims = 1)

                push!(meta_data_list, z_)
            end

        elseif state_rep_str[1] == "PD-y-grad"
            if t == env.max_steps
                hist = zeros(size(y_))
            else
                hist = env.old_f(x_)
            end

            meta_data_list = [y, hist]
            # for i = 1:1
            #     z_1, z_2 = stop_gradient() do
            #         get_local_improvement(env, M, f, x_)
            #     end
            #     meta_data_list = [y, z_1, z_2]

            #     # z_ = stop_gradient() do
            #     #     get_local_improvement(env, M, f, x_)
            #     # end
            #     # meta_data_list = [y, z_]

            #     # aux_stack = reshape(repeat(aux, M), (aux_dim, M))
            #     # z_ = cat(z_, aux_stack, dims = 1)

            # end

        elseif state_rep_str[1] == "PD-xy"
            meta_data_list = [x_, y]

        elseif state_rep_str[1] == "PD-x"
            meta_data_list = [x_]

        elseif state_rep_str[1] == "PD-y"
            meta_data_list = [y]

        elseif state_rep_str[1] == "PD-0"
            # Only outputs needed, handled in outer code
        else
            error("Not valid: ", state_rep_str)
        end

        cat_arr = y_
        x_dim = size(y_)[1]

        for meta_data in meta_data_list
            cat_arr = cat(cat_arr, Float32.(meta_data), dims = 1)
            x_dim += size(meta_data)[1]
        end

        L = 1
        state = reshape(cat_arr, :)
        y_dim = 0
        return vcat(state, aux, aux_dim, x_dim, y_dim, L, M)

    else
        error("Not a valid state Representation for OptEnv!")
    end
end


function get_state(env::AbstractOptEnv)
    # Dummy variable to reduce memory usage
    return [1.0f0]
end

function get_obs(env::AbstractOptEnv)
    return env.obs
end

function get_terminal(env::AbstractOptEnv)
    return env.done
end

function get_reward(env::AbstractOptEnv)
    return env.reward
end

function get_actions(env::AbstractOptEnv)
    return env.action_space
end

function get_info(env::AbstractOptEnv)
    return [hcat(env.acc...)]
end

function random_action(env::AbstractOptEnv)
    a = rand(env.rng, get_actions(env))
    if typeof(a) <: Float64
        return Float32.(a)
    elseif typeof(a) <: Vector{Float64}
        return Float32.(a)
    else
        return a
    end
end

function default_action(env::AbstractOptEnv, state)
    return 3, 1.0f0
end

function estimate_startvalue_optenvgrad(agent::AbstractAgent, env::AbstractEnv; n_evals = 2)
    pre_vals = []
    post_vals = []
    grad_norms = []

    for _ = 1:n_evals
        reset!(env)
        s = get_obs(env)
        if typeof(agent.subagents[1].model) <: AbstractContinuousActionValue
            a = agent.subagents[3](s)
            pre_val = agent.subagents[1](s |> agent.device, a)[1]
        else
            q = agent.subagents[1](s |> agent.device)
            pre_val = maximum(q)
        end

        gs = optimize_student(agent, env, return_gs = true)

        s = get_obs(env)
        if typeof(agent.subagents[1].model) <: AbstractContinuousActionValue
            a = agent.subagents[3](s)
            post_val = agent.subagents[1](s |> agent.device, a)[1]
        else
            q = agent.subagents[1](s |> agent.device)
            post_val = maximum(q)
        end

        grad_norm = sum(LinearAlgebra.norm, gs)

        push!(pre_vals, pre_val)
        push!(post_vals, post_val)
        push!(grad_norms, grad_norm)
    end
    r = [mean(pre_vals), mean(post_vals), mean(grad_norms)]
    name = [
        "Estimate of pre-adapt start-state value",
        "Estimate of post-adapt start-state value",
        "Student Opt GradNorm",
    ]
    return r, name
end
