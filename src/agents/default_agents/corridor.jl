function get_agent(env::AbstractMDP)
    tabular = is_tabular(env)
    measurement_freq = 10000
    max_agent_steps = 10 #IMPORTANT
    measurement_funcs = []
    gamma = 0.99f0
    update_freq = tabular ? 0 : 4
    update_cache = 0
    predict_window = 0 #TODO should be both 0 or 1
    history_window = 1 #TODO should be both 0 or 1
    num_layers = 2
    hidden_size = 32
    activation = Flux.relu
    drop_rate = 0.0f0
    optimizer = Flux.ADAM
    lr = tabular ? 0.5 : 0.001
    device = Flux.cpu
    num_grad_steps = 4
    force = :offline

    max_num_episodes = 100
    batch_size = 32
    overlap = true
    seed = 1
    buffer_rng = MersenneTwister(3*seed-1)

    train_buffer = TransitionReplayBuffer(
        env,
        max_num_episodes,
        max_agent_steps,
        batch_size,
        gamma,
        history_window = history_window,
        predict_window = predict_window,
        overlap = overlap,
        rng = buffer_rng,
        name = "train_buffer",
    )
    buffers = (train_buffer = train_buffer, )
    agent = get_agent("DQN",
                      buffers,
                      env,
                      measurement_freq,
                      max_agent_steps,
                      measurement_funcs,
                      gamma,
                      update_freq,
                      update_cache,
                      predict_window,
                      history_window,
                      num_layers,
                      hidden_size,
                      activation,
                      drop_rate,
                      optimizer,
                      lr,
                      device,
                      num_grad_steps,
                      force,
                      seed,
                      )
end
