### A Pluto.jl notebook ###
# v0.19.8

using Markdown
using InteractiveUtils

# ╔═╡ d90972e6-e388-11ec-37c4-7b94db602b40
begin
	using Pkg
	Pkg.activate("..")
end

# ╔═╡ eec595bf-a788-415a-9619-8e68a5d11a6a
begin
	import RLE2
	import Flux
	import Random
end

# ╔═╡ 4f1c7d5e-dcad-4d4b-8533-76b577604201
begin
	seed = 31
	max_episode_length = 10
	state_representation = "parameters"
	env, max_agent_steps, embedding_f = RLE2.get_env(
	    "OptEnv-FiniteHorizon10-sinWave-ADAM",
	    seed = seed,
	    max_steps = max_episode_length,
	    state_representation = state_representation,
	)
end

# ╔═╡ 9acb7a99-a99f-490b-83fd-e9bbfd7950e9
ep = RLE2.generate_episode(env, nothing, policy = :default, max_steps = max_episode_length);

# ╔═╡ 94014334-d251-47c9-b06e-3c91fd5429fe
begin
	max_num_episodes = 100
	batch_size = 64
	gamma = 0.99f0
	
	buffer_rng = Random.MersenneTwister(3 * seed - 1)
	
	train_buffer = RLE2.TransitionReplayBuffer(
	    env,
	    max_num_episodes,
	    max_agent_steps,
	    batch_size,
	    gamma,
	    rng = buffer_rng,
	    name = "train_buffer",
	)

	test_buffer = deepcopy(train_buffer)
    test_buffer.name = "test_buffer"

	buffers = (train_buffer = train_buffer, test_buffer = test_buffer)

end

# ╔═╡ 88ec5fd7-8c7a-4eb4-a15b-f3305b81b7e1
begin
	device = Flux.cpu
	println(device)

	hidden_size = 128
	num_layers = 1
	activation = Flux.relu
	drop_rate = 0.0f0
	optimizer = Flux.ADAM
	lr = 0.001

	update_freq = 100

	predict_window = 0
	history_window = 1

	num_grad_steps = 1
	force = :offline

	reg = false
end

# ╔═╡ 460590e2-6cb6-433b-b3cd-42d60097aa9a
begin
	num_episodes = 100
	total_reports = 200
	measurement_funcs = [RLE2.optimize_student_metrics]

    if num_episodes < total_reports
        total_reports = num_episodes
    end

    measurement_freq = floor(Int, num_episodes / total_reports)

	AgentType = "MCValue"
	agent = RLE2.get_agent(
		AgentType,
		buffers,
		env,
		measurement_freq,
		max_agent_steps,
		measurement_funcs,
		gamma,
		update_freq,
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
		reg;
		behavior = :default,
	)
end

# ╔═╡ 4a2d41ec-37fc-4f54-9b86-d43fa92009d9


# ╔═╡ Cell order:
# ╠═d90972e6-e388-11ec-37c4-7b94db602b40
# ╠═eec595bf-a788-415a-9619-8e68a5d11a6a
# ╠═4f1c7d5e-dcad-4d4b-8533-76b577604201
# ╠═9acb7a99-a99f-490b-83fd-e9bbfd7950e9
# ╠═94014334-d251-47c9-b06e-3c91fd5429fe
# ╠═88ec5fd7-8c7a-4eb4-a15b-f3305b81b7e1
# ╠═460590e2-6cb6-433b-b3cd-42d60097aa9a
# ╠═4a2d41ec-37fc-4f54-9b86-d43fa92009d9
