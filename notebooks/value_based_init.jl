### A Pluto.jl notebook ###
# v0.19.8

using Markdown
using InteractiveUtils

# ╔═╡ d90972e6-e388-11ec-37c4-7b94db602b40
begin
	using Pkg
	using Revise
	Pkg.activate("..")
end

# ╔═╡ 8de4578b-3f74-4de6-b440-08b1dcaf5c07
begin
	using Replotuce.Plots
	using StatsBase
end

# ╔═╡ eec595bf-a788-415a-9619-8e68a5d11a6a
begin
	import RLE2
	import Flux
	import Random
end

# ╔═╡ 4f1c7d5e-dcad-4d4b-8533-76b577604201
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
	force = :online

	reg = false
	
	num_episodes = 1000
	init_num_episodes = 10

	max_episode_length = 10
	max_num_episodes = 1000
	
	batch_size = 128
	gamma = 0.99f0
end

# ╔═╡ 460590e2-6cb6-433b-b3cd-42d60097aa9a
begin
	seed = 31
	maml_env, max_agent_steps, embedding_f = RLE2.get_env(
	    "OptEnv-FiniteHorizon10-sinWave-ADAM",
	    seed = seed,
	    max_steps = max_episode_length,
	    state_representation = "parameters",
	)
	PE_env, max_agent_steps, embedding_f = RLE2.get_env(
	    "OptEnv-FiniteHorizon10-sinWave-ADAM",
	    seed = seed,
	    max_steps = max_episode_length,
	    state_representation = "PE-y_20",
	)
	
	buffer_rng = Random.MersenneTwister(3 * seed - 1)
	
	train_buffer = RLE2.TransitionReplayBuffer(
	    maml_env,
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

	total_reports = 20
	measurement_funcs = [RLE2.mc_buffer_loss, RLE2.optimize_student_metrics]

    if num_episodes < total_reports
        total_reports = num_episodes
    end

    measurement_freq = floor(Int, num_episodes / total_reports)

	AgentType = "MCValue"
	
	maml_agent = RLE2.get_agent(
		AgentType,
		buffers,
		maml_env,
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
	
	PE_agent = RLE2.get_agent(
		AgentType,
		deepcopy(buffers),
		PE_env,
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
		pooling_func = :mean)
end

# ╔═╡ 4a2d41ec-37fc-4f54-9b86-d43fa92009d9
function train_agent(agent, env; num_iterations, callback_funcs, force)
	train_buffer = agent.buffers.train_buffer
	for buffer in agent.buffers
		RLE2.populate_replay_buffer!(
			buffer,
			env,
			agent,
			policy = :default,
			num_episodes = init_num_episodes,
			max_steps = agent.max_agent_steps,
			greedy = false,
		)
	end
	RLE2.reset!(env)
	RLE2.calculate_and_log_metrics(agent, env, agent.measurement_funcs, agent.measurement_dict, "")
	
	for i = 1:num_iterations
		RLE2.train_subagents(agent)
		if force !== :offline
			step = 1
			done = RLE2.get_terminal(env)
			while !done
				RLE2.train_subagents(agent)
				done = RLE2.interact!(env, agent, greedy = false, buffer = train_buffer)
				if step == agent.max_agent_steps || done
					RLE2.finish_episode(train_buffer)
					done = true
					break
				end
				step += 1
			end
			RLE2.reset!(env)
		end
		[callback_f(agent, env) for callback_f in callback_funcs]
		RLE2.calculate_and_log_metrics(
			agent,
			env,
			agent.measurement_funcs,
			agent.measurement_dict,
			"",
		)
	end
end

# ╔═╡ 242884a4-03a0-4c6e-8073-783ac15fcab1
train_agent(maml_agent,
			maml_env,
			num_iterations = num_episodes,
			callback_funcs = [RLE2.fomaml_student],
			force = force)

# ╔═╡ bc0588ea-fbfb-41ea-bc30-1952df59e393
train_agent(PE_agent,
			PE_env,
			num_iterations = num_episodes,
			callback_funcs = [RLE2.optimize_student],
    		force = force)

# ╔═╡ 23074fa6-7695-440f-9eaa-a56c5241c77b
begin
	yx_init = PE_agent.measurement_dict["init_test_performance"]
	y_init = [yx[1] for yx in yx_init]
	xy_init = [(yx[2], mean(y_init)) for yx in yx_init]
	
	yx_pe = PE_agent.measurement_dict["learned_init_test_performance"]
	yx_maml = maml_agent.measurement_dict["learned_init_test_performance"]
	
	xy_pe = [(yx[2], yx[1]) for yx in yx_pe]
	xy_maml = [(yx[2], yx[1]) for yx in yx_maml]
	
	plot(xy_pe, label = "PE", xlabel = "Num. Meta-Gradient Steps", ylabel = "L2 Error on Train, Avg'd Over 4 Tasks", title = "FOMAML vs. PE, Learned Init Perf., SinWave Regress")
	plot!(xy_maml, label = "FOMAML")
	plot!(xy_init, label = "Default Init")
end

# ╔═╡ f4a57268-91f6-469a-88ea-11f55db5bc6d
begin

	
	yx_pe_test = PE_agent.measurement_dict["adapted_test_performance"]
	yx_maml_test = maml_agent.measurement_dict["adapted_test_performance"]
	xy_pe_test = [(yx[2], yx[1]) for yx in yx_pe_test]
	xy_maml_test = [(yx[2], yx[1]) for yx in yx_maml_test]
	
	plot(xy_pe_test, label = "PE", xlabel = "Num. Meta-Gradient Steps", ylabel = "L2 Error on Train, Avg'd Over 4 Tasks", title = "FOMAML vs. PE, Post Task-Adaptation, SinWave Regress")
	
	plot!(xy_maml_test, label = "FOMAML")
	plot!(xy_init, label = "Default Init")
end

# ╔═╡ Cell order:
# ╠═d90972e6-e388-11ec-37c4-7b94db602b40
# ╠═eec595bf-a788-415a-9619-8e68a5d11a6a
# ╠═4f1c7d5e-dcad-4d4b-8533-76b577604201
# ╠═460590e2-6cb6-433b-b3cd-42d60097aa9a
# ╠═4a2d41ec-37fc-4f54-9b86-d43fa92009d9
# ╠═242884a4-03a0-4c6e-8073-783ac15fcab1
# ╠═bc0588ea-fbfb-41ea-bc30-1952df59e393
# ╠═8de4578b-3f74-4de6-b440-08b1dcaf5c07
# ╠═23074fa6-7695-440f-9eaa-a56c5241c77b
# ╠═f4a57268-91f6-469a-88ea-11f55db5bc6d
