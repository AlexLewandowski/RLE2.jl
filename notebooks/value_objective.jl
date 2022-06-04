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
import RLE2

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


# ╔═╡ 460590e2-6cb6-433b-b3cd-42d60097aa9a
# agent = RLE2.get_agent(
# 	AgentType,
# 	buffers,
# 	env,
# 	measurement_freq,
# 	max_agent_steps,
# 	measurement_funcs,
# 	gamma,
# 	update_freq,
# 	update_cache,
# 	predict_window,
# 	history_window,
# 	num_layers,
# 	hidden_size,
# 	activation,
# 	drop_rate,
# 	optimizer,
# 	lr,
# 	device,
# 	num_grad_steps,
# 	force,
# 	seed,
# 	reg;
# 	behavior = behavior,
# 	kwargs_dict...,
# )


# ╔═╡ 4a2d41ec-37fc-4f54-9b86-d43fa92009d9


# ╔═╡ Cell order:
# ╠═d90972e6-e388-11ec-37c4-7b94db602b40
# ╠═eec595bf-a788-415a-9619-8e68a5d11a6a
# ╠═4f1c7d5e-dcad-4d4b-8533-76b577604201
# ╠═9acb7a99-a99f-490b-83fd-e9bbfd7950e9
# ╠═94014334-d251-47c9-b06e-3c91fd5429fe
# ╠═460590e2-6cb6-433b-b3cd-42d60097aa9a
# ╠═4a2d41ec-37fc-4f54-9b86-d43fa92009d9
