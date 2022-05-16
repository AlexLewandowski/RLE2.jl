import ReinforcementLearningEnvironments: CartPoleEnv, MountainCarEnv, PendulumEnv, AtariEnv
# import LyceumMuJoCo: HopperV2, PointMass, CartpoleSwingup, SwimmerV2
using StatsBase: mean, std

using Flux
import Random
import Random: MersenneTwister
import StatsBase

using RLE2

using Flux.Zygote: @nograd

stop_gradient(f) = f()
@nograd stop_gradient

history_window = 1
predict_window = 0

max_episode_length = 10
max_num_episodes = 10
num_episodes = 2
batch_size = 2
gamma = 0.99f0
skip = 1
gamma = 0.99f0

n_actions = 6 #TODO
corruption_rate = 0.0f0


# seed = 9569612102212
# seed = 10101827296962
seed = 6951693
Random.seed!(2 * seed + 1)
state_representation = "PEN_10"
pooling_func = :mean
recurrent_action = 0
buffer_rng = MersenneTwister(3*seed-1)
overlap = false

env, max_agent_steps, embedding_f = RLE2.get_env("CartPole", skip = skip, seed = seed, max_steps = max_episode_length)

ob = RLE2.get_obs(env)
s  = RLE2.get_state(env)

a = RLE2.random_action(env)

ns = length(s)
na = RLE2.num_actions(env)

device = Flux.cpu
println(device)

hidden_size = 128
num_layers = 1
lr = 0.001
activation = relu
drop_rate = 0.0f0

optimizer = ADAM

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

mc_buffer = TransitionReplayBuffer(
    env,
    max_num_episodes,
    max_agent_steps,
    batch_size,
    gamma,
    history_window = history_window,
    predict_window = predict_window,
    overlap = overlap,
    rng = buffer_rng,
    name = "mc_buffer",
)

test_buffer = TransitionReplayBuffer(
    env,
    max_num_episodes,
    max_agent_steps,
    batch_size,
    gamma,
    history_window = history_window,
    predict_window = predict_window,
    overlap = overlap,
    rng = buffer_rng,
    name = "test_buffer",
)

A = feed_forward(ns, na, hidden_size, num_hidden_layers = num_layers)

buffers = (train_buffer = train_buffer, test_buffer = test_buffer)

force = nothing
metric_freq = 1
list_of_cbs = [rollout_returns, buffer_loss]
update_freq = 2
update_cache = 1
num_grad_steps = 1
reg = 0.1

agents = []

shorter_get_agent = x -> RLE2.get_agent(x,
                                              buffers,
                                              env,
                                              metric_freq,
                                              max_agent_steps,
                                              list_of_cbs,
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
                                              reg,
                                              pooling_func = pooling_func
                                              )

agent = shorter_get_agent("DQN")
push!(agents, agent)

populate_replay_buffer!(train_buffer, env, agent, policy = :random, num_episodes = num_episodes, max_steps = max_agent_steps)
populate_replay_buffer!(test_buffer, env, agent, num_episodes = num_episodes, max_steps = max_agent_steps)

StatsBase.sample(train_buffer)
_, ob, a, p,     r, _, obp, done = RLE2.get_batch(train_buffer)
data = RLE2.get_batch(train_buffer, gamma, device, x -> x)

bc_agent = shorter_get_agent("BehaviorCloning")
push!(agents, bc_agent)

qr_agent = shorter_get_agent("QRDQN10")
push!(agents, qr_agent)

ddqn_agent = shorter_get_agent(
    "DoubleDQN",
)
push!(agents, ddqn_agent)

restd0_agent = shorter_get_agent(
    "ResidualTD0",
)
push!(agents, restd0_agent)

td0_agent = shorter_get_agent(
    "TD0",
)
push!(agents, td0_agent)

mcv_agent = shorter_get_agent(
    "MCValue",
)
push!(agents, mcv_agent)

td3_agent = shorter_get_agent(
    "TD3",
)
push!(agents, td3_agent)

nothing
