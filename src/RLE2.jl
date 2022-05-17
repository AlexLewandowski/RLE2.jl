module RLE2

using CUDA

import Random: AbstractRNG, GLOBAL_RNG, MersenneTwister
import StatsBase
import LinearAlgebra
import Flux
import Parameters
import FileIO

import Flux.Zygote
stop_gradient(f) = f()
Zygote.@nograd stop_gradient

import ReinforcementLearningEnvironments: AbstractEnv

abstract type AbstractAgent end
abstract type AbstractSubagent end

include("distributions.jl")

export feed_forward, lstm_model, rnn_model
include("approximator/approximator.jl")

export RNNActionValue, Policy, PersistActionValue
include("model/model.jl")

# export LQREnv
export ContextualBandit, ContextualMDP
include("env/env.jl")

export populate_replay_buffer!, reset_experience!
export TransitionReplayBuffer
include("buffer/replay_buffer.jl")

export Agent
include("agents/agents.jl")

export imputekl, squared, absolute, policygradient
export train_subagent
include("objective/training.jl")

export push_dict!
include("utils.jl")


export calculate_and_log_metrics, calculate_metrics
export counterfactual, rollout_returns, stable_rank, mean_weights
export estimate_startvalue, estimate_value, action_gap
export buffer_loss, mc_buffer_loss
include("metrics/metrics.jl")

end
