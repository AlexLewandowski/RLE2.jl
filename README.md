# RLE2.jl 

## Installation

``` sh
git clone git@github.com:AlexLewandowski/RLE2.jl.git ~/.julia/dev/RLE2
cd ~/.julia/dev/RLE2/
git submodule update --init --recursive

julia

using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

## Design Philosophy

RLE2.jl provides many primitives that can be flexibly combined to
specify RL / ML algorithms. At the lowest level is the approximator, which
specifies the function approximation architecture, input-output control flow and
some basic API to the parameters. The next level of abstraction is the model,
which uses an approximator to model something, such as value, policy,
reward/transition model. This level comprises many of the individual parts of RL
algorithms, such as DQN, QR-DQN, but cannot be used standalone. To optimize a
model, one must instantiate a Subagent struct. The optimization functions all
work on the level of the Subagent, and the Subagent provides the optimizer and
"linkage" function to connect the model being optimzied with potentially other
models (such as target networks). Finally, the collection of Subagents is stored
in the highest level of abstraction, the Agent, which glues the different
Subagents together and provides the API to the environment, such as the replay
buffer and the behaviour policy.

## Running experiments

You can test your configuration quickly with run_test, which will look at a
subset of parameters under a light workload:
``` sh
cd ~/.julia/dev/RLE2/
julia test/run_test.jl --config_file "configs/cartpole.toml"
```

This can also be done in an interactive session, as follows

``` julia
include("test/setup.jl")
ag, en = run_test("configs/cartpole.toml", return_early = false, test_hyperparams = false);
```

Once ready, the experiment can be run in parallel, in this case with 2 workers
and no plotting:
``` sh
cd ~/.julia/dev/RLE2/
julia experiment/run_parallel.jl --config_file "configs/cartpole.toml" --plots false --num_workers 2
```
