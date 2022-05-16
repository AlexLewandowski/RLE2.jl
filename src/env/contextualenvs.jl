import MLDatasets

include("contextualbandit.jl")
include("contextualmdp.jl")

ContextualEnv = Union{ContextualMDP,ContextualBandit}

function optimal_action(env::ContextualEnv, s)
    return env.state
end
