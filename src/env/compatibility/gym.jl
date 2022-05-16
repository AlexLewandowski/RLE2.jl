function (env::GymEnv)(a::Int64)
    float_a = (4 / (num_actions(env) - 1)) * (a - (num_actions(env) - 1) / 2 - 1)
    env([float_a])
end

function num_actions(env::GymEnv)
    #TODO
end

function random_action(env::GymEnv)
    # TODO
    N = num_actions(env)
    rand(1:N)
end

function action_type(env::GymEnv)
    Int64
end
