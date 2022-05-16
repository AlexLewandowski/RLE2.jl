using ArcadeLearningEnvironment

function get_obs(env::AtariEnv)
    return Float32.(ReinforcementLearningEnvironments.get_state(env))
end
