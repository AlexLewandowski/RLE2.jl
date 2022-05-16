abstract type AbstractEnvValue{A} <: AbstractModel{A} end

mutable struct EnvValue{E<:AbstractEnv} <: AbstractEnvValue{E}
    env::E
    Q
    state_encoder
    device
    rng
end

Base.show(io::IO, m::AbstractEnvValue) = begin
    println(io, typeof(m).name)
    println(io, "  Env Name: ", typeof(m.env).name)
    println(io, "  Q Name: ", typeof(m.Q).name)
end

function get_params(M::AbstractEnvValue)
    get_params(M.Q)
end

function (M::AbstractEnvValue)(s, a)
    gamma = 0.99f0
    max_steps = 4 #TODO ???
    ep = generate_episode(M.env, M; policy = M.Q, state = s, action = a, max_steps = max_steps, greedy = true)
    rs = reshape([exp.r for exp in ep], (1, :, 1))
    return nstep_returns(gamma, rs)[1]
end

function forward(M::AbstractEnvValue, s::Array{Float32, 1})
    gamma = 0.99f0
    max_steps = 200 #TODO ???
    ep = generate_episode(M.env, M; policy = M.Q, state = s, max_steps = max_steps, greedy = true)
    rs = reshape([exp.r for exp in ep], (1, :, 1))
    return nstep_returns(gamma, rs)[1]
end

function forward(M::AbstractEnvValue, s::Array{Float32, 2})
    B = size(s)[end]

    s_0 = s[:, 1]
    Gs = M(s_0)
    for i = 2:B
        s_0 = s[:, i]
        G = M(s_0)
        Gs = vcat(Gs, G)
    end
    return reshape(Gs, :)
end
