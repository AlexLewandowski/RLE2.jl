
abstract type AbstractMetaStateEncoder{A} <: AbstractModel{A} end

mutable struct MetaStateEncoder{A} <: AbstractMetaStateEncoder{A}
    f::A
    buffer
    output_func
end

function forward(model::MetaStateEncoder, s; kwargs...)

    if :output_func in keys(kwargs)
        output_func =  kwargs[:output_func]
    else
        output_func = nothing
    end
    s = get_metastate(model, s, output_func = output_func)
    return model.f(s)
end

function get_metastate(model::AbstractMetaStateEncoder, s; output_func = nothing)
    if isnothing(output_func)
        output_func = model.output_func
    end
    if length(size(s)) == 1
        StatsBase.sample(model.buffer)
        o = get_batch(model.buffer)[2]
    end


    M = size(s)[end]
    in_dim = size(s)[1]

    encode_dim = size(model.f.f[end][end].W)[1]

    dummy_metastate = zeros(Float32, (encode_dim, size(s)[2:end]...))
    dummy_input = cat(s, dummy_metastate, dims = 1)
    output = output_func(dummy_input)
    out_dim = size(output)[1]
    metastate = reshape(cat(dummy_input, output, dims = 1), :)

    meta_info  = [in_dim, in_dim, out_dim, 0, M]
    if length(size(s)) == 3
        o = cat([vcat(metastate, s[:, :, i], meta_info...) for i = 1:M]..., dims = 3)
    elseif length(size(s)) == 2
        o = cat([vcat(metastate, s[:, i], meta_info...) for i = 1:M]..., dims = 2)
    elseif length(size(s)) == 1
        o = vcat(metastate, s, meta_info...)
    end

    return o
end
