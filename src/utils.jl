function push_dict!(dict, key, val)
    if dict == nothing
        dict = Dict()
    end

    if !haskey(dict, key)
        dict[key] = [val]
    else
        push!(dict[key], val)
    end
    return dict
end

function add_to_dict(dict, key, val)
    if dict == nothing
        dict = Dict()
    end
    dict[key] = val
    return dict
end

function nested_eltype(x::AbstractArray)
    y = eltype(x)
    while y <: AbstractArray
        y = eltype(y)
    end
    return (y)
end

function string_to_list_funcs(list_string)
    online_cbs_string = split(list_string, ",")
    online_cbs = []
    for cb in online_cbs_string
        if !occursin("#", cb) && !isempty(cb[2:end])
            cb = lstrip(cb)
            cb = getfield(RLE2, Symbol(cb))
            push!(online_cbs, cb)
        end
    end
    return online_cbs
end

function load_buffer(load_path)
    buffer = FileIO.load(load_path)["train_buffer"]
    return buffer
end

function construct_params_vector(m)
    s = 1
    pars = get_params(m)
    M = length(get_params_vector(m))
    v = Zygote.Buffer(zeros(Float32, M), M)
    # v = []
    for g in pars
        l = length(g)
        # push!(v, vec(g)...)
        v[s:s+l-1] = vec(g)
        s += l
    end
    return copy(v)
end

function _dropgrad(f, dropgrad)
    if dropgrad
        return stop_gradient() do
            Zygote.dropgrad(f)
        end
    else
        return f
    end
end

function preprocess_cb(cbs)
    online_cbs_string = split(cbs, ",")
    online_cbs = []
    for cb in online_cbs_string
        if !occursin("#", cb) && !isempty(cb[2:end])
            cb = lstrip(cb)
            cb = getfield(RLE2, Symbol(cb))
            push!(online_cbs, cb)
        end
    end
    return online_cbs
end
