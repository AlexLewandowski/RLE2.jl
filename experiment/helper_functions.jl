
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
