
function preprocess_list_of_funcs(cbs)
    list_of_funcs_string = split(cbs, ",")
    list_of_funcs = []
    for cb in list_of_funcs_string
        if !occursin("#", cb) && !isempty(cb[2:end])
            cb = lstrip(cb)
            println(cb)
            cb = getfield(RLE2, Symbol(cb))
            push!(list_of_funcs, cb)
        end
    end
    return list_of_funcs
end
