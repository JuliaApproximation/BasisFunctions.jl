# common.jl

macro add_properties(T, props...)
    e = quote end
    for prop in props
        f = quote $(esc(prop))(a::$(esc(T))) = true end
        append!(e.args, f.args)
    end
    e
end

tolerance{T}(::Type{T}) = sqrt(eps(T))
tolerance{T <: Real}(::Type{Complex{T}}) = tolerance(T)


# Convenience definitions for the implementation of traits
typealias True Val{true}
typealias False Val{false}


function copy!(dest, src)
    for i in eachindex(dest, src)
        dest[i] = src[i]
    end
    dest
end


#An efficient way to access elements of a Tuple type using index j
@generated function tuple_index{T <: Tuple}(::Type{T}, j)
    :($T.parameters[j])
end

@generated function tuple_length{T <: Tuple}(::Type{T})
    :($length(T.parameters))
end
