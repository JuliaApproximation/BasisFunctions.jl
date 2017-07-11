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
const True = Val{true}
const False = Val{false}

##################################################
# Copy data between generalized coefficient sets
##################################################

# We try to use the native index for each coefficient vector.
# - If they are both vectors, we can use a linear index
function copy!(dest::AbstractVector, src::AbstractVector)
    @assert length(dest) == length(src)
    for i in eachindex(dest)
        dest[i] = src[i]
    end
    dest
end

# - If only the destination is a vector, use the native index of src
function copy!(dest::AbstractVector, src)
    for (i,j) in enumerate(eachindex(src))
        dest[i] = src[j]
    end
    dest
end

# - If only the src is a vector, use the native index of dest
function copy!(dest, src::AbstractVector)
    for (i,j) in enumerate(eachindex(dest))
        dest[j] = src[i]
    end
    dest
end

# - Generic fallback: rely on eachindex to return a suitable index for both
function copy!(dest, src)
    for i in eachindex(dest, src)
        dest[i] = src[i]
    end
    dest
end


"Return true if the set is indexable and has elements whose type is a subtype of T."
indexable_set{T}(set, ::Type{T}) = typeof(set[1]) <: T

indexable_set{S,T}(set::Array{S}, ::Type{T}) = S <: T
indexable_set{N,S,T}(set::NTuple{N,S}, ::Type{T}) = S <: T


#An efficient way to access elements of a Tuple type using index j
@generated function tuple_index{T <: Tuple}(::Type{T}, j)
    :($T.parameters[j])
end

@generated function tuple_length{T <: Tuple}(::Type{T})
    :($length(T.parameters))
end

function insert_at(a::Vector, idx, b)
    if idx > 1
        [a[1:idx-1]..., b..., a[idx:end]...]
    else
        [b..., a...]
    end
end

isdyadic(n::Int) = n == 1<<round(Int, log2(n))
