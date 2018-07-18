# common.jl

macro add_properties(T, props...)
    e = quote end
    for prop in props
        f = quote $(esc(prop))(a::$(esc(T))) = true end
        append!(e.args, f.args)
    end
    e
end

tolerance(::Type{T}) where {T} = sqrt(eps(T))
tolerance(::Type{Complex{T}}) where {T} = tolerance(T)


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
indexable_list(set, ::Type{T}) where {T} = typeof(set[1]) <: T

indexable_list(set::Array{S}, ::Type{T}) where {S,T} = S <: T
indexable_list(set::NTuple{N,S}, ::Type{T}) where {N,S,T}= S <: T


#An efficient way to access elements of a Tuple type using index j
@generated function tuple_index(::Type{T}, j) where {T <: Tuple}
    :($T.parameters[j])
end

@generated function tuple_length(::Type{T}) where {T <: Tuple}
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


default_threshold(y) = default_threshold(typeof(y))
default_threshold(::Type{T}) where {T <: AbstractFloat} = 100eps(T)
default_threshold(::Type{Complex{T}}) where {T <: AbstractFloat} = 100eps(T)
default_threshold(::AbstractArray{T}) where {T} = default_threshold(T)

# This is a candidate for a better implementation. How does one generate a
# unit vector in a tuple?
# ASK is this indeed a better implementation?
dimension_tuple(n, dim) = ntuple(k -> ((k==dim) ? 1 : 0), n)

# Generate a random value of type T
# Can be removed after 0.7
random_value(::Type{T}) where {T <: Number} = convert(T, rand())
random_value(::Type{Complex{T}}) where {T <: Real} = T(rand()) + im*T(rand())
random_value(::Type{T}) where {T} = rand() * one(T)
