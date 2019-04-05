
"""
A `DiscreteDictionary{I,T}` is a dictionary where `I` is a discrete type, e.g.,
the integers. Expansions in this discrete set typically constitute a vector space.

The elements of the set are the Euclidean unit vectors, with a finite index set.
"""
abstract type DiscreteDictionary{I,T} <: Dictionary{I,T}
end

coefficienttype(::Type{DiscreteDictionary{I,T}}) where {I,T} = T


# The point x is in the support of d exactly when it is within the bounds of
# the index set, so we can do a checkbounds with Bool argument (which does not
# throw an error but returns true or false).
dict_in_support(d::DiscreteDictionary{I}, idx, x::I) where I = checkbounds(Bool, d, x)

function dict_in_support(d::DiscreteDictionary{I}, idx, x) where I
    try
        # Attempt to convert x to a native index. If that fails, the result is false.
        idxn = native_index(d, x)
        checkbounds(Bool, d, idxn)
    catch e
        false
    end
end


# Evaluation of discrete sets works as follows:
# -> eval_element: does bounds check on idx
#   -> unsafe_eval_element1: in_support corresponds to bounds check on x
#     -> unsafe_eval_element: return zero(T), or one(T) if idx == x
unsafe_eval_element(set::DiscreteDictionary{I,T}, idx::I, x::I) where {I,T} =
    idx == x ? one(T) : zero(T)

# In the routine above, we made sure that idx and x have the same type, so that
# they can be compared. If they do not have the same type, we can try to
# promote the indices.
unsafe_eval_element(set::DiscreteDictionary, idx, x) =
    unsafe_eval_element(set, native_index(set, idx), native_index(set, x))


isdiscrete(dict::Dictionary) = false
isdiscrete(dict::DiscreteDictionary) = true

name(d::DiscreteDictionary) = "Discrete set"


#########################
# Concrete discrete sets
#########################

# We have to make these concrete sets because they store the size, and this
# depends on the type of the data. This is so in spite of the fact that type
# parameter I (the index type) already contains information.

"""
A `DiscreteVectorDictionary{T}` describes the linear space of vectors of finite length
`n` with element type `T`.
"""
struct DiscreteVectorDictionary{T} <: DiscreteDictionary{Int,T}
    n   ::  Int
end

# We set a default codomain type Float64
DiscreteVectorDictionary(n::Int) = DiscreteVectorDictionary{Float64}(n)

size(d::DiscreteVectorDictionary) = (d.n,)

similar(d::DiscreteVectorDictionary{T}, ::Type{Int}, n::Int) where {T} = DiscreteVectorDictionary{T}(n)

support(d::DiscreteVectorDictionary) = ClosedInterval{Int}(1, length(d))


"""
A `DiscreteArrayDictionary{N,T}` describes the linear space of arrays of finite size
`size(d)` with element type `T`.
"""
struct DiscreteArrayDictionary{T,N} <: DiscreteDictionary{ProductIndex{N},T}
    size    ::  NTuple{N,Int}
end

DiscreteArrayDictionary(size...) = DiscreteArrayDictionary{Float64}(size...)

DiscreteArrayDictionary{T}(size::Int...) where {T} = DicreteArrayDictionary{T}(size)

DiscreteArrayDictionary{T}(size::NTuple{N,Int}) where {T,N} =
    DiscreteArrayDictionary{T,N}(size)

DiscreteArrayDictionary(array::AbstractArray{T,N}) where {T,N} =
    DiscreteArrayDictionary{T,N}(size(array))

size(d::DiscreteArrayDictionary) = d.size

similar(d::DiscreteArrayDictionary{T}, ::Type{ProductIndex{N}}, dims::Vararg{Int,N}) where {T,N} = DiscreteArrayDictionary{T}(dims)

ordering(d::DiscreteArrayDictionary{N}) where {N} = ProductIndexList{N}(size(d))

native_index(d::DiscreteArrayDictionary, idx) = product_native_index(size(d), idx)


"""
A `DiscreteMultiArrayDictionary{A,T}` describes the linear space of multi-arrays
with element type `T`.
"""
struct DiscreteMultiArrayDictionary{T} <: DiscreteDictionary{MultilinearIndex,T}
    offsets ::  Vector{Int}
end

size(d::DiscreteMultiArrayDictionary) = (d.offsets[end],)

unsafe_offsets(d::DiscreteMultiArrayDictionary) = d.offsets

function similar(d::DiscreteMultiArrayDictionary, T::Type{MultilinearIndex}, size::Int...)
    @assert length(d) == prod(size)
    similar(d, T, d.offsets)
end

similar(d::DiscreteMultiArrayDictionary{T}, ::Type{MultilinearIndex}, offsets::Vector{Int}) where {T} = DiscreteMultiArrayDictionary{T}(offsets)

ordering(d::DiscreteMultiArrayDictionary) = MultilinearIndexList(unsafe_offsets(d))

support(d::DiscreteMultiArrayDictionary) = ClosedInterval{Int}(1, length(d))
