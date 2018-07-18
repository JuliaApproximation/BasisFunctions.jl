# discrete_sets.jl

"""
A `DiscreteDictionary{I,T}` is a dictionary where `I` is a discrete type, e.g.,
the integers. Expansions in this discrete set typically constitute a vector space.

The elements of the set are the Euclidean unit vectors, with a finite index set.
"""
abstract type DiscreteDictionary{I,T} <: Dictionary{I,T}
end

coefficient_type(::Type{DiscreteDictionary{I,T}}) where {I,T} = T


# We can't update the domain type of a discrete set
dict_promote_domaintype(d::DiscreteDictionary, S) =
    error("The domain type of a discrete set is fixed.")

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


is_discrete(dict::Dictionary) = false
is_discrete(dict::DiscreteDictionary) = true

name(d::DiscreteDictionary) = "a discrete set"


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
immutable DiscreteVectorDictionary{T} <: DiscreteDictionary{Int,T}
    n   ::  Int
end

# We set a default codomain type Float64
DiscreteVectorDictionary(n::Int) = DiscreteVectorDictionary{Float64}(n)

length(d::DiscreteVectorDictionary) = d.n

resize(d::DiscreteVectorDictionary{T}, n) where {T} = DiscreteVectorDictionary{T}(n)

support(d::DiscreteVectorDictionary) = ClosedInterval{Int}(1, length(d))


"""
A `DiscreteArrayDictionary{N,T}` describes the linear space of arrays of finite size
`size(d)` with element type `T`.
"""
immutable DiscreteArrayDictionary{N,T} <: DiscreteDictionary{ProductIndex{N},T}
    size    ::  NTuple{N,Int}
end

DiscreteArrayDictionary(size::NTuple{N,Int}, ::Type{T} = Float64) where {N,T} =
    DiscreteArrayDictionary{N,T}(size)

DiscreteArrayDictionary(array::AbstractArray{T,N}) where {N,T} =
    DiscreteArrayDictionary{N,T}(size(array))

size(d::DiscreteArrayDictionary) = d.size
length(d::DiscreteArrayDictionary) = prod(size(d))

resize(d::DiscreteArrayDictionary{N,T}, size) where {N,T} = DiscreteArrayDictionary{N,T}(size)

ordering(d::DiscreteArrayDictionary{N}) where {N} = ProductIndexList{N}(size(d))

native_index(d::DiscreteArrayDictionary, idx) = product_native_index(size(d), idx)


"""
A `DiscreteMultiArrayDictionary{A,T}` describes the linear space of multi-arrays
with element type `T`.
"""
immutable DiscreteMultiArrayDictionary{T} <: DiscreteDictionary{MultilinearIndex,T}
    offsets ::  Vector{Int}
end

length(d::DiscreteMultiArrayDictionary) = d.offsets[end]

unsafe_offsets(d::DiscreteMultiArrayDictionary) = d.offsets

ordering(d::DiscreteMultiArrayDictionary) = MultilinearIndexList(unsafe_offsets(d))

support(d::DiscreteMultiArrayDictionary) = ClosedInterval{Int}(1, length(d))
