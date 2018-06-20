# discrete_sets.jl

"""
A `DiscreteSet{I,T}` is a dictionary where `I` is a discrete type, e.g.,
the integers. Expansions in this discrete set typically constitute a vector space.

The elements of the set are the Euclidean unit vectors, with a finite index set.
"""
abstract type DiscreteSet{I,T} <: Dictionary{I,T}
end

coefficient_type(::Type{DiscreteSet{I,T}}) where {I,T} = T

# We can't update the domain type of a discrete set
dict_promote_domaintype(d::DiscreteSet, S) =
    error("The domain type of a discrete set is fixed.")

# The point x is in the support of d exactly when it is within the bounds of
# the index set, so we can do a checkbounds with Bool argument (which does not
# throw an error but returns true or false).
in_support(d::DiscreteSet{I}, idx, x::I) where I = checkbounds(Bool, d, x)

function in_support(d::DiscreteSet, idx, x)
    try
        in_support(d, idx, native_index(d, x))
    catch e
        false
    end
end

# Evaluation of discrete sets works as follows:
# -> eval_element: does bounds check on idx
#   -> unsafe_eval_element1: in_support corresponds to bounds check on x
#     -> unsafe_eval_element: return zero(T), or one(T) if idx == x
unsafe_eval_element(set::DiscreteSet{I,T}, idx::I, x::I) where {I,T} =
    idx == x ? one(T) : zero(T)

# In the routine above, we made sure that idx and x have the same type, so that
# they can be compared. If they do not have the same type, we can try to
# promote the indices.
unsafe_eval_element(set::DiscreteSet, idx, x) =
    unsafe_eval_element(set, native_index(set, idx), native_index(set, x))


is_discrete(dict::Dictionary) = false
is_discrete(dict::DiscreteSet) = true

name(d::DiscreteSet) = "a discrete set"


#########################
# Concrete discrete sets
#########################

# We have to make these concrete sets because they store the size, and this
# depends on the type of the data. This is so in spite of the fact that type
# parameter I (the index type) already contains information.

"""
A `DiscreteVectorSet{T}` describes the linear space of vectors of finite length
`n` with element type `T`.
"""
immutable DiscreteVectorSet{T} <: DiscreteSet{Int,T}
    n   ::  Int
end

# We set a default codomain type Float64
DiscreteVectorSet(n::Int) = DiscreteVectorSet{Float64}(n)

length(d::DiscreteVectorSet) = d.n

resize(d::DiscreteVectorSet{T}, n) where {T} = DiscreteVectorSet{T}(n)

support(d::DiscreteVectorSet) = ClosedInterval{Int}(1, length(d))


"""
A `DiscreteArraySet{N,T}` describes the linear space of arrays of finite size
`size(d)` with element type `T`.
"""
immutable DiscreteArraySet{N,T} <: DiscreteSet{ProductIndex{N},T}
    size    ::  NTuple{N,Int}
end

DiscreteArraySet(size::NTuple{N,Int}, ::Type{T} = Float64) where {N,T} =
    DiscreteArraySet{N,T}(size)

DiscreteArraySet(array::AbstractArray{T,N}) where {N,T} =
    DiscreteArraySet{N,T}(size(array))

size(d::DiscreteArraySet) = d.size
length(d::DiscreteArraySet) = prod(size(d))

resize(d::DiscreteArraySet{N,T}, size) where {N,T} = DiscreteArraySet{N,T}(size)

ordering(d::DiscreteArraySet{N}) where {N} = ProductIndexList{N}(size(d))

native_index(d::DiscreteArraySet, idx) = product_native_index(size(d), idx)


"""
A `DiscreteMultiArraySet{A,T}` describes the linear space of multi-arrays
with element type `T`.
"""
immutable DiscreteMultiArraySet{T} <: DiscreteSet{MultilinearIndex,T}
    offsets ::  Vector{Int}
end

length(d::DiscreteMultiArraySet) = d.offsets[end]

unsafe_offsets(d::DiscreteMultiArraySet) = d.offsets

ordering(d::DiscreteMultiArraySet) = MultilinearIndexList(unsafe_offsets(d))

support(d::DiscreteMultiArraySet) = ClosedInterval{Int}(1, length(d))
