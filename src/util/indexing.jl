# indexing.jl

############
# Overview #
############

# We collect some methods and definitions having to do with various kinds of
# indexing. We make some assumptions on all index sets:
# - an index set is mathematically a set, i.e., it has no duplicates
# - in addition to being a set, the elements can be ordered and the ordering
#   is defined by the iterator of the set.
#
# The functionality in this file includes:
# - bounds checking: see `checkbounds` function
# - conversion between indices of different types
# - efficient iterators

# Dictionaries can be indexed in various ways. We assume that the semantics
# of the index is determined by its type and, moreover, that linear indices
# are always Int's. This means that no other index can have type Int.
const LinearIndex = Int

# We fall back to whatever membership function (`in`) is defined for the index
# set `I`.
checkbounds(i::LinearIndex, I) = i ∈ I


##################
# Native indices
##################


"""
A native index has to be distinguishable from a linear index by its type, but
a linear index is always an integer. If a native index is also an integer type,
then its value should be wrapped in a different type.

That is the purpose of `NativeIndex`.

A `NativeIndex` inherits from `Integer` and implements basic functionality such
that one can do computations with native indices or construct ranges.
"""
struct NativeIndex{S} <: Integer
    value   ::  LinearIndex
end

NativeIndex{S}(a::NativeIndex{S}) where {S} = a

convert(::Type{NativeIndex{S}}, value::LinearIndex) where {S} = NativeIndex{S}(value)

value(idx::NativeIndex) = idx.value

# With this line we inherit binary operations involving integers and native indices
Base.promote_rule(::Type{NativeIndex{S}}, ::Type{LinearIndex}) where {S} = NativeIndex{S}

for op in (:+, :-, :*)
    @eval $op(a::NativeIndex{S}, b::NativeIndex{S}) where {S} = typeof(a)($op(value(a),value(b)))
end

for op in (:<, :<=, :>, :>=)
    @eval $op(a::NativeIndex{S}, b::NativeIndex{S}) where {S} = $op(value(a),value(b))
end

(-)(a::NativeIndex) = typeof(a)(-value(a))



"""
The `type IndexList` implements a map from linear indices to another family
of indices, and vice-versa.

It is implemented as an abstract vector, and hence the functionality of vectors
is available.

Concrete subtypes should implement:
- `getindex(l::MyIndexList, idx::Int)` -> this is the map from linear indices
  to native indices
- `getindex(l::MyIndexList, idxn::MyIndex)` -> this is the inverse map
- `size(l::MyIndexList)`
"""
abstract type IndexList{T} <: AbstractVector{T}
end

# The concrete subtype should implement size (and perhaps length)

# Assume linear indexing, override if this is not appropriate
Base.IndexStyle(list::IndexList) = Base.IndexLinear()


"""
If a dictionary does not have a native index, we use the `DefaultNativeIndex`
type to distinguish between linear and native indices. However, the default
native index simply wraps the integer linear index.
"""
const DefaultNativeIndex = NativeIndex{:default}

"""
The default index list is an identity map from integers to themselves, wrapped
in a `DefaultNativeIndex`.
"""
struct DefaultIndexList <: IndexList{DefaultNativeIndex}
    n   ::  Int
end

size(list::DefaultIndexList) = (list.n,)

getindex(list::DefaultIndexList, idx::LinearIndex) = DefaultNativeIndex(idx)
getindex(list::DefaultIndexList, idxn::DefaultNativeIndex) = value(idxn)

"""
The `ShiftedIndex` is the linear index shifted by `1`. Thus, its value starts at
`0` and ranges up to `n-1`.
"""
const ShiftedIndex = NativeIndex{:shift}

struct ShiftedIndexList <: IndexList{ShiftedIndex}
    n       ::  Int
    shift   ::  Int
end

# Default value of the shift is 1
ShiftedIndexList(n::Int) = ShiftedIndexList(n, 1)

size(list::ShiftedIndexList) = (list.n,)
shift(list::ShiftedIndexList) = list.shift

getindex(list::ShiftedIndexList, idx::LinearIndex) = ShiftedIndex(idx-shift(list))
getindex(list::ShiftedIndexList, idxn::ShiftedIndex) = value(idxn)+shift(list)


########################
# Product indices
########################

# We reuse the CartesianIndex defined in Base
const ProductIndex{N} = CartesianIndex{N}

# Convenience functions
index(idxn::ProductIndex) = value(idxn)
# Return the index as a tuple
indextuple(idxn::ProductIndex) = value(idxn).I

"""
`ProductIndexList` is a list of product indices, suitable for use as the ordering
of a dictionary with tensor product structure.
"""
struct ProductIndexList{N} <: IndexList{ProductIndex{N}}
    size    ::  NTuple{N,Int}
end

size(list::ProductIndexList) = list.size

# We use Cartesian indexing, because that is more efficient than linear indexing
# when iterating over product sets: we have to convert int's to cartesian indices
# anyway
Base.IndexStyle(list::ProductIndexList) = Base.IndexCartesian()

# We have to override eachindex, because the default eachindex in Base returns
# a linear index for a vector (because any IndexList is an AbstractVector), and
# the most efficient iteration over product dictionaries is using cartesian indices.
eachindex(list::ProductIndexList) = CartesianRange(indices(list))

# We convert between integers and product indices using ind2sub and sub2ind
getindex(list::ProductIndexList, idx::LinearIndex) =
    ProductIndex(ind2sub(size(list), idx))
getindex(list::ProductIndexList, idxn::ProductIndex) =
    sub2ind(size(list), indextuple(idxn)...)

# Convert a tuple of int's or a list of int's to a linear index
getindex(list::ProductIndexList{N}, idx::NTuple{N,Int}) where {N} =
    getindex(list, CartesianIndex(idx))
getindex(list::ProductIndexList{N}, idx1::Int, idx2::Int, idx::Int...) where {N} =
    getindex(list, CartesianIndex{N}(idx1, idx2, idx...))


########################
# Composite indices
########################


const MultilinearIndex = NTuple{2,Int}

"""
`MultiLinearIndexList` is a list of indices suitable for use as the ordering
of a dictionary with composite structure.
"""
struct MultilinearIndexList <: IndexList{MultilinearIndex}
    offsets ::  Vector{Int}
end

const MLIndexList = MultilinearIndexList

length(list::MLIndexList) = list.offsets[end]

function getindex(list::MLIndexList, idx::LinearIndex)
    i = 0
    while idx > list.offsets[i+1]
        i += 1
    end
    (i,idx-list.offsets[i])
end

getindex(list::MLIndexList, idxn::MultilinearIndex) = list.offsets[idxn[1]] + idxn[2]


# Comment out but keep for possible later use: is the indexing below faster than
# simply using lienar indices?
#
# """
# A `MLIndexIterator` iterates over a sequence of linear indices. Each index
# is a tuple, where the first entry refers to the linear index, and the second
# entry is the linear index.
#
# This resembles the `Flatten` iterator in `Base`, but the latter would yield only the
# second entry. For example, the composite indices of `(1:5,1:3)` are:
# ```
# (1,1), (1,2), (1,3), (1,4), (1,5), (2,1), (2,2), (2,3).
# ```
# In contrast, the `Flatten` iterator in Base would yield:
# ```
# 1, 2, 3, 4, 5, 1, 2, 3
# ```
# """
# struct MLIndexIterator{L}
#     lengths ::  L
# end
#
# start(it::MLIndexIterator) = (1,1)
#
# function next(it::MLIndexIterator, state)
#     i = state[1]
#     j = state[2]
#     if j == it.lengths[i]
#         nextstate = (i+1,1)
#     else
#         nextstate = (i,j+1)
#     end
#     (state, nextstate)
# end
#
# done(it::MLIndexIterator, state) = state[1] > length(it.lengths)
#
# length(it::MLIndexIterator) = sum(it.lengths)
