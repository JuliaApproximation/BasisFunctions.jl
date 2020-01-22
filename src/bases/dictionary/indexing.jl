
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



"""
The `type IndexList` implements a map from linear indices to another family
of indices, and vice-versa.

It is implemented as an abstract vector. Hence, the functionality of vectors
is available. This also means that the dimension of an index list is always a
vector, whose length equals that of the dictionary it corresponds to. Still,
in several cases one may index the list with other kinds of indices.

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



##################
# Native indices
##################

# Dictionaries can be indexed in various ways. We assume that the semantics
# of the index is determined by its type and, moreover, that linear indices
# are always Int's. This means that no other index can have type Int.
const LinearIndex = Int

"""
An `AbstractIntegerIndex` represents an integer that is being used as an index
in the BasisFunctions package.

The type implements basic functionality of integers such that one can do
computations with the indices or construct ranges.
"""
abstract type AbstractIntegerIndex <: Integer end

convert(I::Type{<:AbstractIntegerIndex}, value::Int) = I(value)
convert(::Type{T}, idx::AbstractIntegerIndex) where {T <: Number} = convert(T, value(idx))
# Resolve an ambiguity with mpfr code...
convert(::Type{BigFloat}, idx::AbstractIntegerIndex) = convert(BigFloat, value(idx))

# With this line we inherit binary operations involving integers and native indices
Base.promote_rule(I::Type{<:AbstractIntegerIndex}, ::Type{Int}) = I
# For floating points, we choose to convert to the numeric value
Base.promote_rule(I::Type{<:AbstractIntegerIndex}, F::Type{<:AbstractFloat}) = F

for op in (:+, :-, :*)
    @eval $op(a::I, b::I) where {I<:AbstractIntegerIndex} = I($op(value(a),value(b)))
end

for op in (:<, :<=, :>, :>=)
    @eval $op(a::I, b::I) where {I<:AbstractIntegerIndex} = $op(value(a),value(b))
end

(-)(a::AbstractIntegerIndex) = typeof(a)(-value(a))

# Convenience: make a vector indexable using native indices, if possible
# It is possible whenever the size and element type of the vector completely
# determine the index map from native to linear indices
getindex(v::Array, idxn::AbstractIntegerIndex) =
    getindex(v, linear_index(idxn, size(v), eltype(v)))

setindex!(v::Array, val, idxn::AbstractIntegerIndex) =
    setindex!(v, val, linear_index(idxn, size(v), eltype(v)))


"""
A native index is distinguishable from a linear index by its type, but otherwise
it simply wraps an integer and it acts like an integer.
"""
struct NativeIndex{S} <: AbstractIntegerIndex
    value   ::  Int
end

NativeIndex{S}(a::NativeIndex{S}) where {S} = a

value(idx::NativeIndex) = idx.value




"""
The native index list is an identity map from integers to themselves, wrapped
in a `NativeIndex{S}` for some symbol `S`.
"""
struct NativeIndexList{S} <: IndexList{NativeIndex{S}}
    n		::	Int
end

size(list::NativeIndexList) = (list.n,)

getindex(list::NativeIndexList{S}, idx::Int) where {S} = NativeIndex{S}(idx)
getindex(list::NativeIndexList{S}, idxn::NativeIndex{S}) where {S} = value(idxn)

# In this case, we can compute the linear index without reference to
# any dictionary or list:
linear_index(idx::NativeIndex) = value(idx)
# The line below may be called with the size and eltype of an array when indexing
# into an array using a native index.
linear_index(idx::NativeIndex, size, T) = linear_index(idx)


"""
If a dictionary does not have a native index, we use the `DefaultNativeIndex`
type to distinguish between linear and native indices. However, the default
native index simply wraps the integer linear index.
"""
const DefaultNativeIndex = NativeIndex{:default}
const DefaultIndexList = NativeIndexList{:default}



"""
A `ShiftedIndex{S}` is a linear index shifted by `S`. Thus, its value starts at
`1-S` and ranges up to `n-S`. A typical case is `S=1`.
"""
abstract type AbstractShiftedIndex{S} <: AbstractIntegerIndex end

shift(idx::AbstractShiftedIndex{S}) where {S} = S
value(idx::AbstractShiftedIndex) = idx.value

# Here too, as with the DefaultIndex above, we can compute the linear index
# without reference to any dictionary or list
linear_index(idx::AbstractShiftedIndex) = value(idx) + shift(idx)
linear_index(idx::AbstractShiftedIndex, size, T) = linear_index(idx)


struct ShiftedIndex{S} <: AbstractShiftedIndex{S}
    value   ::  Int
end

ShiftedIndex{S}(a::ShiftedIndex{S}) where {S} = a


struct ShiftedIndexList{S,T<:AbstractShiftedIndex{S}} <: IndexList{T}
    n       ::  Int
end

ShiftedIndexList(n, T::Type{<:AbstractShiftedIndex{S}}) where {S} =
	ShiftedIndexList{S,T}(n)

size(list::ShiftedIndexList) = (list.n,)
shift(list::ShiftedIndexList{S,T}) where {S,T} = S

getindex(list::ShiftedIndexList{S,T}, idx::Int) where {S,T} = T(idx-S)
getindex(list::ShiftedIndexList{S,T}, idxn::AbstractShiftedIndex) where {S,T} = value(idxn)+S



########################
# Product indices
########################

# We reuse the CartesianIndex defined in Base
const ProductIndex{N} = CartesianIndex{N}

# Convenience functions
index(idx::ProductIndex) = value(idx)
# Return the index as a tuple
indextuple(idx::ProductIndex) = idx.I


"""
`ProductIndexList` is a list of product indices, suitable for use as the ordering
of a dictionary with tensor product structure.
"""
struct ProductIndexList{N} <: IndexList{ProductIndex{N}}
    size    ::  NTuple{N,Int}
end

# There is something dodgy about defining the index list to be an abstract vector,
# yet having its size be that of a tensor. What matters is that the length of
# the list is correct, and that the list can be indexed with integers.
# All code which assumes that length(list) == size(list)[1] will be fooled...
size(list::ProductIndexList) = list.size

# We use Cartesian indexing, because that is more efficient than linear indexing
# when iterating over product sets: we have to convert int's to cartesian indices
# anyway
Base.IndexStyle(list::ProductIndexList) = Base.IndexCartesian()

# We have to override eachindex, because the default eachindex in Base returns
# a linear index for a vector (because any IndexList is an AbstractVector), and
# the most efficient iteration over product dictionaries is using cartesian indices.
eachindex(list::ProductIndexList) = CartesianIndices(axes(list))

# We convert between integers and product indices using ind2sub and sub2ind
product_native_index(size, idx::Int) = ProductIndex(CartesianIndices(size)[idx])
product_linear_index(size, idxn::ProductIndex) = LinearIndices(size)[indextuple(idxn)...]


# We also know how to convert a tuple
product_native_index(size::NTuple{N,Int}, idx::NTuple{N,Int}) where {N} = ProductIndex(idx)
product_native_index(size::NTuple{N,Int}, idx::ProductIndex{N}) where {N} = idx

getindex(list::ProductIndexList, idx::Int) =
    product_native_index(size(list), idx)
getindex(list::ProductIndexList, idxn::ProductIndex) =
    product_linear_index(size(list), idxn)

# Convert a tuple of int's or a list of int's to a linear index
getindex(list::ProductIndexList{N}, idx::NTuple{N,Int}) where {N} =
    getindex(list, ProductIndex(idx))
getindex(list::ProductIndexList{N}, idx1::Int, idx2::Int, idx::Int...) where {N} =
    getindex(list, ProductIndex{N}(idx1, idx2, idx...))


## We implement a promotion system for two indices.
# The goal is to convert both indices to the Cartesian{N} type, if possible.
# We leave the indices unchanged if it is not possible.

# This is an exhaustive list of types we know how to convert
ProductIndices = Union{NTuple,ProductIndex}

# - Both indices are fine:
promote_product_indices(size::NTuple{N,Int}, idx1::ProductIndex{N}, idx2::ProductIndex{N}) where N = (idx1,idx2)
# - One or both of them are not fine, but they are both in PTI. The formulation
#   includes the case above, since they could both be ProductIndex{N}. By leaving
#   out the type of size, we make sure that the line above is more specific and
#   will be chosen by the compiler, thus avoiding an infinite loop.
promote_product_indices(size, idx1::ProductIndices, idx2::ProductIndices) =
    (product_native_index(size, idx1), product_native_index(size, idx2))
# - At least one of the indices does not have a suitable type, we don't know what to do
promote_product_indices(size, idx1, idx2) = (idx1,idx2)


########################
# Composite indices
########################


const MultilinearIndex = NTuple{2,Int}

"""
`MultiLinearIndexList` is a list of indices suitable for use as the ordering
of a dictionary with composite structure.
"""
struct MultilinearIndexList{O} <: IndexList{MultilinearIndex}
    offsets ::  O
end

const MLIndexList = MultilinearIndexList

size(list::MLIndexList) = (list.offsets[end],)

# Convert a linear index into a multilinear index using the offsets information
function offsets_multilinear_index(offsets, idx::Int)
    i = 0
    while idx > offsets[i+1]
        i += 1
    end
    (i,idx-offsets[i])
end

# and vice-versa
offsets_linear_index(offsets, idx::MultilinearIndex) = offsets[idx[1]] + idx[2]

getindex(list::MLIndexList, idx::Int) = offsets_multilinear_index(list.offsets, idx)
getindex(list::MLIndexList, idx::MultilinearIndex) = offsets_linear_index(list.offsets, idx)



"""
A `MultilinearIndexIterator` iterates over a sequence of linear indices. Each index
is a tuple, where the first entry refers to the linear index, and the second
entry is the linear index.

This resembles the `Flatten` iterator in `Base`, but the latter would yield only the
second entry. For example, the composite indices of `(1:5,1:3)` are:
```
(1,1), (1,2), (1,3), (1,4), (1,5), (2,1), (2,2), (2,3).
```
In contrast, the `Flatten` iterator in Base would yield:
```
1, 2, 3, 4, 5, 1, 2, 3
```
"""
struct MultilinearIndexIterator{L}
    lengths ::  L
end

length(it::MultilinearIndexIterator) = sum(it.lengths)

Base.iterate(it::MultilinearIndexIterator) = (1,1), (1,1)

function Base.iterate(it::MultilinearIndexIterator, state)
    i, j = state
    if j == it.lengths[i]
        next_item = (i+1,1)
    else
        next_item = (i,j+1)
    end
    if next_item[1] <= length(it.lengths)
        next_item, next_item
    end
end

Base.eltype(::Type{MultilinearIndexIterator}) = Tuple{Vararg{Int}}

last(iter::MultilinearIndexIterator) = (length(iter.lengths), iter.lengths[end])
