
export BlockIndex,
	outerindex,
	innerindex

############
# Overview #
############

# We collect some methods and definitions having to do with various kinds of
# indexing. We make some assumptions on all index sets:
# - an index set is mathematically a set, i.e., it has no duplicates
# - in addition to being a set, the elements can be ordered and the ordering
#   is defined by the iterator of the set.



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
- `linear_index(l::MyIndexList, idxn::MyIndex)` -> this is the inverse map
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

linear_index(a::AbstractArray, idx::Int) = idx


"""
An `AbstractIntegerIndex` represents an integer that is being used as an index
in the BasisFunctions package.

The type implements basic functionality of integers such that one can do
computations with the indices.
"""
abstract type AbstractIntegerIndex end

convert(::Type{T}, idx::AbstractIntegerIndex) where {T<:Integer} = convert(T, value(idx))
convert(::Type{T}, idx::AbstractIntegerIndex) where {T <: AbstractFloat} = convert(T, value(idx))

Base.promote_rule(I::Type{<:AbstractIntegerIndex}, ::Type{Int}) = I
Base.promote_rule(I::Type{<:AbstractIntegerIndex}, F::Type{<:AbstractFloat}) = F

# do basic arithmetics while preserving index type
for op in (:+, :-)
	@eval $op(a::I, b::Integer) where {I <: AbstractIntegerIndex} = I($op(value(a),b))
	@eval $op(a::Integer, b::I) where {I <: AbstractIntegerIndex} = I($op(a,value(b)))
	@eval $op(a::I, b::I) where {I <: AbstractIntegerIndex} = I($op(value(a),b))
	@eval $op(a::I, b::Number) where {I <: AbstractIntegerIndex} = $op(value(a),b)
	@eval $op(a::Number, b::I) where {I <: AbstractIntegerIndex} = $op(a,value(b))
end

for op in (:*, :/, :<, :<=, :>, :>=)
	@eval $op(a::AbstractIntegerIndex, b::Number) = $op(value(a),b)
	@eval $op(a::Number, b::AbstractIntegerIndex) = $op(a,value(b))
end

==(a::AbstractIntegerIndex, b::Number) = value(a) == b
==(a::Number, b::AbstractIntegerIndex) = a == value(b)
Base.abs(a::AbstractIntegerIndex) = typeof(a)(abs(value(a)))

(-)(a::AbstractIntegerIndex) = -value(a)

# Convenience, make a vector indexable using native indices.
# This is possible whenever the size and element type of the
# vector completely determine the index map from native to linear indices
# Note: this is potentially dangerous because the index also behaves like
# an integer, but the index and the integer do not point to the same element
# in the vector v.
getindex(v::AbstractVector, idx::AbstractIntegerIndex) =
    getindex(v, to_linear_index(idx, size(v), eltype(v)))

setindex!(v::AbstractVector, val, idx::AbstractIntegerIndex) =
    setindex!(v, val, to_linear_index(idx, size(v), eltype(v)))


"""
A native index is distinguishable from a linear index by its type, but otherwise
it simply wraps an integer and it acts like an integer.
"""
struct NativeIndex{S} <: AbstractIntegerIndex
    value   ::  Int
end

NativeIndex{S}(a::NativeIndex{S}) where {S} = a

value(idx::NativeIndex) = idx.value

to_linear_index(idx::NativeIndex) = value(idx)
to_linear_index(idx::NativeIndex, size, T) = linear_index(idx)



"""
The native index list is an identity map from integers to themselves, wrapped
in a `NativeIndex{S}` for some symbol `S`.
"""
struct NativeIndexList{S} <: IndexList{NativeIndex{S}}
    n		::	Int
end

size(list::NativeIndexList) = (list.n,)

getindex(list::NativeIndexList{S}, idx::Int) where {S} = NativeIndex{S}(idx)
getindex(list::NativeIndexList{S}, idx::NativeIndex{S}) where {S} = idx

linear_index(list::NativeIndexList{S}, idxn::NativeIndex{S}) where {S} = value(idxn)


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

to_linear_index(idx::AbstractShiftedIndex) = value(idx) + shift(idx)
to_linear_index(idx::AbstractShiftedIndex, size, T) = to_linear_index(idx)


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

getindex(list::ShiftedIndexList{S,T}, idx::LinearIndex) where {S,T} = T(idx-S)
getindex(list::ShiftedIndexList{S}, idx::AbstractShiftedIndex{S}) where {S} = idx

linear_index(list::ShiftedIndexList{S}, idx::AbstractShiftedIndex{S}) where {S} =
	to_linear_index(idx)



########################
# Product indices
########################

# We reuse the CartesianIndex defined in Base
const ProductIndex{N} = CartesianIndex{N}

# Convenience functions
index(idx::ProductIndex) = value(idx)
# Return the index as a tuple
indextuple(idx::ProductIndex) = idx.I

"Known types of product indices."
ProductIndices = Union{NTuple,ProductIndex}



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

product_native_index(size, idx::Int) = ProductIndex(CartesianIndices(size)[idx])
product_native_index(size::NTuple{N,Int}, idx::NTuple{N,Int}) where {N} = ProductIndex(idx)
product_native_index(size::NTuple{N,Int}, idx::ProductIndex{N}) where {N} = idx
product_native_index(size::NTuple, idx1::Int, idx2::Int, indices::Int...) =
	product_native_index(size, (idx1,idx2,indices...))

getindex(list::ProductIndexList, idx...) = product_native_index(size(list), idx...)

product_linear_index(size, idx::ProductIndex) = LinearIndices(size)[indextuple(idx)...]
product_linear_index(list::ProductIndexList, idx::Int) = idx
product_linear_index(list::ProductIndexList, I...) =
	product_linear_index(list, product_native_index(I...))

linear_index(list::ProductIndexList, I...) = product_linear_index(size(list), I...)



## We implement a promotion system for two indices.
# The goal is to convert both indices to the Cartesian{N} type, if possible.
# We leave the indices unchanged if it is not possible.

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


const MultilinearIndex = BlockIndex{1}

const MultilinearIndices = Union{MultilinearIndex,Tuple{Int,Any}}

# At the time of writing, BlockIndex has two fields I and α.
# For BlockIndex{1} they are a tuple with one entry. We provide
# outerindex and innerindex as generic access routines for the values they contain.
outerindex(idx::MultilinearIndex) = idx.I[1]
innerindex(idx::MultilinearIndex) = idx.α[1]

# For compatibility, we do the same for tuples with two elements.
outerindex(idx::Tuple{Int,Any}) = idx[1]
innerindex(idx::Tuple{Int,Any}) = idx[2]


"""
`MultiLinearIndexList` is a list of indices suitable for use as the ordering
of a dictionary with composite structure.
"""
struct MultilinearIndexList{O} <: IndexList{MultilinearIndex}
    offsets ::  O
end

const MLIndexList = MultilinearIndexList

size(list::MLIndexList) = (list.offsets[end],)

"Convert a linear index into a multilinear index using the offsets information"
function offsets_multilinear_index(offsets, idx::Int)
    i = 0
    while idx > offsets[i+1]
        i += 1
    end
    BlockIndex(i, idx-offsets[i])
end

getindex(list::MLIndexList, idx::Int) =
	offsets_multilinear_index(list.offsets, idx)
getindex(list::MLIndexList, idx::MultilinearIndex) = idx
getindex(list::MLIndexList, idx::MultilinearIndices) =
	BlockIndex(outerindex(idx),innerindex(idx))

multilinear_index(list::MLIndexList, idx) = getindex(list, idx)

"Convert the given index into a multilinear index based on offsets."
offsets_linear_index(offsets, idx::MultilinearIndices) =
	offsets[outerindex(idx)] + innerindex(idx)
offsets_linear_index(offsets, idx::Int) = idx

linear_index(list::MLIndexList, idx) = offsets_linear_index(list.offsets, idx)



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

iterate(it::MultilinearIndexIterator) = BlockIndex(1,1), (1,1)
function iterate(it::MultilinearIndexIterator, state)
    i, j = state
    if j == it.lengths[i]
        next = (i+1,1)
    else
        next = (i,j+1)
    end
    if next[1] <= length(it.lengths)
        BlockIndex(next...), next
    end
end

Base.eltype(::Type{MultilinearIndexIterator}) = Tuple{Vararg{Int}}
last(iter::MultilinearIndexIterator) = BlockIndex(length(iter.lengths), iter.lengths[end])
