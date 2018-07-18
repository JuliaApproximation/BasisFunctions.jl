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



##################
# Native indices
##################

# Dictionaries can be indexed in various ways. We assume that the semantics
# of the index is determined by its type and, moreover, that linear indices
# are always Int's. This means that no other index can have type Int.
const LinearIndex = Int

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

# Convenience: make a vector indexable using native indices, if possible
# It is possible whenever the size and element type of the vector completely
# determine the index map from native to linear indices
getindex(v::Array, idxn::NativeIndex) =
	getindex(v, linear_index(idxn, size(v), eltype(v)))




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

# In this case, we can compute the linear index without reference to
# any dictionary or list:
linear_index(idx::DefaultNativeIndex) = value(idx)
# The line below may be called with the size and eltype of an array when indexing
# into an array using a native index, see dictionary.jl
linear_index(idx::DefaultNativeIndex, size, T) = linear_index(idx)


"""
A `ShiftedIndex{S}` is a linear index shifted by `S`. Thus, its value starts at
`1-S` and ranges up to `n-S`. A typical case is `S=1`.

We use the `S` type parameter of `NativeIndex` to store the shift.
"""
const ShiftedIndex{S} = NativeIndex{S}

shift(idx::ShiftedIndex{S}) where {S} = S

struct ShiftedIndexList{S} <: IndexList{ShiftedIndex{S}}
    n       ::  Int
end

# Default value of the shift is 1
ShiftedIndexList(n::Int) = ShiftedIndexList{1}(n)

size(list::ShiftedIndexList) = (list.n,)
shift(list::ShiftedIndexList{S}) where {S} = S

getindex(list::ShiftedIndexList{S}, idx::LinearIndex) where {S} = ShiftedIndex{S}(idx-S)
getindex(list::ShiftedIndexList{S}, idxn::ShiftedIndex) where {S} = value(idxn)+S

# Here too, as with the DefaultIndex above, we can compute the linear index
# without reference to any dictionary or list
linear_index(idx::ShiftedIndex) = value(idx) + shift(idx)
linear_index(idx::ShiftedIndex, size, T) = linear_index(idx)


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
product_native_index(size, idx::LinearIndex) = (VERSION < v"0.7-") ?
    ProductIndex(ind2sub(size, idx)) : ProductIndex(CartesianIndices(size)[idx])
product_linear_index(size, idxn::ProductIndex) = (VERSION < v"0.7-") ?
    sub2ind(size, indextuple(idxn)...) : LinearIndices(size)[indextuple(idxn)...]


# We also know how to convert a tuple
product_native_index(size::NTuple{N,Int}, idx::NTuple{N,Int}) where {N} = ProductIndex(idx)
product_native_index(size::NTuple{N,Int}, idx::ProductIndex{N}) where {N} = idx

getindex(list::ProductIndexList, idx::LinearIndex) =
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
ProductIndices = Union{NTuple,ProductIndex,Int}

# - Both indices are fine:
promote_product_indices(size::NTuple{N,Int}, idx1::ProductIndex{N}, idx2::ProductIndex{N}) where {N} = (idx1,idx2)
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

getindex(list::MLIndexList, idx::LinearIndex) = offsets_multilinear_index(list.offsets, idx)
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

if VERSION < v"0.7-"
    start(it::MultilinearIndexIterator) = (1,1)

    function next(it::MultilinearIndexIterator, state)
        i = state[1]
        j = state[2]
        if j == it.lengths[i]
            nextstate = (i+1,1)
        else
            nextstate = (i,j+1)
        end
        (state, nextstate)
    end

    done(it::MultilinearIndexIterator, state) = state[1] > length(it.lengths)

else
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
end
