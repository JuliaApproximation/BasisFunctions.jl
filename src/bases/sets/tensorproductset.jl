# tensorproductset.jl

"""
A TensorProductSet is itself a set: the tensor product of a number of sets.

struct TensorProductSet{TS,N,T} <: FunctionSet{N,T}

Parameters:
- TS is a tuple of types, representing the (possibly different) types of the sets.
- N is the total dimension of the corresponding space and T the numeric type.
"""
struct TensorProductSet{TS,N,T} <: FunctionSet{N,T}
    sets   ::  TS
end

# Generic functions for composite types:
is_composite(set::TensorProductSet) = true
elements(set::TensorProductSet) = set.sets
element(set::TensorProductSet, j::Int) = set.sets[j]
element(s::TensorProductSet, range::Range) = tensorproduct(s.sets[range]...)
composite_length{TS}(s::TensorProductSet{TS}) = tuple_length(TS)

function TensorProductSet(set::FunctionSet)
    warn("A one element tensor product function set should not exist, use tensorproduct instead of TensorProductSet.")
    set
end

function TensorProductSet(sets::FunctionSet...)
    ELT = promote_type(map(eltype,sets)...)
    psets = map( s -> promote_eltype(s, ELT), sets)
    TensorProductSet{typeof(psets),sum(map(ndims, psets)),ELT}(psets)
end

ndims(s::TensorProductSet, j::Int) = ndims(element(s, j))

^(s::FunctionSet, n::Int) = tensorproduct(s, n)

## Properties

for op in (:isreal, :is_basis, :is_frame, :is_orthogonal, :is_biorthogonal)
    @eval $op(s::TensorProductSet) = reduce(&, map($op, elements(s)))
end

## Indices

# A tensor product set s has three types of indices:
# - Linear index: this is an Int, ranging from 1 to length(s)
# - Multilinear index: tuple of Ints, or CartesianIndex. Each element of the tuple
#   is a linear index for the corresponding element of the set
# - Native index: any other tuple type. Each element is the native index of the
#   corresponding element of the set.
#
# We define some conversion routines below.

# Convert the given index to a linear index.
# - If it is an Int, it already is a linear index
linear_index(s::TensorProductSet, i::Int) = i
# - If the argument is a tuple of integers or a CartesianIndex, then it is
#   a multilinear index.
linear_index{N}(s::TensorProductSet, i::NTuple{N,Int}) = sub2ind(size(s), i...)
linear_index(s::TensorProductSet, i::CartesianIndex) = sub2ind(size(s), i.I...)
# - If its type is anything else, it may be a tuple of native indices
linear_index(s::TensorProductSet, idxn::Tuple) = linear_index(s, map(linear_index, elements(s), idxn))

# Convert the given index to a multilinear index.
# - A tuple of Int's is already a multilinear index
multilinear_index{N}(s::TensorProductSet, idx::NTuple{N,Int}) = idx
# - From linear index to multilinear
multilinear_index(s::TensorProductSet, idx::Int) = ind2sub(size(s), idx)
# - Convert a CartesianIndex to a tuple (! this uses CartesianIndex internals currently)
multilinear_index(s::TensorProductSet, idx::CartesianIndex) = idx.I
# - From any other tuple
multilinear_index(s::TensorProductSet, idx::Tuple) = map(linear_index, elements(s), idx)

# Convert the given index to a native index.
# - From a multilinear index
native_index{N}(s::TensorProductSet, idx::NTuple{N,Int}) = map(native_index, elements(s), idx)
# - Assume that another kind of tuple is the native index
native_index(s::TensorProductSet, idx::Tuple) = idx
# - From a linear index
native_index(s::TensorProductSet, idx::Int) = native_index(multilinear_index(s, idx))

# Convert an index into an index that is indexable, with length equal to the length of the product set
indexable_index(set::TensorProductSet, idx::Tuple) = idx
indexable_index(set::TensorProductSet, idx::Int) = multilinear_index(set, idx)
indexable_index(set::TensorProductSet, idx::CartesianIndex) = multilinear_index(set, idx)


## Feature methods

for op in (:has_grid, :has_extension, :has_derivative, :has_antiderivative)
    @eval $op(s::TensorProductSet) = reduce(&, map($op, elements(s)))
end

has_grid_transform(s::TensorProductSet, dgs, grid::TensorProductGrid) =
    reduce(&, map(has_transform, elements(s), elements(grid)))

has_grid_transform(s::TensorProductSet, dgs, grid::AbstractGrid) = false

for op in (:derivative_set, :antiderivative_set)
    @eval $op{TS,N}(s::TensorProductSet{TS,N}, order::NTuple{N}; options...) =
        TensorProductSet( map( i -> $op(element(s,i), order[i]; options...), 1:N)... )
end

set_promote_eltype{S}(s::TensorProductSet, ::Type{S}) =
    TensorProductSet(map(i -> promote_eltype(i,S), s.sets)...)

resize(s::TensorProductSet, n) = TensorProductSet(map( (s_i,n_i)->resize(s_i, n_i), elements(s), n)...)
resize(s::TensorProductSet, n::Int) = resize(s, approx_length(s, n))

nested_vector{TS,N,T}(set::TensorProductSet{TS,N,T}, x::SVector{N}) = x

# Delegate in_support to _in_support with the composing sets as extra arguments,
# in order to avoid extra memory allocation.
in_support(set::TensorProductSet, idx, x) = _in_support(set, elements(set), indexable_index(idx), x)

# This line is a bit slower than the lines below:
_in_support(::TensorProductSet, sets, idx, x) = reduce(&, map(in_support, sets, idx, x))

# That is why we handcode a few cases:
_in_support(::TensorProductSet, sets, idx::NTuple{1,Int}, x) =
    in_support(sets[1], idx[1], x[1])

_in_support(::TensorProductSet, sets, idx::NTuple{2,Int}, x) =
    in_support(sets[1], idx[1], x[1]) && in_support(sets[2], idx[2], x[2])

_in_support(::TensorProductSet, sets, idx::NTuple{3,Int}, x) =
    in_support(sets[1], idx[1], x[1]) && in_support(sets[2], idx[2], x[2]) && in_support(sets[3], idx[3], x[3])

_in_support(::TensorProductSet, sets, idx::NTuple{4,Int}, x) =
    in_support(sets[1], idx[1], x[1]) && in_support(sets[2], idx[2], x[2]) && in_support(sets[3], idx[3], x[3]) && in_support(sets[4], idx[4], x[4])

function approx_length(s::TensorProductSet, n::Int)
    # Rough approximation: distribute n among all dimensions evenly, rounded upwards
    N = ndims(s)
    m = ceil(Int, n^(1/N))
    tuple([approx_length(element(s, j), m^ndims(s, j)) for j in 1:composite_length(s)]...)
end

extension_size(s::TensorProductSet) = map(extension_size, elements(s))

# It would be odd if the first method below was ever called, because LEN=1 makes
# little sense for a tensor product. But perhaps in generic code somewhere...
name(s::TensorProductSet) = "tensor product (" * name(element(s,1)) * names(s.sets[2:end]...) * ")"
names(s1::FunctionSet) = " x " * name(s1)
names(s1::FunctionSet, s::FunctionSet...) = " x " * name(s1) * names(s...)

size(s::TensorProductSet) = map(length, s.sets)
size(s::TensorProductSet, j::Int) = length(s.sets[j])

length(s::TensorProductSet) = prod(size(s))

getindex(s::TensorProductSet, ::Colon, i::Int) = (@assert composite_length(s)==2; element(s,1))
getindex(s::TensorProductSet, i::Int, ::Colon) = (@assert composite_length(s)==2; element(s,2))
getindex(s::TensorProductSet, ::Colon, ::Colon) = (@assert composite_length(s)==2; s)

getindex(s::TensorProductSet, ::Colon, i::Int, j::Int) = (@assert composite_length(s)==3; element(s,1))
getindex(s::TensorProductSet, i::Int, ::Colon, j::Int) = (@assert composite_length(s)==3; element(s,2))
getindex(s::TensorProductSet, i::Int, j::Int, ::Colon) = (@assert composite_length(s)==3; element(s,3))
getindex(s::TensorProductSet, ::Colon, ::Colon, i::Int) =
    (@assert composite_length(s)==3; TensorProductSet(element(s,1),element(s,2)))
getindex(s::TensorProductSet, ::Colon, i::Int, ::Colon) =
    (@assert composite_length(s)==3; TensorProductSet(element(s,1),element(s,3)))
getindex(s::TensorProductSet, i::Int, ::Colon, ::Colon) =
    (@assert composite_length(s)==3; TensorProductSet(element(s,2),element(s,3)))
getindex(s::TensorProductSet, ::Colon, ::Colon, ::Colon) = (@assert composite_length(s)==3; s)


grid(s::TensorProductSet) = TensorProductGrid(map(grid, elements(s))...)
#grid(b::TensorProductSet, j::Int) = grid(element(b,j))

# In general, left(f::FunctionSet, j::Int) returns the left of the jth function in the set, not the jth dimension.
# The methods below follow this convention.
left(s::TensorProductSet) = SVector(map(left, elements(s)))
left{TS,N,T}(s::TensorProductSet{TS,N,T}, j::Int) = SVector{N}([left(element(s,i),multilinear_index(s,j)[i]) for i=1:composite_length(s)])
#left(b::TensorProductSet, idx::Int, j) = left(b, multilinear_index(b,j), j)
#left(b::TensorProductSet, idxt::NTuple, j) = left(b.sets[j], idxt[j])

right(s::TensorProductSet) = SVector(map(right, elements(s)))
right{TS,N,T}(s::TensorProductSet{TS,N,T}, j::Int) = SVector{N}([right(element(s,i),multilinear_index(s,j)[i]) for i=1:composite_length(s)])
#right(b::TensorProductSet, j::Int) = right(element(b,j))
#right(b::TensorProductSet, idx::Int, j) = right(b, multilinear_index(b,j), j)
#right(b::TensorProductSet, idxt::NTuple, j) = right(b.sets[j], idxt[j])


@generated function eachindex{TS}(s::TensorProductSet{TS})
    LEN = tuple_length(TS)
    startargs = fill(1, LEN)
    stopargs = [:(size(s,$i)) for i=1:LEN]
    :(CartesianRange(CartesianIndex{$LEN}($(startargs...)), CartesianIndex{$LEN}($(stopargs...))))
end

# Convert CartesianIndex argument to a tuple
getindex(s::TensorProductSet, idx::CartesianIndex) = getindex(s, idx.I)


# Routine for linear indices: convert into multilinear index
eval_element(s::TensorProductSet, i::Int, x) = eval_element_native(s, multilinear_index(s, i), x)

# For any other type of index: assume it is a native one.
eval_element(s::TensorProductSet, i, x) = eval_element_native(s, i, x)

# We have to pass on the elements of s as an extra argument in order to avoid
# memory allocations in the lines below
eval_element_native(s::TensorProductSet, i, x) = _eval_element_native(s, elements(s), i, x)

# For now, we assume that each set in the tensor product is a 1D set.
# This may not always be the case.
_eval_element_native{TS}(s::TensorProductSet{TS,1}, sets, i, x) =
    eval_element(sets[1], i[1], x[1])

_eval_element_native{TS}(s::TensorProductSet{TS,2}, sets, i, x) =
    eval_element(sets[1], i[1], x[1]) * eval_element(sets[2], i[2], x[2])

_eval_element_native{TS}(s::TensorProductSet{TS,3}, sets, i, x) =
    eval_element(sets[1], i[1], x[1]) * eval_element(sets[2], i[2], x[2]) * eval_element(sets[3], i[3], x[3])

_eval_element_native{TS}(s::TensorProductSet{TS,4}, sets, i, x) =
    eval_element(sets[1], i[1], x[1]) * eval_element(sets[2], i[2], x[2]) * eval_element(sets[3], i[3], x[3]) * eval_element(sets[4], i[4], x[4])

# Generic implementation, slightly slower
_eval_element_native(s::TensorProductSet, sets, i, x) =
    reduce(*, map(eval_element, sets, i, x))





"Return a list of all tensor product indices (1:s+1)^n."
index_set_tensorproduct(s,n) = CartesianRange(CartesianIndex(fill(1,n)...), CartesianIndex(fill(s+1,n)...))

"Return a list of all indices of total degree at most s, in n dimensions."
function index_set_total_degree(s, n)
    # We make a list of arrays because that is easier
    I = _index_set_total_degree(s, n)
    # and then we convert to tuples
    [tuple((1+i)...) for i in I]
end

function _index_set_total_degree(s, n)
    if n == 1
        I = [[i] for i in 0:s]
    else
        I = Array(Array{Int,1},0)
        I_rec = _index_set_total_degree(s, n-1)
        for idx in I_rec
            for m in 0:s-sum(abs(idx))
                push!(I, [idx...; m])
            end
        end
        I
    end
end

"Return a list of all indices in an n-dimensional hyperbolic cross."
function index_set_hyperbolic_cross(s, n, α = 1)
    I = _index_set_hyperbolic_cross(s, n, α)
    [tuple((1+i)...) for i in I]
end

function _index_set_hyperbolic_cross(s, n, α = 1)
    if n == 1
        smax = floor(Int, s^(1/α))-1
        I = [[i] for i in 0:smax]
    else
        I = Array(Array{Int,1},0)
        I_rec = _index_set_total_degree(s, n-1)
        for idx in I_rec
            for m in 0:floor(Int,s^(1/α)/prod(1+abs(idx)))-1
                push!(I, [idx...; m])
            end
        end
        I
    end
end
