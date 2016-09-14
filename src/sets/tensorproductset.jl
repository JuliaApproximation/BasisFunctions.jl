# tensorproductset.jl

using Base.Cartesian


"""
A TensorProductSet is itself a set: the tensor product of a number of sets.

immutable TensorProductSet{TS,N,T} <: FunctionSet{N,T}

Parameters:
- TS is a tuple of types, representing the (possibly different) types of the sets.
- N is the total dimension of the corresponding space and T the numeric type.
"""
immutable TensorProductSet{TS,N,T} <: FunctionSet{N,T}
    sets   ::  TS
end

# Generic functions for composite types:
elements(set::TensorProductSet) = set.sets
element(set::TensorProductSet, j::Int) = set.sets[j]
element(s::TensorProductSet, range::Range) = tensorproduct(s.sets[range]...)
composite_length{TS}(s::TensorProductSet{TS}) = tuple_length(TS)

function TensorProductSet(set::FunctionSet)
    warning("A one element tensor product function set should not exist, use tensorproduct instead of TensorProductSet.")
    set
end

function TensorProductSet(sets::FunctionSet...)
    ELT = promote_type(map(eltype,sets)...)
    psets = map( s -> promote_eltype(s, ELT), sets)
    TensorProductSet{typeof(psets),sum(map(ndims, psets)),ELT}(psets)
end

ndims(s::TensorProductSet, j::Int) = ndims(element(s, j))

## Properties

for op in (:isreal, :is_basis, :is_frame, :is_orthogonal, :is_biorthogonal)
    @eval $op(s::TensorProductSet) = reduce(&, map($op, elements(s)))
end

## Feature methods
for op in (:has_grid, :has_extension, :has_transform, :has_derivative)
    @eval $op(s::TensorProductSet) = reduce(&, map($op, elements(s)))
end

for op in (:derivative_set, :antiderivative_set)
    @eval $op{TS,N}(s::TensorProductSet{TS,N}, order::NTuple{N}; options...) =
        TensorProductSet( map( i -> $op(element(s,i), order[i]; options...), 1:N)... )
end

promote_eltype{S}(s::TensorProductSet, ::Type{S}) =
    TensorProductSet(map(i -> promote_eltype(i,S), s.sets)...)

resize(s::TensorProductSet, n) = TensorProductSet(map( (s_i,n_i)->resize(s_i, n_i), elements(s), n)...)

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
names(s1::FunctionSet, s::FunctionSet...) = " x " * name(s1) * names(s)

size(s::TensorProductSet) = map(length, s.sets)
size(s::TensorProductSet, j::Int) = length(s.sets[j])

length(s::TensorProductSet) = prod(size(s))


grid(s::TensorProductSet) = tensorproduct(map(grid, elements(s))...)
#grid(b::TensorProductSet, j::Int) = grid(element(b,j))

# In general, left(f::FunctionSet, j::Int) returns the left of the jth function in the set, not the jth dimension.
# The methods below follow this convention.
left(s::TensorProductSet) = Vec([left(element(s,j)) for j=1:composite_length(s)])
left(s::TensorProductSet, j::Int) = Vec([left(element(s,i),multilinear_index(s,j)[i]) for i=1:composite_length(s)])
#left(b::TensorProductSet, idx::Int, j) = left(b, multilinear_index(b,j), j)
#left(b::TensorProductSet, idxt::NTuple, j) = left(b.sets[j], idxt[j])

right(s::TensorProductSet) = Vec([right(element(s,j)) for j=1:composite_length(s)])
right(s::TensorProductSet, j::Int) = Vec([right(element(s,i),multilinear_index(s,j)[i]) for i=1:composite_length(s)])
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
getindex(s::TensorProductSet, i::CartesianIndex{2}) = getindex(s, (i[1],i[2]))
getindex(s::TensorProductSet, i::CartesianIndex{3}) = getindex(s, (i[1],i[2],i[3]))
getindex(s::TensorProductSet, i::CartesianIndex{4}) = getindex(s, (i[1],i[2],i[3],i[4]))

# This is a more general, but less readable, definition of getindex for CartesianIndex
# @generated function getindex{TS}(s::TensorProductSet{TS}, index::CartesianIndex)
#     LEN = tuple_length(TS)
#     :(@nref $LEN s d->index[d])
# end



# Routine for linear indices: convert into multilinear index
call_element(s::TensorProductSet, i::Int, x) = call_element_native(s, multilinear_index(s, i), x)

# For any other type of index: assume it is a native one.
call_element(s::TensorProductSet, i, x) = call_element_native(s, i, x)

# For now, we assume that each set in the tensor product is a 1D set.
# This may not always be the case.
call_element_native{TS}(s::TensorProductSet{TS,2}, i, x) =
    call_element(element(s,1), i[1], x[1]) * call_element(element(s,2), i[2], x[2])

call_element_native{TS}(s::TensorProductSet{TS,3}, i, x) =
    call_element(element(s,1), i[1], x[1]) * call_element(element(s,2), i[2], x[2]) * call_element(element(s,3), i[3], x[3])

call_element_native{TS}(s::TensorProductSet{TS,4}, i, x) =
    call_element(element(s,1), i[1], x[1]) * call_element(element(s,2), i[2], x[2]) * call_element(element(s,3), i[3], x[3]) * call_element(element(s,4), i[4], x[4])


# A multilinear index of a set is a tuple consisting of linear indices of
# each of the subsets. Its type is a tuple of integers.
multilinear_index(s::TensorProductSet, idx::Int) = ind2sub(size(s), idx)
# In contrast, the native index is a tuple consisting of native indices of each
# of the subsets. Its type depends on the type of the native indices.
function native_index(s::TensorProductSet, idx::Int)
    # Compute multilinear index first
    idx_ml = multilinear_index(s, idx)
    # Map each linear index to a native index
    map(native_index, elements(s), idx_ml)
end

# Convert to linear index. If the argument is a tuple of integers, it can be assumed to
# be a multilinear index. Same for CartesianIndex.
linear_index(s::TensorProductSet, i::NTuple{2,Int}) = sub2ind(size(s), i[1], i[2])
linear_index(s::TensorProductSet, i::NTuple{3,Int}) = sub2ind(size(s), i[1], i[2], i[3])
linear_index(s::TensorProductSet, i::NTuple{4,Int}) = sub2ind(size(s), i[1], i[2], i[3], i[4])
linear_index(s::TensorProductSet, i::CartesianIndex{2}) = sub2ind(size(s), i[1], i[2])
linear_index(s::TensorProductSet, i::CartesianIndex{3}) = sub2ind(size(s), i[1], i[2], i[3])
linear_index(s::TensorProductSet, i::CartesianIndex{4}) = sub2ind(size(s), i[1], i[2], i[3], i[4])

# If its type is anything else, it is a tuple of native indices.
linear_index(s::TensorProductSet, idxn) = linear_index(s, map(linear_index, elements(s), idxn))
