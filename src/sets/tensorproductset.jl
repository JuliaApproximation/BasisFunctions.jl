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
element(b::TensorProductSet, range::Range) = tensorproduct(b.sets[range]...)
composite_length{TS}(b::TensorProductSet{TS}) = tuple_length(TS)

function TensorProductSet(set::FunctionSet)
    warning("A one element tensor product function set should not exist, use tensorproduct instead of TensorProductSet.")
    set
end

function TensorProductSet(sets::FunctionSet...)
    ELT = promote_type(map(eltype,sets)...)
    psets = map( s -> promote_eltype(s, ELT), sets)
    TensorProductSet{typeof(psets),sum(map(ndims, psets)),ELT}(psets)
end

ndims(b::TensorProductSet, j::Int) = ndims(element(b, j))

## Properties

for op in (:isreal, :is_basis, :is_frame, :is_orthogonal, :is_biorthogonal)
    @eval $op(s::TensorProductSet) = reduce(&, map($op, elements(s)))
end

## Feature methods
for op in (:has_grid, :has_extension, :has_transform)
    @eval $op(b::TensorProductSet) = reduce(&, map($op, elements(b)))
end

for op in (:derivative_set, :antiderivative_set)
    @eval $op{TS,N}(s::TensorProductSet{TS,N}, order::NTuple{N} = tuple(ones(N)...); options...) =
        TensorProductSet( map( i -> $op(element(s,i), order[i]; options...), 1:N)... )
end

promote_eltype{S}(b::TensorProductSet, ::Type{S}) =
    TensorProductSet(map(i -> promote_eltype(i,S), b.sets)...)

resize(b::TensorProductSet, n) = TensorProductSet(map( (b_i,n_i)->resize(b_i, n_i), elements(b), n)...)

function approx_length(b::TensorProductSet, n::Int)
    # Rough approximation: distribute n among all dimensions evenly, rounded upwards
    N = ndims(b)
    m = ceil(Int, n^(1/N))
    tuple([approx_length(element(b, j), m^ndims(b, j)) for j in 1:composite_length(b)]...)
end

extension_size(s::TensorProductSet) = map(extension_size, elements(s))

# It would be odd if the first method below was ever called, because LEN=1 makes
# little sense for a tensor product. But perhaps in generic code somewhere...
name(b::TensorProductSet) = "tensor product (" * name(element(b,1)) * names(b.sets[2:end]...) * ")"
names(b1::FunctionSet) = " x " * name(b1)
names(b1::FunctionSet, b::FunctionSet...) = " x " * name(b1) * names(b)

size(b::TensorProductSet) = map(length, b.sets)
size(b::TensorProductSet, j::Int) = length(b.sets[j])

length(b::TensorProductSet) = prod(size(b))


grid(b::TensorProductSet) = tensorproduct(map(grid, elements(b))...)
#grid(b::TensorProductSet, j::Int) = grid(element(b,j))

# In general, left(f::FunctionSet, j::Int) returns the left of the jth function in the set, not the jth dimension.
# The methods below follow this convention.
left(b::TensorProductSet) = Vec([left(element(b,j)) for j=1:composite_length(b)])
left(b::TensorProductSet, j::Int) = Vec([left(element(b,i),native_index(b,j)[i]) for i=1:composite_length(b)])
#left(b::TensorProductSet, idx::Int, j) = left(b, native_index(b,j), j)
#left(b::TensorProductSet, idxt::NTuple, j) = left(b.sets[j], idxt[j])

right(b::TensorProductSet) = Vec([right(element(b,j)) for j=1:composite_length(b)])
right(b::TensorProductSet, j::Int) = Vec([right(element(b,i),native_index(b,j)[i]) for i=1:composite_length(b)])
#right(b::TensorProductSet, j::Int) = right(element(b,j))
#right(b::TensorProductSet, idx::Int, j) = right(b, native_index(b,j), j)
#right(b::TensorProductSet, idxt::NTuple, j) = right(b.sets[j], idxt[j])


@generated function eachindex{TS}(b::TensorProductSet{TS})
    LEN = tuple_length(TS)
    startargs = fill(1, LEN)
    stopargs = [:(size(b,$i)) for i=1:LEN]
    :(CartesianRange(CartesianIndex{$LEN}($(startargs...)), CartesianIndex{$LEN}($(stopargs...))))
end

@generated function getindex{TS}(b::TensorProductSet{TS}, index::CartesianIndex)
    LEN = tuple_length(TS)
    :(@nref $LEN b d->index[d])
end


checkbounds(b::TensorProductSet, i::Int) = checkbounds(b, native_index(b, i))

function checkbounds(b::TensorProductSet, i)
    for k in 1:composite_length(b)
        checkbounds(element(b, k), i[k])
    end
end

function call_element{TS,SN,LEN}(b::TensorProductSet{TS,SN,LEN}, i, x...)
    z = element(b,1)(i[1], x[1])
    for j = 2:LEN
        z = z * element(b,j)(i[j], x[j])
    end
    z
end

call_element(b::TensorProductSet, i::Int, x, y) = call_element(b, native_index(b, i), x, y)
call_element(b::TensorProductSet, i::Int, x, y, z) = call_element(b, native_index(b, i), x, y, z)
call_element(b::TensorProductSet, i::Int, x, y, z, t) = call_element(b, native_index(b, i), x, y, z, t)

# For now, we assume that each set in the tensor product is a 1D set.
# This may not always be the case.
call_element(b::TensorProductSet, i, x, y) =
    call_element(element(b,1), i[1], x) * call_element(element(b,2), i[2], y)

call_element(b::TensorProductSet, i, x, y, z) =
    call_element(element(b,1), i[1], x) * call_element(element(b,2), i[2], y) * call_element(element(b,3), i[3], z)

call_element(b::TensorProductSet, i, x, y, z, t) =
    call_element(element(b,1), i[1], x) * call_element(element(b,2), i[2], y) * call_element(element(b,3), i[3], z) * call_element(element(b,4), i[4], t)


native_index(b::TensorProductSet, idx::Int) = ind2sub(size(b), idx)
linear_index(b::TensorProductSet, idxn) = sub2ind(size(b), idxn...)

# Transform linear indexing into multivariate indices
#getindex(b::TensorProductSet, i::Int) = getindex(b, ind2sub(b, i))
