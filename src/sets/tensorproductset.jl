# tensorproductset.jl

using Base.Cartesian


"""
A TensorProductSet is itself a set: the tensor product of length(SN) sets with dimension SN[i].

immutable TensorProductSet{TS, SN, LEN, N, T} <: FunctionSet{N,T}

Parameters:
- TS is a tuple of types, representing the (possibly different) types of the sets.
- SN is a tuple of the dimensions of these types.
- LEN is the length of the tuples S and N (the index dimension).
- N is the total dimension of the corresponding space and T the numeric type.
"""
immutable TensorProductSet{TS, SN, LEN, N, T} <: FunctionSet{N,T}
    sets   ::  TS

    TensorProductSet(sets::Tuple) = new(sets)
end

# Generic functions for composite types:
elements(set::TensorProductSet) = set.sets
element(set::TensorProductSet, j::Int) = set.sets[j]
element(b::TensorProductSet, range::Range) = tensorproduct(b.sets[range]...)
composite_length(b::TensorProductSet) = length(elements(b))

# The functions below are depecrated
sets(b::TensorProductSet) = b.sets
set(b::TensorProductSet, j::Int) = b.sets[j]
set(b::TensorProductSet, range::Range) = tensorproduct(b.sets[range]...)

function TensorProductSet(set::FunctionSet)
    warning("This is not okay. A one element tensor product function set.")
    set
end

function TensorProductSet(sets::FunctionSet...)
    ELT = promote_type(map(eltype,sets)...)
    psets = map( s -> promote_eltype(s, ELT), sets)
    TensorProductSet{typeof(psets),map(dim,psets),length(psets),sum(map(dim, psets)),ELT}(psets)
end


dim{TS,SN,LEN,N,T}(s::TensorProductSet{TS,SN,LEN,N,T}) = N

## Traits

index_dim{TS,SN,LEN,N,T}(::Type{TensorProductSet{TS,SN,LEN,N,T}}) = LEN

#An efficient way to access elements of a Tuple type using index j
@generated function tuple_index{T <: Tuple}(::Type{T}, j)
    :($T.parameters[j])
end

for op in (:is_basis, :is_frame, :isreal, :is_orthogonal, :is_biorthogonal)
    @eval $op(s::TensorProductSet) = reduce(&, map($op, elements(s)))

    # The lines below took some experimenting: you can't index into or map over Tuple types
    @eval $op{TS,SN,N,T}(::Type{TensorProductSet{TS,SN,1,N,T}}) = $op(tuple_index(TS,1))
    @eval $op{TS,SN,N,T}(::Type{TensorProductSet{TS,SN,2,N,T}}) =
        $op(tuple_index(TS,1)) & $op(tuple_index(TS,2))
    @eval $op{TS,SN,N,T}(::Type{TensorProductSet{TS,SN,3,N,T}}) =
        $op(tuple_index(TS,1)) & $op(tuple_index(TS,2)) & $op(tuple_index(TS,3))
    @eval $op{TS,SN,N,T}(::Type{TensorProductSet{TS,SN,4,N,T}}) =
        $op(tuple_index(TS,1)) & $op(tuple_index(TS,2)) & $op(tuple_index(TS,3)) & $op(tuple_index(TS,4))
end

## Feature methods
for op in (:has_grid, :has_extension, :has_transform, :has_extension)
    @eval $op(b::TensorProductSet) = reduce(&, map($op, elements(b)))
end

for op in (:derivative_set, :antiderivative_set)
    @eval $op{TS,SN,LEN,N}(s::TensorProductSet{TS,SN,LEN,N}, order::NTuple{N} = tuple(ones(N)...); options...) =
        TensorProductSet( map( i -> $op(set(s,i), order[i]; options...), 1:N)... )
end

extension_size(b::TensorProductSet) = map(extension_size, elements(b))

promote_eltype{TS,SN,LEN,N,T,S}(b::TensorProductSet{TS,SN,LEN,N,T}, ::Type{S}) =
    TensorProductSet(map(i -> promote_eltype(i,S), b.sets)...)

resize(b::TensorProductSet, n) = TensorProductSet(map( (b_i,n_i)->resize(b_i, n_i), elements(b), n)...)

function approx_length(b::TensorProductSet, n::Int)
    # Rough approximation: distribute n among all dimensions evenly, rounded upwards
    N = dim(b)
    m = ceil(Int, n^(1/N))
    tuple([approx_length(element(b, j), m^dim(b, j)) for j in 1:composite_length(b)]...)
end

# It would be odd if the first method below was ever called, because LEN=1 makes
# little sense for a tensor product. But perhaps in generic code somewhere...
name{TS,SN}(b::TensorProductSet{TS,SN,1}) = "tensor product " * name(b.sets[1])
name{TS,SN}(b::TensorProductSet{TS,SN,2}) = "tensor product (" * name(b.sets[1]) * " x " * name(b.sets[2]) * ")"
name{TS,SN}(b::TensorProductSet{TS,SN,3}) = "tensor product (" * name(b.sets[1]) * " x " * name(b.sets[2]) * " x " * name(b.sets[3]) * ")"
name{TS,SN}(b::TensorProductSet{TS,SN,4}) = "tensor product (" * name(b.sets[1]) * " x " * name(b.sets[2]) * " x " * name(b.sets[3]) * " x " * name(b.sets[4]) * ")"

size(b::TensorProductSet) = map(length, b.sets)
size(b::TensorProductSet, j::Int) = length(b.sets[j])

dim{TS,SN}(b::TensorProductSet{TS,SN}, j::Int) = SN[j]

length(b::TensorProductSet) = prod(size(b))


grid(b::TensorProductSet) = tensorproduct(map(grid, elements(b))...)
grid(b::TensorProductSet, j::Int) = grid(element(b,j))

left(b::TensorProductSet) = Vec([left(element(b,j)) for j=1:composite_length(b)])
left(b::TensorProductSet, j::Int) = left(element(b,j))
left(b::TensorProductSet, idx::Int, j) = left(b, ind2sub(b,j), j)
left(b::TensorProductSet, idxt::NTuple, j) = left(b.sets[j], idxt[j])

right(b::TensorProductSet) = Vec([right(element(b,j)) for j=1:composite_length(b)])
right(b::TensorProductSet, j::Int) = right(element(b,j))
right(b::TensorProductSet, idx::Int, j) = right(b, ind2sub(b,j), j)
right(b::TensorProductSet, idxt::NTuple, j) = right(b.sets[j], idxt[j])


@generated function eachindex{TS,SN,LEN}(b::TensorProductSet{TS,SN,LEN})
    startargs = fill(1, LEN)
    stopargs = [:(size(b,$i)) for i=1:LEN]
    :(CartesianRange(CartesianIndex{$LEN}($(startargs...)), CartesianIndex{$LEN}($(stopargs...))))
end

@generated function getindex{TS,SN,LEN}(b::TensorProductSet{TS,SN,LEN}, index::CartesianIndex{LEN})
    :(@nref $LEN b d->index[d])
end


checkbounds(b::TensorProductSet, i::Int) = checkbounds(b, ind2sub(b, i))

function checkbounds{TS,SN,LEN}(b::TensorProductSet{TS,SN,LEN}, i)
    for k in 1:LEN
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

call_element{TS,SN}(b::TensorProductSet{TS,SN,2}, i::Int, x, y) = call_element(b, ind2sub(b, i), x, y)
call_element{TS,SN}(b::TensorProductSet{TS,SN,3}, i::Int, x, y, z) = call_element(b, ind2sub(b, i), x, y, z)
call_element{TS,SN}(b::TensorProductSet{TS,SN,4}, i::Int, x, y, z, t) = call_element(b, ind2sub(b, i), x, y, z, t)

call_element{TS,SN}(b::TensorProductSet{TS,SN,1}, i, x) =
    call_element(element(b,1), i, x)

call_element{TS,SN}(b::TensorProductSet{TS,SN,2}, i, x, y) =
    call_element(element(b,1), i[1], x) * call_element(element(b,2), i[2], y)

call_element{TS,SN}(b::TensorProductSet{TS,SN,3}, i, x, y, z) =
    call_element(element(b,1), i[1], x) * call_element(element(b,2), i[2], y) * call_element(element(b,3), i[3], z)

call_element{TS,SN}(b::TensorProductSet{TS,SN,4}, i, x, y, z, t) =
    call_element(element(b,1), i[1], x) * call_element(element(b,2), i[2], y) * call_element(element(b,3), i[3], z) * call_element(element(b,4), i[4], t)


ind2sub(b::TensorProductSet, idx::Int) = ind2sub(size(b), idx)
sub2ind(b::TensorProductSet, idx...) = sub2ind(size(b), idx...)

# Transform linear indexing into multivariate indices
getindex(b::TensorProductSet, i::Int) = getindex(b, ind2sub(b, i))

# but avoid the 1d case.
getindex{TS,SN}(b::TensorProductSet{TS,SN,1}, i::Int) = SetFunction(b, i)
