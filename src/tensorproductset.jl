# tensorproductbasis.jl

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

function TensorProductSet(sets::FunctionSet...)
    ELT = eltype(map(eltype,sets)...)
    
    sets = initializesets(ELT,sets...)
    TensorProductSet{typeof(sets),map(dim,sets),length(sets),sum(map(dim, sets)),ELT}(sets)
end
⊗(s1::FunctionSet, s::FunctionSet...) = TensorProductSet(s1, s...)

# Disallow TensorProductSets of only one dimension.
function TensorProductSet(set::FunctionSet)
    set
end
# Expand tensorproductsets in a tuple of sets to their individual sets.
function initializesets(ELT,sets::FunctionSet...)
    flattened = FunctionSet[]
    for i = 1:length(sets)
        appendsets(ELT,flattened, sets[i])
    end
    flattened = tuple(flattened...)
end

appendsets(ELT,flattened::Array{FunctionSet,1}, f::FunctionSet) = append!(flattened, [similar(f,ELT,length(f))])

function appendsets(ELT,flattened::Array{FunctionSet,1}, f::TensorProductSet)
    for j = 1:tp_length(f)
        append!(flattened, [similar(set(f,j),ELT,length(set(f,j)))])
    end
end


tensorproduct(b::FunctionSet, n) = TensorProductSet([b for i=1:n]...)

dim{TS,SN,LEN,N,T}(s::TensorProductSet{TS,SN,LEN,N,T}, j::Int) = SN[j]

## Traits

index_dim{TS,SN,LEN,N,T}(::Type{TensorProductSet{TS,SN,LEN,N,T}}) = LEN

#An efficient way to access elements of a Tuple type using index j
@generated function tuple_index{T <: Tuple}(::Type{T}, j)
    :($T.parameters[j])
end

for op in (:is_basis, :is_frame, :isreal, :is_orthogonal, :is_biorthogonal)
    @eval $op(s::TensorProductSet) = reduce(&, map($op, sets(s)))
    
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
#for op in (:has_grid, :has_derivative, :has_transform, :has_extension)
for op in (:has_grid, :has_extension,)
    @eval $op(b::TensorProductSet) = reduce(&, map($op, sets(b)))
end

extension_size(b::TensorProductSet) = map(extension_size, sets(b))

similar(b::TensorProductSet, ELT, n) = TensorProductSet(map((b,n)->similar(b,ELT,n), sets(b), n)...)

function approx_length(b::TensorProductSet, n::Int)
    # Rough approximation: distribute n among all dimensions evenly, rounded upwards
    N = dim(b)
    m = ceil(Int, n^(1/N))
    tuple([approx_length(set(b, j), m^dim(b, j)) for j in 1:tp_length(b)]...)
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

sets(b::TensorProductSet) = b.sets
set(b::TensorProductSet, j::Int) = b.sets[j]
set(b::TensorProductSet, range::Range) = TensorProductSet(b.sets[range]...)
tp_length(b::TensorProductSet) = length(sets(b))

grid(b::TensorProductSet) = TensorProductGrid(map(grid, sets(b))...)
grid(b::TensorProductSet, j::Int) = grid(set(b,j))

left(b::TensorProductSet) = Vec([left(set(b,j)) for j=1:tp_length(b)])
left(b::TensorProductSet, j::Int) = left(set(b,j))
left(b::TensorProductSet, idx::Int, j) = left(b, ind2sub(b,j), j)
left(b::TensorProductSet, idxt::NTuple, j) = left(b.sets[j], idxt[j])

right(b::TensorProductSet) = Vec([right(set(b,j)) for j=1:tp_length(b)])
right(b::TensorProductSet, j::Int) = right(set(b,j))
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
        checkbounds(set(b, k), i[k])
    end
end

function call_element{TS,SN,LEN}(b::TensorProductSet{TS,SN,LEN}, i, x, xt...)
    z = set(b,1)(i[1], x)
    for j = 1:LEN-1
        z = z * set(b,j+1)(i[j], xt[j])
    end
    z
end

call_element{TS,SN}(b::TensorProductSet{TS,SN,2}, i::Int, x, y) = call_element(b, ind2sub(b, i), x, y)
call_element{TS,SN}(b::TensorProductSet{TS,SN,3}, i::Int, x, y, z) = call_element(b, ind2sub(b, i), x, y, z)
call_element{TS,SN}(b::TensorProductSet{TS,SN,4}, i::Int, x, y, z, t) = call_element(b, ind2sub(b, i), x, y, z, t)

call_element{TS,SN}(b::TensorProductSet{TS,SN,1}, i, x) = set(b,1)(i,x)
call_element{TS,SN}(b::TensorProductSet{TS,SN,2}, i, x, y) = set(b,1)(i[1],x) * set(b,2)(i[2], y)
call_element{TS,SN}(b::TensorProductSet{TS,SN,3}, i, x, y, z) = set(b,1)(i[1],x) * set(b,2)(i[2], y) * set(b,3)(i[3], z)
call_element{TS,SN}(b::TensorProductSet{TS,SN,4}, i, x, y, z, t) = set(b,1)(i[1],x) * set(b,2)(i[2],y) * set(b,3)(i[3], z) * set(b,4)(i[4], t)

ind2sub(b::TensorProductSet, idx::Int) = ind2sub(size(b), idx)
sub2ind(b::TensorProductSet, idx...) = sub2ind(size(b), idx...)

# Transform linear indexing into multivariate indices
getindex(b::TensorProductSet, i::Int) = getindex(b, ind2sub(b, i))

# but avoid the 1d case.
getindex{TS,SN}(b::TensorProductSet{TS,SN,1}, i::Int) = SetFunction(b, i)



