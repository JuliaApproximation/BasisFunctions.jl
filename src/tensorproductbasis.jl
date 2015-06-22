# tensorproductbasis.jl

using Base.Cartesian


# A TensorProductSet is itself a set: the tensor product of SN sets.
# Parameter S is a tuple of types, representing the SN (possibly different) types of the sets.
# N is the total dimension of the corresponding space and T the numeric type as usual.
immutable TensorProductSet{S, SN, N, T} <: AbstractFunctionSet{N,T}
    sets   ::  S

    TensorProductSet(sets::Tuple) = new(sets)
end

TensorProductSet(sets::AbstractFunctionSet...) = TensorProductSet{typeof(sets),length(sets),sum(map(dim, sets)), numtype(sets[1])}(sets)

tensorproduct(b::AbstractFunctionSet1d, n) = TensorProductSet(tuple([b for i=1:n]...))

# It would be odd if the first method below was ever called, because SN=1 makes
# little sense. But perhaps in generic code somewhere...
name{S}(b::TensorProductSet{S,1}) = "tensor product " * name(b.sets[1])
name{S}(b::TensorProductSet{S,2}) = "tensor product (" * name(b.sets[1]) * " x " * name(b.sets[2]) * ")"
name{S}(b::TensorProductSet{S,3}) = "tensor product (" * name(b.sets[1]) * " x " * name(b.sets[2]) * " x " * name(b.sets[3]) * ")"
name{S}(b::TensorProductSet{S,3}) = "tensor product (" * name(b.sets[1]) * " x " * name(b.sets[2]) * " x " * name(b.sets[3]) * " x " * name(b.sets[4]) * ")"

size(b::TensorProductSet) = map(length, b.sets)

size(b::TensorProductSet, j::Int) = length(b.sets[j])

length(b::TensorProductSet) = prod(size(b))

sets(b::TensorProductSet) = b.sets
set(b::TensorProductSet, i::Int) = b.sets[i]

@generated function eachindex{S,SN}(b::TensorProductSet{S,SN})
    startargs = fill(1, SN)
    stopargs = [:(size(b,$i)) for i=1:SN]
    :(CartesianRange(CartesianIndex{$SN}($(startargs...)), CartesianIndex{$SN}($(stopargs...))))
end

@generated function getindex{S,SN}(b::TensorProductSet{S,SN}, index::CartesianIndex{SN})
    :(@nref $SN b d->index[d])
end


index_dim{S,SN}(::TensorProductSet{S,SN}) = SN
index_dim{S,SN}(::Type{TensorProductSet{S,SN}}) = SN
index_dim{B <: TensorProductSet}(::Type{B}) = index_dim(super(B))


function call{S,SN,N,T}(b::TensorProductSet{S,SN,N,T}, i, x, xt...)
    z = set(b,1)(i[1], x)
    for j = 1:length(i)
        z = z * set(b,j+1)(i[j], xt[j])
    end
    z
end

call{S}(b::TensorProductSet{S,1}, i, x) = set(b,1)(i,x)
call{S}(b::TensorProductSet{S,2}, i, x, y) = set(b,1)(i[1],x) * set(b,2)(i[2], y)
call{S}(b::TensorProductSet{S,3}, i, x, y, z) = set(b,1)(i[1],x) * set(b,2)(i[2], y) * set(b,3)(i[3], z)
call{S}(b::TensorProductSet{S,4}, i, x, y, z, t) = set(b,1)(i[1],x) * set(b,2)(i[2],y) * set(b,3)(i[3], z) * set(b,4)(i[4], t)

isreal(b::TensorProductSet) = reduce(&, map(isreal, sets(b)))
isreal{S,SN,N,T}(::Type{TensorProductSet{S,SN,N,T}}) = reduce(&, map(isreal, S))

ind2sub(b::TensorProductSet, idx::Int) = ind2sub(size(b), idx)
sub2ind(b::TensorProductSet, idx...) = sub2ind(size(b), idx...)

# Transform linear indexing into multivariate indices
getindex(b::TensorProductSet, i::Int) = getindex(b, ind2sub(b, i))



immutable TensorProductBasis{B <: AbstractBasis1d, G, N, T} <: AbstractBasis{N,T}
	bases	::	NTuple{N,B}
	n		::	NTuple{N,Int}
	ntot	::	Int
	grid	::	TensorProductGrid{G,N,T}

	TensorProductBasis(b::NTuple) = new(b, map(t -> length(t), b), prod(map(t->length(t), b)), TensorProductGrid(map(t->grid(t),b)) )
end

TensorProductBasis{B <: AbstractBasis1d,N}(b::NTuple{N,B}) = TensorProductBasis{B,gridtype(b[1]),N,numtype(b[1])}(b)

tensorproduct(b::AbstractBasis1d, n) = TensorProductBasis(tuple([b for i=1:n]...))

name(b::TensorProductBasis) = "tensor product " * name(b.bases[1])

isreal(b::TensorProductBasis) = isreal(b.bases[1])
isreal{B <: AbstractBasis1d,G,N,T}(::Type{TensorProductBasis{B,G,N,T}}) = isreal(B)


# has_compact_support{B}(::Type{TensorProductBasis{B}}) = has_compact_support(B)
# has_compact_support{B,N}(::Type{TensorProductBasis{B,N}}) = has_compact_support(B)

length(b::TensorProductBasis) = b.ntot

size(b::TensorProductBasis) = b.n
size(b::TensorProductBasis, j) = b.n[j]

@generated function eachindex{B,G,N,T}(b::TensorProductBasis{B,G,N,T})
    startargs = fill(1, N)
    stopargs = [:(size(b,$i)) for i=1:N]
    :(CartesianRange(CartesianIndex{$N}($(startargs...)), CartesianIndex{$N}($(stopargs...))))
end

@generated function getindex{B,G,N,T}(b::TensorProductBasis{B,G,N,T}, index::CartesianIndex{N})
    :(@nref $N b d->index[d])
end

index_dim{B,G,N,T}(::TensorProductBasis{B,G,N,T}) = N
index_dim{B,G,N,T}(::Type{TensorProductBasis{B,G,N,T}}) = N
index_dim{B <: TensorProductBasis}(::Type{B}) = index_dim(super(B))



grid(b::TensorProductBasis) = b.grid

ind2sub(b::TensorProductBasis, idx::Int) = ind2sub(size(b), idx)

sub2ind(b::TensorProductBasis, idx...) = sub2ind(size(b), idx...)

# Translate calls with integer index into calls with tuple index
call(b::TensorProductBasis, idx::Int, x...) = call(b, ind2sub(b,idx), x...)

call!{T <: Number}(b::TensorProductBasis, idx::Int, result, x::T...) = call!(b, ind2sub(b,idx), result, x...)

call{B,G,N}(b::TensorProductBasis{B,G,N}, idxt::NTuple{N}, x...) = prod([call(b.bases[i], idxt[i], x[i]) for i=1:N])

# Write out common specific cases to make it easier for the compiler to inline
# This could be generalized using Base.Cartesian
call{B,G}(b::TensorProductBasis{B,G,1}, idxt::NTuple{1}, x) = call(b.bases[1], idxt[1], x)
call{B,G}(b::TensorProductBasis{B,G,2}, idxt::NTuple{2}, x, y) = call(b.bases[1], idxt[1], x)*call(b.bases[2], idxt[2], y)
call{B,G}(b::TensorProductBasis{B,G,3}, idxt::NTuple{3}, x, y, z) = call(b.bases[1], idxt[1], x)*call(b.bases[2], idxt[2], y)*call(b.bases[3], idxt[3], z)

#@ngenerate N nothing call!{B,G,N}(b::TensorProductBasis{B,G,N}, idxt::NTuple{N}, result::AbstractArray, x::AbstractArray...) = broadcast!( (@ntuple N t) -> call(b, idxt, (@ntuple N t)...), result, x...)



support(b::TensorProductBasis, idx::Int) = support(b, ind2sub(b,idx))

support{B,G,N}(b::TensorProductBasis{B,G,N}, idxt::NTuple{N}) = [support(b.bases[i], idxt[i]) for i=1:N]


support(b::TensorProductBasis, idx::Int, dim) = support(b, ind2sub(b,idx), dim)

support(b::TensorProductBasis, idxt::NTuple, dim) = support(b.bases[dim], idxt[dim])

left(b::TensorProductBasis, dim::Int) = left(b.bases[dim])
left(b::TensorProductBasis, idx::Int, dim) = left(b, ind2sub(b,idx), dim)
left(b::TensorProductBasis, idxt::NTuple, dim) = left(b.bases[dim], idxt[dim])


right(b::TensorProductBasis, dim::Int) = right(b.bases[dim])
right(b::TensorProductBasis, idx::Int, dim) = right(b, ind2sub(b,idx), dim)
right(b::TensorProductBasis, idxt::NTuple, dim) = right(b.bases[dim], idxt[dim])


# Code below are remnants from some experiments - todo: move functionality elsewhere and remove
# waypoints(b::TensorProductBasis, idx::Int, dim) = waypoints(b, ind2sub(b,idx), dim)
# 
# waypoints(b::TensorProductBasis, idxt::NTuple, dim) = waypoints(b.bases[dim], idxt[dim])
# 
# 
# overlap(b::TensorProductBasis, idx1::Int, idx2::Int) = overlap(b, ind2sub(b, idx1), ind2sub(b, idx2))
# 
# overlap{B,G,N}(b::TensorProductBasis{B,G,N}, idx1::NTuple{N}, idx2::NTuple{N}) = reduce(&, [overlap(b.bases[i], idx1[i], idx2[i]) for i=1:N])
# 
# 
# function active_indices{B,G}(b::TensorProductBasis{B,G,2}, x...)
# 	i1 = active_indices(b.bases[1], x[1])
# 	i2 = active_indices(b.bases[2], x[2])
# 	I = Array(Int, length(i1)*length(i2))
#	j = 0
#	for k = 1:length(i1)
#		for l = 1:length(i2)
#			j = j+1
#			I[j] = sub2ind(b, i1[k], i2[l])
#		end
#	end
#	I
#end


