# tensorproductbasis.jl

using Base.Cartesian



immutable TensorProductBasis{B <: AbstractBasis1d, G, N, T} <: AbstractBasis{N,T}
	bases	::	NTuple{N,B}
	n		::	NTuple{N,Int}
	ntot	::	Int
	grid	::	TensorProductGrid{G,N,T}

	TensorProductBasis(b::NTuple) = new(b, map(t -> length(t), b), prod(map(t->length(t), b)), TensorProductGrid(map(t->natural_grid(t),b)) )
end

TensorProductBasis{B <: AbstractBasis1d,N}(b::NTuple{N,B}) = TensorProductBasis{B,gridtype(b[1]),N,numtype(b[1])}(b)

tensorproduct(b::AbstractBasis1d, n) = TensorProductBasis(tuple([b for i=1:n]...))

isreal(b::TensorProductBasis) = isreal(b.bases[1])
isreal{B <: AbstractBasis1d,G,N,T}(::Type{TensorProductBasis{B,G,N,T}}) = isreal(B)


# has_compact_support{B}(::Type{TensorProductBasis{B}}) = has_compact_support(B)
# has_compact_support{B,N}(::Type{TensorProductBasis{B,N}}) = has_compact_support(B)

length(b::TensorProductBasis) = b.ntot

size(b::TensorProductBasis) = b.n
size(b::TensorProductBasis, j) = b.n[j]

stagedfunction eachindex{B,G,N,T}(b::TensorProductBasis{B,G,N,T})
    startargs = fill(1, N)
    stopargs = [:(size(b,$i)) for i=1:N]
    :(CartesianRange(CartesianIndex{$N}($(startargs...)), CartesianIndex{$N}($(stopargs...))))
end

stagedfunction getindex{B,G,N,T}(b::TensorProductBasis{B,G,N,T}, index::CartesianIndex{N})
    :(@nref $N b d->index[d])
end

index_dim{B,G,N,T}(::TensorProductBasis{B,G,N,T}) = N
index_dim{B,G,N,T}(::Type{TensorProductBasis{B,G,N,T}}) = N
index_dim{B <: TensorProductBasis}(::Type{B}) = index_dim(super(B))



natural_grid(b::TensorProductBasis) = b.grid

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


