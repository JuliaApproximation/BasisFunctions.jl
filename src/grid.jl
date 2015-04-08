# grid.jl


abstract AbstractGrid{N,T}

typealias AbstractGrid1d{T} AbstractGrid{1,T}
typealias AbstractGrid2d{T} AbstractGrid{2,T}
typealias AbstractGrid3d{T} AbstractGrid{3,T}
typealias AbstractGrid4d{T} AbstractGrid{4,T}

dim{N,T}(::AbstractGrid{N,T}) = N
dim{N,T}(::Type{AbstractGrid{N,T}}) = N
dim{G <: AbstractGrid}(::Type{G}) = dim(super(G))

eltype{N,T}(::AbstractGrid{N,T}) = T
eltype{N,T}(::Type{AbstractGrid{N,T}}) = T
eltype{G <: AbstractGrid}(::Type{G}) = eltype(super(G))


# iterate over grid points
start(g::AbstractGrid) = 1
done(g::AbstractGrid, i::Int) = (length(g) < i)
next(g::AbstractGrid, i::Int) = (g[i], i+1)

getindex!(g::AbstractGrid1d, x, i::Int) = (x[1] = g[i])

checkbounds(g::AbstractGrid, idx::Int) = (1 <= idx <= length(g) || throw(BoundsError()))

eachindex(g::AbstractGrid) = 1:length(g)

# Default dimension of the index is 1
index_dim(::AbstractGrid) = 1
index_dim{N,T}(::Type{AbstractGrid{N,T}}) = 1
index_dim{G <: AbstractGrid}(::Type{G}) = 1

size(g::AbstractGrid1d) = (length(g),)

support(g::AbstractGrid) = (left(g),right(g))


immutable TensorProductGrid{G <: AbstractGrid1d,N,T} <: AbstractGrid{N,T}
	grids	::	NTuple{N,G}
	n		::	NTuple{N,Int}
	ntot	::	Int

	TensorProductGrid(grids::NTuple) = new(grids, map(g -> length(g), grids), prod(map(g->length(g), grids)))
end

TensorProductGrid{G <: AbstractGrid1d,N}(grids::NTuple{N,G}) = TensorProductGrid{G,N,eltype(grids[1])}(grids)

tensorproduct(g::AbstractGrid1d, n) = TensorProductGrid(tuple([g for i=1:n]...))

checkbounds(g::AbstractGrid, idx::Int) = (1 <= idx <= length(g) || throw(BoundsError()))

stagedfunction eachindex{G,N,T}(g::TensorProductGrid{G,N,T})
    startargs = fill(1, N)
    stopargs = [:(size(g,$i)) for i=1:N]
    :(CartesianRange(CartesianIndex{$N}($(startargs...)), CartesianIndex{$N}($(stopargs...))))
end

# Default dimension of the index is 1
index_dim{G,N,T}(::TensorProductGrid{G,N,T}) = N
index_dim{G,N,T}(::Type{TensorProductGrid{G,N,T}}) = N
index_dim{G <: TensorProductGrid}(::Type{G}) = index_dim(super(G))


size(g::TensorProductGrid) = g.n
size(g::TensorProductGrid, j) = g.n[j]

left(g::TensorProductGrid) = map(t -> left(t), g.grids)
left(g::TensorProductGrid, j) = left(g.grids[j])

right(g::TensorProductGrid) = map(t -> right(t), g.grids)
right(g::TensorProductGrid, j) = right(g.grids[j])

range(g::TensorProductGrid, j::Int) = range(g.grids[j])

length(g::TensorProductGrid) = g.ntot

ind2sub(g::TensorProductGrid, idx::Int) = ind2sub(size(g), idx)

sub2ind(g::TensorProductGrid, idx...) = sub2ind(size(g), idx...)


# Getindex allocates memory
function getindex{G,N}(g::TensorProductGrid{G,N}, idx...)
	x = Array(eltype(g),N)
	getindex!(g, x, idx...)
	x
end


getindex!{G,N}(g::TensorProductGrid{G,N}, x, idx::Int) = getindex!(g, x, ind2sub(g, idx)...)

function getindex!{G}(g::TensorProductGrid{G,1}, x, i1::Int)
	x[1] = g.grids[1][i1]
	nothing
end

function getindex!{G}(g::TensorProductGrid{G,2}, x, i1::Int, i2::Int)
	x[1] = g.grids[1][i1]
	x[2] = g.grids[2][i2] 
	nothing
end

function getindex!{G}(g::TensorProductGrid{G,3}, x, i1::Int, i2::Int)
	x[1] = g.grids[1][i1]
	x[2] = g.grids[2][i2] 
	x[3] = g.grids[3][i3] 
	nothing
end

function getindex!{G}(g::TensorProductGrid{G,4}, x, i1::Int, i2::Int)
	x[1] = g.grids[1][i1]
	x[2] = g.grids[2][i2] 
	x[3] = g.grids[3][i3] 
	x[4] = g.grids[4][i4] 
	nothing
end


stagedfunction getindex{G,N}(g::TensorProductGrid{G,N}, index::CartesianIndex{N})
    :(@nref $N g d->index[d])
end

stagedfunction getindex!{G,N}(g::TensorProductGrid{G,N}, x, index::CartesianIndex{N})
    :(@ncall $N getindex! g x d->index[d])
end



abstract AbstractIntervalGrid{T} <: AbstractGrid1d{T}


abstract AbstractEquispacedGrid{T} <: AbstractIntervalGrid{T}


immutable EquispacedGrid{T} <: AbstractEquispacedGrid{T}
	n	::	Int
	a	::	T
	b	::	T
	h	::	T

	EquispacedGrid(n, a, b) = new(n, a, b, (b-a)/n)
end

EquispacedGrid{T}(n, a::T = -1.0, b::T = 1.0) = EquispacedGrid{T}(n, a, b)


# A periodic equispaced grid is an equispaced grid that omits the right endpoint.
immutable PeriodicEquispacedGrid{T} <: AbstractEquispacedGrid{T}
	n	::	Int
	a	::	T
	b	::	T
	h	::	T

	PeriodicEquispacedGrid(n, a, b) = new(n, a, b, (b-a)/n)
end

PeriodicEquispacedGrid{T}(n, a::T = -1.0, b::T = 1.0) = PeriodicEquispacedGrid{T}(n, a, b)


left(g::AbstractEquispacedGrid) = g.a

right(g::AbstractEquispacedGrid) = g.b

stepsize(g::AbstractEquispacedGrid) = g.h

length(g::EquispacedGrid) = g.n+1

length(g::PeriodicEquispacedGrid) = g.n

range(g::AbstractEquispacedGrid) = range(left(g), stepsize(g), length(g))

function getindex(g::AbstractEquispacedGrid, i)
	checkbounds(g, i)
	unsafe_getindex(g, i)
end

unsafe_getindex(g::AbstractEquispacedGrid, i) = g.a + (i-1)*g.h



