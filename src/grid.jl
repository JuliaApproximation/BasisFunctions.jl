# grid.jl

abstract AbstractGrid{N,T}

typealias AbstractGrid1d{T} AbstractGrid{1,T}
typealias AbstractGrid2d{T} AbstractGrid{2,T}
typealias AbstractGrid3d{T} AbstractGrid{3,T}
typealias AbstractGrid4d{T} AbstractGrid{4,T}

dim{N,T}(::AbstractGrid{N,T}) = N
dim{N,T}(::Type{AbstractGrid{N,T}}) = N
dim{G <: AbstractGrid}(::Type{G}) = dim(super(G))

numtype{N,T}(::AbstractGrid{N,T}) = T
numtype{N,T}(::Type{AbstractGrid{N,T}}) = T
numtype{G <: AbstractGrid}(::Type{G}) = numtype(super(G))

eltype{N,T}(::AbstractGrid{N,T}) = T
eltype{N,T}(::Type{AbstractGrid{N,T}}) = T
eltype{G <: AbstractGrid}(::Type{G}) = eltype(super(G))

# Default dimension of the index is 1
index_dim(::AbstractGrid) = 1
index_dim{N,T}(::Type{AbstractGrid{N,T}}) = 1
index_dim{G <: AbstractGrid}(::Type{G}) = 1

size(g::AbstractGrid1d) = (length(g),)

support(g::AbstractGrid) = (left(g),right(g))



# Getindex allocates memory because it has to return an array (for now).
# General implementation for abstract grids: allocate memory and call getindex!
function getindex{N,T}(g::AbstractGrid{N,T}, idx...)
	x = Array(T,N)
	getindex!(g, x, idx...)
	x
end

# getindex! is a bit silly in 1D, but provide it anyway because it could be called from general code
#getindex!(g::AbstractGrid1d, x, i::Int) = (x[1] = g[i])

checkbounds(g::AbstractGrid, idx::Int) = (1 <= idx <= length(g) || throw(BoundsError()))

# Default implementation of index iterator: construct a range
eachindex(g::AbstractGrid) = 1:length(g)


# We provide two ways of iterating over the elements of a grid.
#
# First approach:
#	for x in grid
#		do stuff...
#	end
# This allocates memory for each point in the grid.
#
# Second approach:
#	for x in eachelement(grid)
#		do stuff...
#	end
# This attempts to reuse the memory for x while iterating.

# First approach follows:

function start(g::AbstractGrid)
	iter = eachindex(g)
	(iter, start(iter))
end

function next(g::AbstractGrid, state)
	iter = state[1]
	iter_state = state[2]
	idx,iter_newstate = next(iter,iter_state)
	(g[idx], (iter,iter_newstate))
end

done(g::AbstractGrid, state) = done(state[1], state[2])


# For the second approach we need a type to store reusable memory for x.
immutable GridIterator{N,T,G <: AbstractGrid,ITER}
	grid		::	G
	griditer	::	ITER
	x			::	Array{T,1}
end

function GridIterator{N,T}(grid::AbstractGrid{N,T})
	iter = eachindex(grid)
	x = zeros(T,N)
	GridIterator{N,T,typeof(grid),typeof(iter)}(grid, iter, x)
end

eachelement(grid::AbstractGrid) = GridIterator(grid)

start(iter::GridIterator) = start(iter.griditer)

function next(iter::GridIterator, state)
	(i,state) = next(iter.griditer, state)
	getindex!(iter.grid, iter.x, i)
	(iter.x, state)
end

function next(iter::GridIterator{1}, state)
	(i,state) = next(iter.griditer, state)
	x = getindex(iter.grid, i)
	(x, state)
end

done(iter::GridIterator, state) = done(iter.griditer, state)



# A TensorProductGrid represents the tensor product of other grids.
# Parameter TG is a tuple of (grid) types.
# Parameter GN is a tuple of the dimensions of each of the grids.
# Parameter LEN is the length of TG and GN (the index dimension).
# Parametes N and T are the total dimension and numeric type of this grid.
immutable TensorProductGrid{TG,GN,LEN,N,T} <: AbstractGrid{N,T}
	grids	::	TG

	TensorProductGrid(grids::Tuple) = new(grids)
end

TensorProductGrid(grids...) = TensorProductGrid{typeof(grids),map(dim,grids),length(grids),sum(map(dim, grids)),numtype(grids[1])}(grids)

tensorproduct(g::AbstractGrid, n) = TensorProductGrid([g for i=1:n]...)

index_dim{TG,GN,LEN,N,T}(::TensorProductGrid{TG,GN,LEN,N,T}) = LEN
index_dim{TG,GN,LEN,N,T}(::Type{TensorProductGrid{TG,GN,LEN,N,T}}) = LEN
index_dim{G <: TensorProductGrid}(::Type{G}) = index_dim(super(G))

size(g::TensorProductGrid) = map(length, g.grids)
size(g::TensorProductGrid, j::Int) = length(g.grids[j])

dim{TG,GN}(g::TensorProductGrid{TG,GN}, j::Int) = GN[j]

length(g::TensorProductGrid) = prod(size(g))

grids(g::TensorProductGrid) = g.grids
grid(g::TensorProductGrid, j::Int) = g.grids[j]

left(g::TensorProductGrid) = map(left, g.grids)
left(g::TensorProductGrid, j) = left(g.grids[j])

right(g::TensorProductGrid) = map(right, g.grids)
right(g::TensorProductGrid, j) = right(g.grids[j])


@generated function eachindex{TG,GN,LEN}(g::TensorProductGrid{TG,GN,LEN})
    startargs = fill(1, LEN)
    stopargs = [:(size(g,$i)) for i=1:LEN]
    :(CartesianRange(CartesianIndex{$LEN}($(startargs...)), CartesianIndex{$LEN}($(stopargs...))))
end

@generated function getindex{TG,GN,LEN}(g::TensorProductGrid{TG,GN,LEN}, index::CartesianIndex{LEN})
    :(@nref $LEN g d->index[d])
end

ind2sub(g::TensorProductGrid, idx::Int) = ind2sub(size(g), idx)
sub2ind(G::TensorProductGrid, idx...) = sub2ind(size(g), idx...)

getindex!(g::TensorProductGrid, x, idx::Int) = getindex!(g, x, ind2sub(g,idx))

getindex!(g::TensorProductGrid, x, idxt::Int...) = getindex!(g, x, idxt)

function getindex!{TG,GN,LEN}(g::TensorProductGrid{TG,GN,LEN}, x, idx::Union(CartesianIndex{LEN},NTuple{LEN,Int}))
	l = 0
    for i = 1:LEN
    	z = grid(g, i)[idx[i]]	# FIX: this allocates memory if GN[i] > 1
    	for j = 1:GN[i]
    		l += 1
    		x[l] = z[j]
    	end
    end
end


# Use the Latex \otimes operator for constructing a tensor product grid
⊗(g1::AbstractGrid, g2::AbstractGrid) = TensorProductGrid(g1, g2)
⊗(g::AbstractGrid...) = TensorProductGrid(g...)




# An AbstractIntervalGrid is a grid that is defined on an interval, i.e. it is connected.
abstract AbstractIntervalGrid{T} <: AbstractGrid1d{T}

# Some default implementations for interval grids follow

left(g::AbstractIntervalGrid) = g.a

right(g::AbstractIntervalGrid) = g.b

length(g::AbstractIntervalGrid) = g.n



# An equispaced grid has equispaced points, and therefore it has a stepsize.
abstract AbstractEquispacedGrid{T} <: AbstractIntervalGrid{T}

range(g::AbstractEquispacedGrid) = range(left(g), stepsize(g), length(g))

function getindex(g::AbstractEquispacedGrid, i)
	checkbounds(g, i)
	unsafe_getindex(g, i)
end

unsafe_getindex(g::AbstractEquispacedGrid, i) = g.a + (i-1)*stepsize(g)


immutable EquispacedGrid{T} <: AbstractEquispacedGrid{T}
	n	::	Int
	a	::	T
	b	::	T
	# h	::	T	# a possible optimization is to precompute and store the stepsize

	EquispacedGrid(n, a, b) = new(n, a, b)
end

# Parameter n is the total number of points in the equispaced grid.
EquispacedGrid{T <: FloatingPoint}(n, a::T = -1.0, b::T = 1.0) = EquispacedGrid{T}(n, a, b)

# Promote the numeric type of a and b to a floating point type.
function EquispacedGrid{S <: Number}(n, a::S, b::S)
	T = promote_type(S, typeof((b-a)/n))
	EquispacedGrid(n, T(a), T(b))
end

stepsize(g::EquispacedGrid) = (g.b-g.a)/(g.n-1)


# A periodic equispaced grid is an equispaced grid that omits the right endpoint.
immutable PeriodicEquispacedGrid{T} <: AbstractEquispacedGrid{T}
	n	::	Int
	a	::	T
	b	::	T

	PeriodicEquispacedGrid(n, a, b) = new(n, a, b)
end

# Parameter n is the total number of points in the periodic equispaced grid.
PeriodicEquispacedGrid{T <: FloatingPoint}(n, a::T = -1.0, b::T = 1.0) = PeriodicEquispacedGrid{T}(n, a, b)

function PeriodicEquispacedGrid{S <: Number}(n, a::S, b::S)
	T = promote_type(S, typeof((b-a)/n))
	PeriodicEquispacedGrid(n, T(a), T(b))
end


stepsize(g::PeriodicEquispacedGrid) = (g.b-g.a)/g.n




