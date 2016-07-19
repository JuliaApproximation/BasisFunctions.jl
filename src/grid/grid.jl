# grid.jl

"AbstractGrid is the supertype of all grids."
abstract AbstractGrid{N,T}

typealias AbstractGrid1d{T} AbstractGrid{1,T}
typealias AbstractGrid2d{T} AbstractGrid{2,T}
typealias AbstractGrid3d{T} AbstractGrid{3,T}
typealias AbstractGrid4d{T} AbstractGrid{4,T}

ndims{N,T}(::Type{AbstractGrid{N,T}}) = N
ndims{G <: AbstractGrid}(::Type{G}) = ndims(supertype(G))
ndims{N,T}(::AbstractGrid{N,T}) = N

numtype{N,T}(::AbstractGrid{N,T}) = T
numtype{N,T}(::Type{AbstractGrid{N,T}}) = T
numtype{G <: AbstractGrid}(::Type{G}) = numtype(supertype(G))

# The element type of a grid is the type returned by getindex.
eltype{T}(::Type{AbstractGrid{1,T}}) = T
eltype{N,T}(::Type{AbstractGrid{N,T}}) = Vec{N,T}
eltype{G <: AbstractGrid}(::Type{G}) = eltype(supertype(G))

# Default dimension of the index is 1
index_dim{N,T}(::Type{AbstractGrid{N,T}}) = 1
index_dim{G <: AbstractGrid}(::Type{G}) = 1
index_dim(g::AbstractGrid) = index_dim(typeof(g))

size(g::AbstractGrid1d) = (length(g),)

support(g::AbstractGrid) = (left(g),right(g))



checkbounds(g::AbstractGrid, idx::Int) = (1 <= idx <= length(g) || throw(BoundsError()))

function getindex(g::AbstractGrid, idx)
	checkbounds(g, idx)
	unsafe_getindex(g, idx)
end

# Default implementation of index iterator: construct a range
eachindex(g::AbstractGrid) = 1:length(g)


# Grid iteration:
#	for x in grid
#		do stuff...
#	end
# Implemented by start, next and done.
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

"Sample the function f on the given grid."
sample(g::AbstractGrid, f::Function, ELT = numtype(g)) = sample!(zeros(ELT, size(g)), g, f)

@generated function sample!(result, g::AbstractGrid, f::Function)
	xargs = [:(x[$d]) for d = 1:ndims(g)]
	quote
		for i in eachindex(g)
			x = g[i]
			result[i] = f($(xargs...))
		end
		result
	end
end


include("tensorproductgrid.jl")
include("mappedgrid.jl")
include("intervalgrids.jl")
