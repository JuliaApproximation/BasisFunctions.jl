# grid.jl

"AbstractGrid is the supertype of all grids."
abstract AbstractGrid{N,T}

typealias AbstractGrid1d{T} AbstractGrid{1,T}
typealias AbstractGrid2d{T} AbstractGrid{2,T}
typealias AbstractGrid3d{T} AbstractGrid{3,T}
typealias AbstractGrid4d{T} AbstractGrid{4,T}

typealias Point{N,T} SVector{N,T}

ndims{N,T}(::Type{AbstractGrid{N,T}}) = N
ndims{G <: AbstractGrid}(::Type{G}) = ndims(supertype(G))
ndims{N,T}(::AbstractGrid{N,T}) = N

numtype{N,T}(::AbstractGrid{N,T}) = T
numtype{N,T}(::Type{AbstractGrid{N,T}}) = T
numtype{G <: AbstractGrid}(::Type{G}) = numtype(supertype(G))

# The element type of a grid is the type returned by getindex.
eltype{T}(::Type{AbstractGrid{1,T}}) = T
eltype{N,T}(::Type{AbstractGrid{N,T}}) = Point{N,T}
eltype{G <: AbstractGrid}(::Type{G}) = eltype(supertype(G))

size(g::AbstractGrid1d) = (length(g),)
endof(g::AbstractGrid) = length(g)

first(g::AbstractGrid) = g[first(eachindex(g))]
last(g::AbstractGrid) = g[last(eachindex(g))]

has_extension(::AbstractGrid) = false

support(g::AbstractGrid) = (left(g),right(g))

checkbounds(g::AbstractGrid, idx::Int) = (1 <= idx <= length(g) || throw(BoundsError()))

# If the given argument is not an integer, we convert to a linear index
checkbounds(g::AbstractGrid, idxn) = checkbounds(g, linear_index(g, idxn))

# Catch indexing of 1d grids with CartesianIndex
getindex(g::AbstractGrid1d, idx::CartesianIndex{1}) = getindex(g, idx[1])

# Pack a list of integers into a tuple
getindex(g::AbstractGrid, idx1::Int, idx2::Int, indices::Int...) = getindex(g, (idx1,idx2,indices...))

function getindex(g::AbstractGrid, idx)
	checkbounds(g, idx)
	unsafe_getindex(g, idx)
end

# Default implementation of index iterator: construct a range
eachindex(g::AbstractGrid) = 1:length(g)

# Native and linear indices for grids are like native and linear indices for
# function sets.
native_index(g::AbstractGrid, idx) = idx

linear_index(g::AbstractGrid, idxn) = idxn

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
sample(g::AbstractGrid, f, ELT = numtype(g)) = sample!(zeros(ELT, size(g)), g, f)

# We use a generated function to avoid the overhead of splatting when we
# evaluate f with several arguments
@generated function sample!(result, g::AbstractGrid, f)
	xargs = [:(x[$d]) for d = 1:ndims(g)]
	quote
		for i in eachindex(g)
			x = g[i]
			result[i] = f($(xargs...))
		end
		result
	end
end
