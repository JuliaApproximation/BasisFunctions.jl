# grid.jl

"AbstractGrid is the supertype of all grids."
abstract type AbstractGrid{T}
end

const AbstractGrid1d{T <: Number} = AbstractGrid{T}
# Todo: remove this one again, it is not sufficiently generic
const AbstractGrid2d{T <: Number} = AbstractGrid{SVector{2,T}}

GridPoint{N,T} = SVector{N,T}

# The element type of a grid is the type returned by getindex.
eltype(::Type{AbstractGrid{T}}) where {T} = T
eltype(::Type{G}) where {G <: AbstractGrid} = eltype(supertype(G))

# The subeltype of a grid is the T in SVector{N,T}
subeltype(::Type{AbstractGrid{T}}) where {T} = subeltype(T)
subeltype(::Type{G}) where {G <: AbstractGrid} = subeltype(supertype(G))
subeltype(g::AbstractGrid) = subeltype(typeof(g))

subeltype(::Type{T}) where {T <: Number} = T
subeltype(::Type{GridPoint{N,T}}) where {N,T} = T

# The dimension of a grid is the dimension of its elements
dimension(grid::AbstractGrid) = dimension(eltype(grid))

size(g::AbstractGrid1d) = (length(g),)
endof(g::AbstractGrid) = length(g)

first(g::AbstractGrid) = g[first(eachindex(g))]
last(g::AbstractGrid) = g[last(eachindex(g))]

has_extension(::AbstractGrid) = false

#support(g::AbstractGrid) = interval(left(g),right(g))

checkbounds(g::AbstractGrid, idx::Int) = (1 <= idx <= length(g) || throw(BoundsError()))

# If the given argument is not an integer, we convert to a linear index
checkbounds(g::AbstractGrid, idxn) = checkbounds(g, linear_index(g, idxn))

# For convenience, catch indexing of 1d grids with CartesianIndex or 1-element tuple
getindex(g::AbstractGrid1d, idx::Union{CartesianIndex{1},Tuple{Int}}) = getindex(g, idx[1])

# Pack a list of integers into a tuple
getindex(g::AbstractGrid, idx1, idx2) = getindex(g, (idx1, idx2))
getindex(g::AbstractGrid, idx1, idx2, idx3) = getindex(g, (idx1, idx2, idx3))
getindex(g::AbstractGrid, idx1, idx2, idx3, idx4) = getindex(g, (idx1, idx2, idx3, idx4))
getindex(g::AbstractGrid, idx1, idx2, idx3, idx4, indices...) = getindex(g, (idx1, idx2, idx3, idx4, indices...))

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



if VERSION < v"0.7-"
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
else
    function Base.iterate(g::AbstractGrid)
        iter = eachindex(g)
        first_item, first_state = iterate(iter)
        (g[first_item], (iter, (first_item, first_state)))
    end

    function Base.iterate(g::AbstractGrid, state)
        iter, iter_tuple = state
        iter_item, iter_state = iter_tuple
        next_tuple = iterate(iter, iter_state)
        if next_tuple != nothing
            next_item, next_state = next_tuple
            (g[next_item], (iter,next_tuple))
        end
    end
end
