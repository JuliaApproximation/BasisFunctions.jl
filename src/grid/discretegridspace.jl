# discretegridspace.jl

"""
A DiscreteGridSpace is a discrete basis that can represent a sampled function on a grid.
"""
immutable DiscreteGridSpace{G,N,T} <: FunctionSet{N,T}
	# TODO: remove the G parameter and leave grid untyped.
	# Currently, this clashes with dispatch on Extension in FrameFun's subgrid.jl
	grid		::	G

	DiscreteGridSpace(grid::AbstractGrid{N}) = new(grid)
end

typealias DiscreteGridSpace1d{G,ELT} DiscreteGridSpace{G,1,ELT}

name(s::DiscreteGridSpace) = "A discrete grid space"

DiscreteGridSpace{N,T}(grid::AbstractGrid{N,T}, ELT = T) = DiscreteGridSpace{typeof(grid),N,ELT}(grid)

DiscreteGridSpace(set::FunctionSet) = DiscreteGridSpace(grid(set), eltype(set))

# We choose for the DiscreteGridSpace of a tensor product set to be a tensor
# product of discrete grid spaces, rather than a discrete grid space of a tensor
# product set. This has implications elsewhere, for example in the definitions
# of operators for tensor product sets. With our convention, src and dest of such
# operators both have tensor product structure.
DiscreteGridSpace(grid::TensorProductGrid, ELT = numtype(grid)) =
    tensorproduct( [ DiscreteGridSpace(element(grid, j), ELT) for j in 1:composite_length(grid)]...)

promote_eltype{G,N,T,S}(s::DiscreteGridSpace{G,N,T}, ::Type{S}) = DiscreteGridSpace(s.grid, promote_type(T,S))

resize{G,N,T}(s::DiscreteGridSpace{G,N,T}, n) = DiscreteGridSpace(resize(grid(s), n), T)

grid(s::DiscreteGridSpace) = s.grid

for op in (:length, :size, :left, :right)
    @eval $op(b::DiscreteGridSpace) = $op(grid(b))
end

is_discrete(s::DiscreteGridSpace) = true

# Delegate a map to the underlying grid, but retain the element type of the space
apply_map(s::DiscreteGridSpace, map) = DiscreteGridSpace(apply_map(grid(s), map), eltype(s))
