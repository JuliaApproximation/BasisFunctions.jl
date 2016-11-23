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

promote_eltype{G,N,T,S}(s::DiscreteGridSpace{G,N,T}, ::Type{S}) = DiscreteGridSpace(s.grid, promote_type(T,S))

resize{G,N,T}(s::DiscreteGridSpace{G,N,T}, n) = DiscreteGridSpace(resize(grid(s), n), T)

grid(s::DiscreteGridSpace) = s.grid

for op in (:length, :size, :left, :right)
    @eval $op(b::DiscreteGridSpace) = $op(grid(b))
end

is_discrete(s::DiscreteGridSpace) = true

# Delegate a map to the underlying grid, but retain the element type of the space
apply_map(s::DiscreteGridSpace, map) = DiscreteGridSpace(apply_map(grid(s), map), eltype(s))

function tensorproduct(s::DiscreteGridSpace...)
	ELT = promote_type(map(eltype, s)...)
	DiscreteGridSpace(tensorproduct(map(grid,s)...), ELT)
end

# Implement composite interface for spaces with a tensorproductgrid
element(s::DiscreteGridSpace, i) = _element(s, i, grid(s))
_element(s::DiscreteGridSpace, i, grid) = DiscreteGridSpace(element(grid, i), eltype(s))

composite_length(s::DiscreteGridSpace) = composite_length(grid(s))

elements(s::DiscreteGridSpace) = map(t->DiscreteGridSpace(t, eltype(s)), elements(grid(s)))
