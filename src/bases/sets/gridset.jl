# gridset.jl

"""
A `GridSet` is a discrete function set that is associated with a grid.
"""
immutable GridSet{G <: AbstractGrid,T} <: DiscreteVectorSpace{T}
    grid    ::  G
end

const GridSet1d{G <: AbstractGrid1d,T} = GridSet{G,T}

GridSet(grid::AbstractGrid) = GridSet{typeof(grid),typeof(first(eachindex(grid)))}(grid)

gridset(grid::AbstractGrid) = GridSet(grid)

grid(set::GridSet) = set.grid

for op in (:length, :size, :domaintype)
    @eval $op(set::GridSet) = $op(grid(set))
end

coefficient_type(set::GridSet) = float_type(eltype(grid(set)))

name(set::GridSet) = "a discrete set associated with a grid"

tensorproduct(set1::GridSet, set2::GridSet) = GridSet(tensorproduct(grid(set1), grid(set2)))

# Convenience function: add grid as extra parameter to has_transform
has_transform(s1::FunctionSet, s2::GridSet) =
	has_grid_transform(s1, s2, grid(s2))
has_transform(s1::GridSet, s2::FunctionSet) =
	has_grid_transform(s2, s1, grid(s1))
# and provide a default
has_grid_transform(s1::FunctionSet, s2, grid) = false



###############################################
# A DiscreteGridSpace is the span of a GridSet
###############################################

"""
A DiscreteGridSpace is a discrete basis that can represent a sampled function on a grid.
"""
const DiscreteGridSpace{A,S <: GridSet} = Span{A,S}
const DiscreteGridSpace1d{A,S <: GridSet1d} = Span{A,S}

gridspace(grid::AbstractGrid, ::Type{T} = float_type(eltype(grid))) where {T} =
    Span(GridSet(grid), T)

gridspace(s::Span, g::AbstractGrid = grid(s)) = Span(gridset(g), coeftype(s))

name(s::DiscreteGridSpace) = "A discrete grid space"

gridset(dgs::DiscreteGridSpace) = set(dsg)

similar(dgs::DiscreteGridSpace, grid::AbstractGrid) = DiscreteGridSpace(grid, coeftype(dgs))

grid(span::DiscreteGridSpace) = grid(set(span))

# Delegate a map to the underlying grid, but retain the element type of the space
apply_map(dgs::DiscreteGridSpace, map) = Span(apply_map(grid(dgs), map), coeftype(s))

sample(s::DiscreteGridSpace, f) = sample(grid(s), f, coeftype(s))
