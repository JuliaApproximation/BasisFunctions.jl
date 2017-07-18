# gridset.jl

"""
A `GridSet` is a discrete function set that is associated with a grid.
"""
immutable GridSet{G <: AbstractGrid,T} <: DiscreteVectorSpace{T}
    grid    ::  G
end

const GridSet1d{G <: AbstractGrid1d,T} = GridSet{G,T}

GridSet(grid::AbstractGrid) = GridSet{typeof(grid),typeof(first(eachindex(grid)))}(grid)

grid(set::GridSet) = set.grid

for op in (:length, :size, :domaintype)
    @eval $op(set::GridSet) = $op(grid(set))
end

coefficient_type(set::GridSet) = float_type(eltype(grid(set)))

name(set::GridSet) = "a discrete set associated with a grid"

tensorproduct(set1::GridSet, set2::GridSet) = GridSet(tensorproduct(grid(set1), grid(set2)))


###############################################
# A DiscreteGridSpace is the span of a GridSet
###############################################

"""
A DiscreteGridSpace is a discrete basis that can represent a sampled function on a grid.
"""
const DiscreteGridSpace{S <: GridSet,A} = Span{S,A}
const DiscreteGridSpace1d{S <: GridSet1d,A} = Span{S,A}

DiscreteGridSpace(grid::AbstractGrid, ::Type{T} = float_type(eltype(grid))) where {T} =
    Span(GridSet(grid), T)

name(s::DiscreteGridSpace) = "A discrete grid space"

gridset(dgs::DiscreteGridSpace) = set(dsg)

similar(dgs::DiscreteGridSpace, grid::AbstractGrid) = DiscreteGridSpace(grid, coeftype(dgs))

grid(span::DiscreteGridSpace) = grid(set(span))

# Convenience function: add grid as extra parameter to has_transform
has_transform(s::FunctionSet, dgs::DiscreteGridSpace) =
	has_grid_transform(s, dgs, grid(dgs))
# and provide a default
has_grid_transform(s::FunctionSet, dgs, grid) = false

# Delegate a map to the underlying grid, but retain the element type of the space
apply_map(dgs::DiscreteGridSpace, map) = DiscreteGridSpace(apply_map(grid(dgs), map), coeftype(s))

# Make a DiscreteGridSpace with the same eltype as the given function set
gridspace(s::FunctionSet, g = grid(s)) = DiscreteGridSpace(g, coefficient_type(s))

sample(s::DiscreteGridSpace, f) = sample(grid(s), f, coeftype(s))
