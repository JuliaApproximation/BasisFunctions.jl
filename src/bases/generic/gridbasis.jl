# gridbasis.jl

"""
A `GridBasis` is a discrete basis that is associated with a grid.
"""
immutable GridBasis{G <: AbstractGrid,T} <: DiscreteVectorSpace{T,T}
    grid    ::  G
end

const GridBasis1d{G <: AbstractGrid1d,T} = GridBasis{G,T}
const ProductGridBasis{G <: ProductGrid,T} = GridBasis{G,T}

GridBasis(grid::AbstractGrid) = GridBasis{typeof(grid),typeof(first(eachindex(grid)))}(grid)

gridbasis(grid::AbstractGrid) = GridBasis(grid)

grid(b::GridBasis) = b.grid

for op in (:length, :size)
    @eval $op(b::GridBasis) = $op(grid(b))
end

coefficient_type(b::GridBasis) = float_type(eltype(grid(b)))

name(b::GridBasis) = "a discrete basis associated with a grid"

tensorproduct(dicts::GridBasis...) = GridBasis(cartesianproduct(map(grid, dicts)...))

# Convenience function: add grid as extra parameter to has_transform
has_transform(s1::Dictionary, s2::GridBasis) =
	has_grid_transform(s1, s2, grid(s2))
has_transform(s1::GridBasis, s2::Dictionary) =
	has_grid_transform(s2, s1, grid(s1))
# and provide a default
has_grid_transform(s1::Dictionary, s2, grid) = false

elements(s::ProductGridBasis) = map(GridBasis, elements(grid(s)))
element(s::ProductGridBasis, i) = GridBasis(element(grid(s), i))

apply_map(s::GridBasis, map) = GridBasis(apply_map(grid(s), map))


###############################################
# A DiscreteGridSpace is the span of a GridBasis
###############################################

"""
A DiscreteGridSpace is a discrete basis that can represent a sampled function on a grid.
"""
const DiscreteGridSpace{A,S,T,D <: GridBasis} = Span{A,S,T,D}
const DiscreteGridSpace1d{A,S,T,D <: GridBasis1d} = Span{A,S,T,D}

gridspace(grid::AbstractGrid, ::Type{T} = float(eltype(grid))) where {T} =
    Span(GridBasis(grid), T)

gridspace(s::Span, g::AbstractGrid = grid(s)) = Span(gridbasis(g), coeftype(s))

name(s::DiscreteGridSpace) = "A discrete grid space"

gridbasis(s::DiscreteGridSpace) = dictionary(s)

similar(s::DiscreteGridSpace, grid::AbstractGrid) = Span(grid, coeftype(s))

grid(span::DiscreteGridSpace) = grid(dictionary(span))

# Delegate a map to the underlying grid, but retain the element type of the space
apply_map(s::DiscreteGridSpace, map) = Span(apply_map(dictionary(s), map), coeftype(s))

sample(s::DiscreteGridSpace, f) = sample(grid(s), f, coeftype(s))
