# gridbasis.jl

"""
A `GridBasis` is a discrete basis that is associated with a grid.

The domain of the grid basis is the index set of the grid.
"""
immutable GridBasis{G <: AbstractGrid,T} <: DiscreteVectorSpace{LinearIndex,T}
    grid    ::  G
end

const GridBasis1d{G <: AbstractGrid1d,T} = GridBasis{G,T}
const ProductGridBasis{G <: ProductGrid,T} = GridBasis{G,T}

# We don't know the expected coefficient type of a grid basis in general.
# If the codomain type is Int (the default), then we try to guess a coefficient
# type from the element type. If the codomain type is anything but Int, we choose
# that.
coefficient_type(gb::GridBasis) = _coefficient_type(gb, codomaintype(gb))
_coefficient_type(gb, ::Type{Int}) = subeltype(eltype(gb))
_coefficient_type(gb, ::Type{T}) where {T} = T

# We don't know the codomain type either. The safest bet is Int.
# As a user, it is best to provide T.
GridBasis(grid::AbstractGrid, T = Int) =
    GridBasis{typeof(grid),T}(grid)

gridbasis(grid::AbstractGrid) = GridBasis(grid)
gridbasis(grid::AbstractGrid, T) = GridBasis(grid, T)

grid(b::GridBasis) = b.grid

for op in (:length, :size)
    @eval $op(b::GridBasis) = $op(grid(b))
end

dimension(b::GridBasis) = dimension(grid(b))

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

gridspace(grid::AbstractGrid, T = subeltype(eltype(grid))) = Span(GridBasis(grid, T))

gridspace(s::Span, g::AbstractGrid = grid(s)) = Span(gridbasis(g, dict_codomaintype(s)), coeftype(s))

name(s::DiscreteGridSpace) = "A discrete grid space"

gridbasis(s::DiscreteGridSpace) = dictionary(s)

similar(s::DiscreteGridSpace, grid::AbstractGrid) = Span(grid, coeftype(s))

grid(span::DiscreteGridSpace) = grid(dictionary(span))

# Delegate a map to the underlying grid, but retain the element type of the space
apply_map(s::DiscreteGridSpace, map) = Span(apply_map(dictionary(s), map), coeftype(s))

sample(s::DiscreteGridSpace, f) = sample(grid(s), f, coeftype(s))
