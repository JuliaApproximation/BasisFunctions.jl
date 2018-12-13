
"""
A `GridBasis` is a discrete basis that is associated with a grid.

The domain of the grid basis is the index set of the grid.
"""
struct GridBasis{T,G <: AbstractGrid} <: DiscreteDictionary{LinearIndex,T}
    grid    ::  G
end

const GridBasis1d{T,G <: AbstractGrid1d} = GridBasis{T,G}
const ProductGridBasis{T,G <: ProductGrid} = GridBasis{T,G}

coefficienttype(gb::GridBasis{T,G}) where {T,G} = T

# If not specified, we guess the codomaintype of the grid from the element
# type of its  elements.
GridBasis(grid::AbstractGrid) = GridBasis{subeltype(grid)}(grid)

GridBasis{T}(grid::AbstractGrid) where {T} = GridBasis{T,typeof(grid)}(grid)

# gridbasis(grid::AbstractGrid) = GridBasis(grid)
# gridbasis(grid::AbstractGrid, T) = GridBasis{T}(grid)
GridBasis(d::Dictionary, g::AbstractGrid = interpolation_grid(d)) = GridBasis{coefficienttype(d)}(g)

grid(b::GridBasis) = b.grid

for op in (:size, :eachindex)
    @eval $op(b::GridBasis) = $op(grid(b))
end

dimension(b::GridBasis) = dimension(grid(b))

name(b::GridBasis) = "a discrete basis associated with a grid"

tensorproduct(dicts::GridBasis...) = GridBasis{promote_type(map(coefficienttype,dicts)...)}(cartesianproduct(map(grid, dicts)...))

support(b::GridBasis) = support(grid(b))

# Convenience function: add grid as extra parameter to has_transform
has_transform(s1::Dictionary, s2::GridBasis) =
	has_grid_transform(s1, s2, grid(s2))
has_transform(s1::GridBasis, s2::Dictionary) =
	has_grid_transform(s2, s1, grid(s1))
# and provide a default
has_grid_transform(s1::Dictionary, s2, grid) = false

elements(s::ProductGridBasis) = map(GridBasis{coefficienttype(s)}, elements(grid(s)))
element(s::ProductGridBasis, i) = GridBasis{coefficienttype(s)}(element(grid(s), i))

apply_map(s::GridBasis, map) = GridBasis{coefficienttype(s)}(apply_map(grid(s), map))

sample(s::GridBasis, f) = sample(grid(s), f, coefficienttype(s))

grid_multiplication_operator(a::Function, GB::GridBasis) =
	DiagonalOperator(GB, GB, map(a,grid(GB)))
grid_multiplication_opearator(a::Function, grid::AbstractGrid) =
	grid_multiplication_operator(a,GridBasis(grid))

native_index(d::ProductGridBasis, idx) = product_native_index(size(d), idx)
ordering(d::ProductGridBasis) = ProductIndexList{dimension(grid(d))}(size(d))
