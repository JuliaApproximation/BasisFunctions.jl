
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

dimension(b::GridBasis) = Grids.dimension(grid(b))

name(dict::GridBasis) = "Discrete grid basis"

string(dict::GridBasis) = name(dict) * " for coefficient type $(coefficienttype(dict))"
strings(dict::GridBasis) = (string(dict), strings(grid(dict)))


tensorproduct(dicts::GridBasis...) = GridBasis{promote_type(map(coefficienttype,dicts)...)}(cartesianproduct(map(grid, dicts)...))

support(b::GridBasis) = support(grid(b))

# Convenience function: add grid as extra parameter to hastransform
hastransform(s1::Dictionary, s2::GridBasis) =
	hasgrid_transform(s1, s2, grid(s2))
hastransform(s1::GridBasis, s2::Dictionary) =
	hasgrid_transform(s2, s1, grid(s1))
# and provide a default
hasgrid_transform(s1::Dictionary, s2, grid) = false

elements(s::ProductGridBasis) = map(GridBasis{coefficienttype(s)}, elements(grid(s)))
element(s::ProductGridBasis, i) = GridBasis{coefficienttype(s)}(element(grid(s), i))

apply_map(s::GridBasis, map) = GridBasis{coefficienttype(s)}(apply_map(grid(s), map))

sample(s::GridBasis, f) = sample(grid(s), f, coefficienttype(s))

grid_multiplication_operator(a::Function, GB::GridBasis; T=op_eltype(GB)) =
	DiagonalOperator(GB, GB, map(a,grid(GB)); T=T)
grid_multiplication_opearator(a::Function, grid::AbstractGrid; options...) =
	grid_multiplication_operator(a,GridBasis(grid); options...)

native_index(d::ProductGridBasis, idx) = product_native_index(size(d), idx)
ordering(d::ProductGridBasis) = ProductIndexList{dimension(grid(d))}(size(d))


function grid_extension_operator(src::GridBasis, dest::GridBasis, src_grid::IndexSubGrid, dest_grid::AbstractGrid; options...)
    @assert supergrid(src_grid) == dest_grid
    IndexExtensionOperator(src, dest, subindices(src_grid))
end

function grid_restriction_operator(src::GridBasis, dest::GridBasis, src_grid::AbstractGrid, dest_grid::IndexSubGrid; options...)
    @assert supergrid(dest_grid) == src_grid
    IndexRestrictionOperator(src, dest, subindices(dest_grid))
end
