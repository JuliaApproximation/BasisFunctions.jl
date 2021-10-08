
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

@forward GridBasis.grid size, eachindex

dimension(b::GridBasis) = GridArrays.dimension(grid(b))

show(io::IO, mime::MIME"text/plain", d::GridBasis) = composite_show(io, mime, d)
Display.displaystencil(d::GridBasis{T}) where T = ["GridBasis{$(T)}(", grid(d), ")"]

tensorproduct(dicts::GridBasis...) = GridBasis{promote_type(map(coefficienttype,dicts)...)}(productgrid(map(grid, dicts)...))

support(b::GridBasis) = coverdomain(grid(b))

# Convenience function: add grid as extra parameter to hastransform
hastransform(s1::Dictionary, s2::GridBasis) =
	hasgrid_transform(s1, s2, grid(s2))
hastransform(s1::GridBasis, s2::Dictionary) =
	hasgrid_transform(s2, s1, grid(s1))
# and provide a default
hasgrid_transform(s1::Dictionary, s2, grid) = false

components(s::ProductGridBasis) = map(GridBasis{coefficienttype(s)}, components(grid(s)))
component(s::ProductGridBasis, i) = GridBasis{coefficienttype(s)}(component(grid(s), i))

apply_map(s::GridBasis, map) = GridBasis{coefficienttype(s)}(apply_map(grid(s), map))

sample(s::GridBasis, f) = sample(grid(s), f, coefficienttype(s))

native_index(d::ProductGridBasis, idx) = product_native_index(size(d), idx)
ordering(d::ProductGridBasis) = ProductIndexList{dimension(grid(d))}(size(d))


function gridextension(T, src::GridBasis, dest::GridBasis, src_grid::IndexSubGrid, dest_grid::AbstractGrid; options...)
    @assert supergrid(src_grid) == dest_grid
    IndexExtension{T}(src, dest, subindices(src_grid))
end

function gridrestriction(T, src::GridBasis, dest::GridBasis, src_grid::AbstractGrid, dest_grid::IndexSubGrid; options...)
    @assert supergrid(dest_grid) == src_grid
    IndexRestriction{T}(src, dest, subindices(dest_grid))
end
