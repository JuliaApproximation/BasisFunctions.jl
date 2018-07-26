# mappedgrid.jl

"""
A MappedGrid consists of a grid and a map. Each grid point of the mapped grid
is the map of the corresponding point of the underlying grid.
"""
struct MappedGrid{G,M,T} <: AbstractGrid{T}
	supergrid	::	G
	map			::	M

	MappedGrid{G,M,T}(supergrid::AbstractGrid{T}, map) where {G,M,T} = new(supergrid, map)
end

const MappedGrid1d{G,M,T<:Number} = MappedGrid{G,M,T}

MappedGrid(grid::AbstractGrid{T}, map::AbstractMap) where {T} =
	MappedGrid{typeof(grid),typeof(map),T}(grid, map)

supergrid(g::MappedGrid) = g.supergrid

mapping(g::MappedGrid) = g.map

mapped_grid(grid::AbstractGrid, map::AbstractMap) = MappedGrid(grid, map)

# avoid multiple mappings
mapped_grid(g::MappedGrid, map::AbstractMap) = MappedGrid(supergrid(g), map∘mapping(g))

# Convenience function, similar to apply_map for Dictionary's
apply_map(grid::AbstractGrid, map::AbstractMap) = mapped_grid(grid, map)

for op in (:length, :size, :eachindex, :indextype)
	@eval $op(g::MappedGrid) = $op(supergrid(g))
end

for op in (:leftendpoint, :rightendpoint, :support)
	@eval $op(g::MappedGrid1d) = applymap(g.map, $op(supergrid(g)))
end

resize(g::MappedGrid, n::Int) = apply_map(resize(supergrid(g), n), mapping(g))

# This is necessary for mapped tensorproductgrids etc.
linear_index(g::MappedGrid, idx) = linear_index(g.supergrid, idx)

native_index(g::MappedGrid, idx) = native_index(g.supergrid, idx)

unsafe_getindex(g::MappedGrid, idx) = applymap(g.map, g.supergrid[idx])


function rescale(g::AbstractGrid1d, a, b)
	m = interval_map(leftendpoint(g), rightendpoint(g), a, b)
	mapped_grid(g, m)
end


# Preserve tensor product structure
function rescale(g::ProductGrid, a::SVector{N}, b::SVector{N}) where {N}
	scaled_grids = [ rescale(element(g, i), a[i], b[i]) for i in 1:N]
	ProductGrid(scaled_grids...)
end
