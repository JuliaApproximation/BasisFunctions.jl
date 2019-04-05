
"""
A MappedGrid consists of a grid and a map. Each grid point of the mapped grid
is the map of the corresponding point of the underlying grid.
"""
struct MappedGrid{G,M,T,N} <: AbstractGrid{T,N}
	supergrid	::	G
	map			::	M

	MappedGrid{G,M,T,N}(supergrid::AbstractGrid{T,N}, map) where {G,M,T,N} = new(supergrid, map)
end

const MappedGrid1d{G,M,T<:Number,N} = MappedGrid{G,M,T,N}

MappedGrid(grid::AbstractGrid{T,N}, map::AbstractMap) where {T,N} =
	MappedGrid{typeof(grid),typeof(map),T,N}(grid, map)

name(grid::MappedGrid) = "Mapped grid"

supergrid(g::MappedGrid) = g.supergrid

mapping(g::MappedGrid) = g.map

mapped_grid(grid::AbstractGrid, map::AbstractMap) = MappedGrid(grid, map)

# avoid multiple mappings
mapped_grid(g::MappedGrid, map::AbstractMap) = MappedGrid(supergrid(g), map∘mapping(g))

# Convenience function, similar to apply_map for Dictionary's
apply_map(grid::AbstractGrid, map::AbstractMap) = mapped_grid(grid, map)

for op in (:length, :size, :eachindex, :indextype, :isperiodic)
	@eval $op(g::MappedGrid) = $op(supergrid(g))
end

for op in (:leftendpoint, :rightendpoint, :support)
	@eval $op(g::MappedGrid1d) = applymap(g.map, $op(supergrid(g)))
end

resize(g::MappedGrid, n::Int) = apply_map(resize(supergrid(g), n), mapping(g))

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


## Printing

hasstencil(grid::MappedGrid) = true
stencilarray(grid::MappedGrid) = [ mapping(grid), "(", supergrid(grid), ")" ]
