# mappedgrid.jl

"""
A MappedGrid consists of a grid and a map. Each grid point of the mapped grid
is the map of the corresponding point of the underlying grid.
"""
struct MappedGrid{G,M,T} <: AbstractGrid{T}
	grid	::	G
	map		::	M

	MappedGrid{G,M,T}(grid::AbstractGrid{T}, map) where {G,M,T} = new(grid, map)
end

MappedGrid(grid::AbstractGrid{T}, map::AbstractMap) where {T} =
	MappedGrid{typeof(grid),typeof(map),T}(grid, map)

mapped_grid(grid::AbstractGrid, map::AbstractMap) = MappedGrid(grid, map)

# avoid multiple mappings
mapped_grid(g::MappedGrid, map::AbstractMap) = MappedGrid(grid(g), map∘mapping(g))

# Convenience function, similar to apply_map for FunctionSet's
apply_map(grid::AbstractGrid, map::AbstractMap) = mapped_grid(grid, map)

grid(g::MappedGrid) = g.grid
mapping(g::MappedGrid) = g.map

for op in (:length, :size, :eachindex)
	@eval $op(g::MappedGrid) = $op(grid(g))
end

resize(g::MappedGrid, n::Int) = apply_map(resize(grid(g), n), mapping(g))

# This is necessary for mapped tensorproductgrids etc.
linear_index(g::MappedGrid, idx) = linear_index(g.grid, idx)

native_index(g::MappedGrid, idx) = native_index(g.grid, idx)

unsafe_getindex(g::MappedGrid, idx) = applymap(g.map, g.grid[idx])


function rescale(g::AbstractGrid1d, a, b)
	m = interval_map(left(g), right(g), a, b)
	mapped_grid(g, m)
end


# Preserve tensor product structure
function rescale(g::ProductGrid, a::SVector{N}, b::SVector{N}) where {N}
	scaled_grids = [ rescale(grid(g,i), a[i], b[i]) for i in 1:N]
	ProductGrid(scaled_grids...)
end
