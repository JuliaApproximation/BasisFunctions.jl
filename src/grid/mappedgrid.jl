# mappedgrid.jl

"""
A MappedGrid consists of a grid and a map. Each grid point of the mapped grid
is the map of the corresponding point of the underlying grid.
"""
immutable MappedGrid{G,M,N,T} <: AbstractGrid{N,T}
	grid	::	G
	map		::	M

	MappedGrid(grid::AbstractGrid{N,T}, map) = new(grid, map)
end

typealias MappedGrid1d{G,M,T} MappedGrid{G,M,1,T}

MappedGrid{N,T}(grid::AbstractGrid{N,T}, map::AbstractMap) =
	MappedGrid{typeof(grid),typeof(map),N,T}(grid, map)

mapped_grid(grid::AbstractGrid, map::AbstractMap) = MappedGrid(grid, map)

# avoid multiple mappings
mapped_grid(g::MappedGrid, map::AbstractMap) = MappedGrid(grid(g), map*mapping(g))

# Convenience function, similar to apply_map for FunctionSet's
apply_map(grid::AbstractGrid, map::AbstractMap) = mapped_grid(grid, map)

grid(g::MappedGrid) = g.grid
mapping(g::MappedGrid) = g.map

for op in (:length, :size, :eachindex)
	@eval $op(g::MappedGrid) = $op(grid(g))
end

for op in (:left, :right)
	@eval $op(g::MappedGrid1d) = forward_map(g.map, $op(grid(g)))
end

getindex(g::MappedGrid, idx::Int) = forward_map(g.map, g.grid[idx])


function rescale(g::AbstractGrid1d, a, b)
	m = interval_map(left(g), right(g), a, b)
	mapped_grid(g, m)
end


# Preserve tensor product structure
function rescale{N}(g::TensorProductGrid, a::SVector{N}, b::SVector{N})
	scaled_grids = [ rescale(grid(g,i), a[i], b[i]) for i in 1:N]
	TensorProductGrid(scaled_grids...)
end
