# mappedgrid.jl

# Map a grid 'g' defined on [left(g),right(g)] to the interval [a,b].
immutable LinearMappedGrid{G,T} <: AbstractGrid1d{T}
	grid	::	G
	a		::	T
	b		::	T

	LinearMappedGrid(grid::AbstractGrid1d{T}, a, b) = new(grid, a, b)
end

LinearMappedGrid{T}(g::AbstractGrid1d{T}, a, b) = LinearMappedGrid{typeof(g),T}(g, a, b)

left(g::LinearMappedGrid) = g.a
right(g::LinearMappedGrid) = g.b

grid(g::LinearMappedGrid) = g.grid

length(g::LinearMappedGrid) = length(g.grid)

for op in (:size,:eachindex)
	@eval $op(g::LinearMappedGrid) = $op(grid(g))
end

getindex(g::LinearMappedGrid, idx::Int) = map_linear(getindex(grid(g),idx), left(g), right(g), left(grid(g)), right(grid(g)))


rescale(g::AbstractGrid1d, a, b) = LinearMappedGrid(g, a, b)

# Avoid multiple linear mappings
rescale(g::LinearMappedGrid, a, b) = LinearMappedGrid(grid(g), a, b)


# Preserve tensor product structure
function rescale{N}(g::TensorProductGrid, a::Vec{N}, b::Vec{N})
	scaled_grids = [ rescale(grid(g,i), a[i], b[i]) for i in 1:N]
	TensorProductGrid(scaled_grids...)
end
