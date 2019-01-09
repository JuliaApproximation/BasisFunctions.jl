
"""
A `ProductGrid` represents the cartesian product of other grids.

`struct ProductGrid{TG,T,N} <: AbstractGrid{T,N}`

Parameters:
- TG is a tuple of (grid) types
- T is the element type of the grid
- N is the dimension of the grid layout
"""
struct ProductGrid{TG,T,N} <: AbstractGrid{T,N}
	grids	::	TG
end

# Generic functions for composite types:
elements(grid::ProductGrid) = grid.grids
element(grid::ProductGrid, j::Int) = grid.grids[j]
element(grid::ProductGrid, range::AbstractRange) = cartesianproduct(grid.grids[range]...)

function ProductGrid(grids...)
	TG = typeof(grids)
	T1 = Tuple{map(eltype, grids)...}
	T2 = DomainSets.simplify_product_eltype(T1)
	ProductGrid{typeof(grids),T2,length(grids)}(grids)
end

size(g::ProductGrid) = map(length, g.grids)
size(g::ProductGrid, j::Int) = length(g.grids[j])

leftendpoint(g::ProductGrid) = SVector(map(leftendpoint, g.grids))
leftendpoint(g::ProductGrid, j) = leftendpoint(g.grids[j])

rightendpoint(g::ProductGrid) = SVector(map(rightendpoint, g.grids))
rightendpoint(g::ProductGrid, j) = rightendpoint(g.grids[j])

getindex(g::ProductGrid{TG,T,N}, I::Vararg{Int,N}) where {TG,T,N} =
	convert(T, map(getindex, g.grids, I))

similargrid(grid::ProductGrid, ::Type{T}, dims...) where T = ProductGrid([similargrid(g, eltype(T), dims[i]) for (i,g) in enumerate(elements(grid))]...)
