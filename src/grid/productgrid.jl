# productgrid.jl

"""
A ProductGrid represents the cartesian product of other grids.

`struct ProductGrid{TG,T} <: AbstractGrid{T}`

Parameters:
- TG is a tuple of (grid) types
- T is the element type of the grid
"""
struct ProductGrid{TG,T} <: AbstractGrid{T}
	grids	::	TG
end

# Generic functions for composite types:
elements(grid::ProductGrid) = grid.grids
element(grid::ProductGrid, j::Int) = grid.grids[j]
element(grid::ProductGrid, range::Range) = cartesianproduct(grid.grids[range]...)

# Disallow cartesian products of a single grid
function ProductGrid(grid::AbstractGrid)
	println("Use cartesianproduct instead of ProductGrid.")
	grid
end

function ProductGrid(grids...)
	TG = typeof(grids)
	T1 = Tuple{map(eltype, grids)...}
	T2 = Domains.simplify_product_eltype(T1)
	ProductGrid{typeof(grids),T2}(grids)
end

size(g::ProductGrid) = map(length, g.grids)
size(g::ProductGrid, j::Int) = length(g.grids[j])

ndims(g::ProductGrid, j::Int) = ndims(element(g,j))

length(g::ProductGrid) = prod(size(g))

# left(g::ProductGrid) = SVector(map(left, g.grids))
# left(g::ProductGrid, j) = left(g.grids[j])
#
# right(g::ProductGrid) = SVector(map(right, g.grids))
# right(g::ProductGrid, j) = right(g.grids[j])

# Convert to linear index. If the argument is a tuple of integers, it can be assumed to
# be a multilinear index. Same for CartesianIndex.
linear_index{N}(grid::ProductGrid, i::NTuple{N,Int}) = sub2ind(size(grid), i...)
linear_index(grid::ProductGrid, i::CartesianIndex) = linear_index(grid, i.I)

# If its type is anything else, it may be a tuple of native indices
linear_index(grid::ProductGrid, idxn::Tuple) = linear_index(grid, map(linear_index, elements(grid), idxn))

multilinear_index(grid::ProductGrid, idx::Int) = ind2sub(size(grid), idx)

@generated function eachindex{TG}(g::ProductGrid{TG})
	LEN = tuple_length(TG)
	startargs = fill(1, LEN)
	stopargs = [:(size(g,$i)) for i=1:LEN]
	:(CartesianRange(CartesianIndex{$LEN}($(startargs...)), CartesianIndex{$LEN}($(stopargs...))))
end

unsafe_getindex(grid::ProductGrid, idx::CartesianIndex{2}) =
	unsafe_getindex(grid, idx[1], idx[2])
unsafe_getindex(grid::ProductGrid, idx::CartesianIndex{3}) =
	unsafe_getindex(grid, idx[1], idx[2], idx[3])
unsafe_getindex(grid::ProductGrid, idx::CartesianIndex{4}) =
	unsafe_getindex(grid, idx[1], idx[2], idx[3], idx[4])

unsafe_getindex(g::ProductGrid, index::Tuple) = unsafe_getindex(g, index...)

# For the recursive evaluation of grids, we want to flatten any Vec's
# (Since in the future a single grid may return a vector rather than a number)
# This is achieved with FlatVec below:
FlatVector(x) = SVector(x)
FlatVector(x, y) = SVector(x, y)
FlatVector(x, y, z) = SVector(x, y, z)
FlatVector(x, y, z, t) = SVector(x, y, z, t)

FlatVector(x::Number, y::SVector{2}) = SVector(x, y[1], y[2])
FlatVector(x::Number, y::SVector{2}, z::Number) = SVector(x, y[1], y[2], z)
FlatVector(x::Number, y::SVector{3}) = SVector(x, y[1], y[2], y[3])
FlatVector(x::SVector{2}, y::SVector{2}) = SVector(x[1], x[2], y[1], y[2])
FlatVector(x::SVector{2}, y::Number) = SVector(x[1], x[2], y)
FlatVector(x::SVector{2}, y::Number, z::Number) = SVector(x[1], x[2], y, z)

unsafe_getindex(g::ProductGrid, i1, i2) =
	FlatVector(g.grids[1][i1], g.grids[2][i2])

unsafe_getindex(g::ProductGrid, i1, i2, i3) =
	FlatVector(g.grids[1][i1], g.grids[2][i2], g.grids[3][i3])

unsafe_getindex(g::ProductGrid, i1, i2, i3, i4) =
	FlatVector(g.grids[1][i1], g.grids[2][i2], g.grids[3][i3], g.grids[4][i4])

unsafe_getindex(grid::ProductGrid, idx::Int) = unsafe_getindex(grid, multilinear_index(grid, idx))
