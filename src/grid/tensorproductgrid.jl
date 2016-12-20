# tensorproductgrid.jl

"""
A TensorProductGrid represents the tensor product of other grids.

immutable TensorProductGrid{TG,N,T} <: AbstractGrid{N,T}

Parameters:
- Parameter TG is a tuple of (grid) types.
- Parametes N and T are the total dimension and numeric type of this grid.
"""
immutable TensorProductGrid{TG,N,T} <: AbstractGrid{N,T}
	grids	::	TG
end

# Generic functions for composite types:
elements(grid::TensorProductGrid) = grid.grids
element(grid::TensorProductGrid, j::Int) = grid.grids[j]
element(grid::TensorProductGrid, range::Range) = tensorproduct(grid.grids[range]...)
composite_length(grid::TensorProductGrid) = length(elements(grid))

# Disallow tensor products of a single grid
function TensorProductGrid(grid::AbstractGrid)
	println("Use tensorproduct instead of TensorProductGrid.")
	grid
end

TensorProductGrid(grids...) = TensorProductGrid{typeof(grids),sum(map(ndims, grids)),numtype(grids[1])}(grids)

size(g::TensorProductGrid) = map(length, g.grids)
size(g::TensorProductGrid, j::Int) = length(g.grids[j])

ndims(g::TensorProductGrid, j::Int) = ndims(element(g,j))

length(g::TensorProductGrid) = prod(size(g))

left(g::TensorProductGrid) = SVector(map(left, g.grids))
left(g::TensorProductGrid, j) = left(g.grids[j])

right(g::TensorProductGrid) = SVector(map(right, g.grids))
right(g::TensorProductGrid, j) = right(g.grids[j])

# Convert to linear index. If the argument is a tuple of integers, it can be assumed to
# be a multilinear index. Same for CartesianIndex.
linear_index{N}(grid::TensorProductGrid, i::NTuple{N,Int}) = sub2ind(size(grid), i...)
linear_index(grid::TensorProductGrid, i::CartesianIndex) = linear_index(grid, i.I)

# If its type is anything else, it may be a tuple of native indices
linear_index(grid::TensorProductGrid, idxn::Tuple) = linear_index(grid, map(linear_index, elements(grid), idxn))

multilinear_index(grid::TensorProductGrid, idx::Int) = ind2sub(size(grid), idx)

@generated function eachindex{TG}(g::TensorProductGrid{TG})
	LEN = tuple_length(TG)
	startargs = fill(1, LEN)
	stopargs = [:(size(g,$i)) for i=1:LEN]
	:(CartesianRange(CartesianIndex{$LEN}($(startargs...)), CartesianIndex{$LEN}($(stopargs...))))
end

unsafe_getindex(grid::TensorProductGrid, idx::CartesianIndex{2}) =
	unsafe_getindex(grid, idx[1], idx[2])
unsafe_getindex(grid::TensorProductGrid, idx::CartesianIndex{3}) =
	unsafe_getindex(grid, idx[1], idx[2], idx[3])
unsafe_getindex(grid::TensorProductGrid, idx::CartesianIndex{4}) =
	unsafe_getindex(grid, idx[1], idx[2], idx[3], idx[4])

unsafe_getindex(g::TensorProductGrid, index::Tuple) = unsafe_getindex(g, index...)

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

unsafe_getindex(g::TensorProductGrid, i1, i2) =
	FlatVector(g.grids[1][i1], g.grids[2][i2])

unsafe_getindex(g::TensorProductGrid, i1, i2, i3) =
	FlatVector(g.grids[1][i1], g.grids[2][i2], g.grids[3][i3])

unsafe_getindex(g::TensorProductGrid, i1, i2, i3, i4) =
	FlatVector(g.grids[1][i1], g.grids[2][i2], g.grids[3][i3], g.grids[4][i4])

unsafe_getindex(grid::TensorProductGrid, idx::Int) = unsafe_getindex(grid, multilinear_index(grid, idx))
