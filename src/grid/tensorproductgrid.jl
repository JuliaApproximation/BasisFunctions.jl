# tensorproductgrid.jl

"""
A TensorProductGrid represents the tensor product of other grids.

immutable TensorProductGrid{TG,GN,LEN,N,T} <: AbstractGrid{N,T}

Parameters:
- Parameter TG is a tuple of (grid) types.
- Parameter GN is a tuple of the dimensions of each of the grids.
- Parameter LEN is the length of TG and GN (the index dimension).
- Parametes N and T are the total dimension and numeric type of this grid.
"""
immutable TensorProductGrid{TG,GN,LEN,N,T} <: AbstractGrid{N,T}
	grids	::	TG

	TensorProductGrid(grids::Tuple) = new(grids)
end

# Generic functions for composite types:
elements(grid::TensorProductGrid) = grid.grids
element(grid::TensorProductGrid, j::Int) = grid.grids[j]
composite_length(grid::TensorProductGrid) = length(elements(grid))

# Disallow tensor products of a single grid
function TensorProductGrid(grid::AbstractGrid)
	println("Use tensorproduct instead of TensorProductGrid.")
	grid
end

TensorProductGrid(grids...) = TensorProductGrid{typeof(grids),map(dim,grids),length(grids),sum(map(dim, grids)),numtype(grids[1])}(grids)


index_dim{TG,GN,LEN,N,T}(::Type{TensorProductGrid{TG,GN,LEN,N,T}}) = LEN

size(g::TensorProductGrid) = map(length, g.grids)
size(g::TensorProductGrid, j::Int) = length(g.grids[j])

dim{TG,GN}(g::TensorProductGrid{TG,GN}, j::Int) = GN[j]

length(g::TensorProductGrid) = prod(size(g))

left(g::TensorProductGrid) = Vec(map(left, g.grids)...)
left(g::TensorProductGrid, j) = left(g.grids[j])

right(g::TensorProductGrid) = Vec(map(right, g.grids)...)
right(g::TensorProductGrid, j) = right(g.grids[j])


@generated function eachindex{TG,GN,LEN}(g::TensorProductGrid{TG,GN,LEN})
    startargs = fill(1, LEN)
    stopargs = [:(size(g,$i)) for i=1:LEN]
    :(CartesianRange(CartesianIndex{$LEN}($(startargs...)), CartesianIndex{$LEN}($(stopargs...))))
end

@generated function getindex{TG,GN,LEN}(g::TensorProductGrid{TG,GN,LEN}, index::CartesianIndex{LEN})
    :(@nref $LEN g d->index[d])
end

# This first set of routines applies when LEN ≠ N
# TODO: optimize with generated functions to remove all splatting.
getindex{TG,GN,N,T}(g::TensorProductGrid{TG,GN,1,N,T}, i1::Int) = Vec{N,T}(g.grids[1][i1]...)
getindex{TG,GN,N,T}(g::TensorProductGrid{TG,GN,2,N,T}, i1::Int, i2) =
	Vec{N,T}(g.grids[1][i1]..., g.grids[2][i2]...)
getindex{TG,GN,N,T}(g::TensorProductGrid{TG,GN,3,N,T}, i1::Int, i2, i3) =
	Vec{N,T}(g.grids[1][i1]..., g.grids[2][i2]..., g.grids[3][i3]...)
getindex{TG,GN,N,T}(g::TensorProductGrid{TG,GN,4,N,T}, i1::Int, i2, i3, i4) =
	Vec{N,T}(g.grids[1][i1]..., g.grids[2][i2]..., g.grids[3][i3]..., g.grids[4][i4]...)

# These routines apply when LEN = N
getindex{TG,GN,T}(g::TensorProductGrid{TG,GN,2,2,T}, i1::Int, i2) =
	Vec{2,T}(g.grids[1][i1], g.grids[2][i2])
getindex{TG,GN,T}(g::TensorProductGrid{TG,GN,3,3,T}, i1::Int, i2, i3) =
	Vec{3,T}(g.grids[1][i1], g.grids[2][i2], g.grids[3][i3])
getindex{TG,GN,T}(g::TensorProductGrid{TG,GN,4,4,T}, i1::Int, i2, i3, i4) =
	Vec{4,T}(g.grids[1][i1], g.grids[2][i2], g.grids[3][i3], g.grids[4][i4])

ind2sub(g::TensorProductGrid, idx::Int) = ind2sub(size(g), idx)
sub2ind(G::TensorProductGrid, idx...) = sub2ind(size(g), idx...)
