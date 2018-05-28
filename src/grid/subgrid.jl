# subgrid.jl

abstract type AbstractSubGrid{T} <: AbstractGrid{T} end

supergrid(g::AbstractSubGrid) = g.supergrid

"""
An IndexSubGrid is a subgrid corresponding to a certain range of indices of the
underlying grid.
"""
struct IndexSubGrid{G,I,T} <: AbstractSubGrid{T}
	supergrid  :: G
	subindices :: I

	function IndexSubGrid{G,I,T}(supergrid::AbstractGrid{T}, subindices) where {G,I,T}
		@assert length(subindices) <= length(supergrid)

		new(supergrid, subindices)
	end
end

IndexSubGrid(grid::AbstractGrid{T}, i) where {T} =
    IndexSubGrid{typeof(grid),typeof(i),T}(grid, i)

supergrid(g::IndexSubGrid) = g.supergrid

subindices(g::IndexSubGrid) = g.subindices

similar_subgrid(g::IndexSubGrid, g2::AbstractGrid) = IndexSubGrid(g2, subindices(g))

length(g::IndexSubGrid) = length(subindices(g))

eachindex(g::IndexSubGrid) = eachindex(subindices(g))

# The speed of this routine is the main reason why supergrid and subindices
# are typed fields, leading to extra type parameters.
unsafe_getindex(g::IndexSubGrid, idx) = unsafe_getindex(g.supergrid, g.subindices[idx])

left(g::IndexSubGrid) = first(g)

right(g::IndexSubGrid) = last(g)

function mask(g::IndexSubGrid)
    mask = zeros(Bool,size(supergrid(g)))
    [mask[i]=true for i in g.subindices]
    mask
end


support(g::IndexSubGrid{G}) where G<:AbstractIntervalGrid = interval(first(g), last(g))



# Check whether element grid[i] (of the underlying grid) is in the indexed subgrid.
is_subindex(i, g::IndexSubGrid) = in(i, subindices(g))

function grid_extension_operator(src::DiscreteGridSpace, dest::DiscreteGridSpace, src_grid::IndexSubGrid, dest_grid::AbstractGrid; options...)
    @assert supergrid(src_grid) == dest_grid
    IndexExtensionOperator(src, dest, subindices(src_grid))
end

function grid_restriction_operator(src::DiscreteGridSpace, dest::DiscreteGridSpace, src_grid::AbstractGrid, dest_grid::IndexSubGrid; options...)
    @assert supergrid(dest_grid) == src_grid
    IndexRestrictionOperator(src, dest, subindices(dest_grid))
end

getindex(grid::AbstractGrid, i::Range) = IndexSubGrid(grid, i)

strings(grid::IndexSubGrid) = ("IndexSubGrid with subindices $(subindices(grid))", (strings(supergrid(grid)),))
