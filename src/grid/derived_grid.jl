
"""
A `DerivedGrid` is a grid that derives from an underlying grid. Any concrete grid
that inherits from DerivedGrid is functionally equivalent to the underlying grid.
However, since it has its own type, it may change some of the functionality.
"""
abstract type DerivedGrid{T,N} <: AbstractGrid{T,N}
end

# We assume that the underlying grid is stored in the supergrid field.
# Override if it isn't.
supergrid(g::DerivedGrid) = g.supergrid

getindex(g::DerivedGrid{1}, i::Int) = g.supergrid[i]

getindex(g::DerivedGrid{T,N}, I::Vararg{Int, N}) where {T,N} = g.supergrid[I...]

size(g::DerivedGrid) = size(g.supergrid)


struct ConcreteDerivedGrid{G,T,N} <: DerivedGrid{T,N}
    supergrid   ::  G

    ConcreteDerivedGrid{G,T,N}(supergrid::AbstractArray{T,N}) where {G,T,N} = new(supergrid)
end

ConcreteDerivedGrid(supergrid::AbstractArray{T,N}) where {T,N} =
    ConcreteDerivedGrid{typeof(supergrid),T,N}(supergrid)
