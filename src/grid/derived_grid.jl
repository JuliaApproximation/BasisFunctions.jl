# derived_grid.jl

"""
A DerivedGrid is a grid that derives from an underlying grid. Any concrete grid
that inherits from DerivedGrid is functionally equivalent to the underlying grid.
However, since it is its own type, it may change some of the functionality.
"""
abstract DerivedGrid{N,T} <: AbstractGrid{N,T}

# We assume that the underlying grid is stored in the supergrid field.
# Override if it isn't.
supergrid(g::DerivedGrid) = g.supergrid

for method in (:first, :last, :eachindex, :size, :length, :endof)
    @eval $method(g::DerivedGrid) = $method(supergrid(g))
end

for method in (:support,)
    @eval $method(g::DerivedGrid) = $method(supergrid(g))
end

for method in (:getindex, :checkbounds, :native_index, :linear_index)
    @eval $method(g::DerivedGrid, idx) = $method(supergrid(g), idx)
end


immutable ConcreteDerivedGrid{G,N,T} <: DerivedGrid{N,T}
    supergrid   ::  G

    ConcreteDerivedGrid(supergrid::AbstractGrid{N,T}) = new(supergrid)
end
