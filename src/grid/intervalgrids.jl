# intervalgrids.jl


# An AbstractIntervalGrid is a grid that is defined on an interval, i.e. it is connected.
abstract AbstractIntervalGrid{T} <: AbstractGrid1d{T}

# Some default implementations for interval grids follow

left(g::AbstractIntervalGrid) = g.a

right(g::AbstractIntervalGrid) = g.b

length(g::AbstractIntervalGrid) = g.n

index_dim{G <: AbstractIntervalGrid}(::Type{G}) = 1

# An equispaced grid has equispaced points, and therefore it has a stepsize.
abstract AbstractEquispacedGrid{T} <: AbstractIntervalGrid{T}

range(g::AbstractEquispacedGrid) = range(left(g), stepsize(g), length(g))

unsafe_getindex(g::AbstractEquispacedGrid, i) = g.a + (i-1)*stepsize(g)


"""
An equispaced grid with n points on an interval [a,b].
"""
immutable EquispacedGrid{T} <: AbstractEquispacedGrid{T}
    n   ::  Int
    a   ::  T
    b   ::  T

    EquispacedGrid(n, a = -one(T), b = one(T)) = (@assert a < b; new(n, a, b))
end

EquispacedGrid{T}(n, ::Type{T} = Float64) = EquispacedGrid{T}(n)

EquispacedGrid{T}(n, a, b, ::Type{T} = typeof((b-a)/n)) = EquispacedGrid{T}(n, a, b)

stepsize(g::EquispacedGrid) = (g.b-g.a)/(g.n-1)

# Equispaced grids already support rescaling - avoid the construction of a LinearMappedGrid,
# but make sure to retain the type of the original grid.
rescale{T}(g::EquispacedGrid{T}, a, b) = EquispacedGrid(length(g), a, b, T)


# A periodic equispaced grid is an equispaced grid that omits the right endpoint.
immutable PeriodicEquispacedGrid{T} <: AbstractEquispacedGrid{T}
    n   ::  Int
    a   ::  T
    b   ::  T

    PeriodicEquispacedGrid(n, a = -one(T), b = one(T)) = (@assert a < b; new(n, a, b))
end

PeriodicEquispacedGrid{T}(n, ::Type{T} = Float64) = PeriodicEquispacedGrid{T}(n)

PeriodicEquispacedGrid{T}(n, a, b, ::Type{T} = typeof((b-a)/n)) = PeriodicEquispacedGrid{T}(n, a, b)


stepsize(g::PeriodicEquispacedGrid) = (g.b-g.a)/g.n


rescale{T}(g::PeriodicEquispacedGrid{T}, a, b) = PeriodicEquispacedGrid(length(g), a, b, T)


# A MidpointEquispaced grid is an equispaced grid with grid points in the centers of the equispaced
# subintervals. In other words, this is a DCT-II grid.
immutable MidpointEquispacedGrid{T} <: AbstractEquispacedGrid{T}
    n   ::  Int
    a   ::  T
    b   ::  T

    MidpointEquispacedGrid(n, a = -one(T), b = one(T)) = (@assert a < b; new(n, a, b))
end

MidpointEquispacedGrid{T}(n, ::Type{T} = Float64) = MidpointEquispacedGrid{T}(n)

MidpointEquispacedGrid{T}(n, a, b, ::Type{T} = typeof((b-a)/n)) = MidpointEquispacedGrid{T}(n, a, b)

unsafe_getindex{T}(g::MidpointEquispacedGrid{T}, i) = g.a + (i-one(T)/2)*stepsize(g)

stepsize(g::MidpointEquispacedGrid) = (g.b-g.a)/g.n

rescale{T}(g::MidpointEquispacedGrid{T}, a, b) = MidpointEquispacedGrid(length(g), a, b, T)



immutable ChebyshevIIGrid{T} <: AbstractIntervalGrid{T}
    n   ::  Int
end

typealias ChebyshevGrid ChebyshevIIGrid

ChebyshevIIGrid{T}(n::Int, ::Type{T} = Float64) = ChebyshevIIGrid{T}(n)


left{T}(g::ChebyshevIIGrid{T}) = -one(T)
right{T}(g::ChebyshevIIGrid{T}) = one(T)

# The minus sign is added to avoid having to flip the inputs to the dct. More elegant fix required.
unsafe_getindex{T}(g::ChebyshevIIGrid{T}, i) = T(-1)*cos((i-1/2) * T(pi) / (g.n) )
