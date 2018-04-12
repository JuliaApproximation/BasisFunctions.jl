# intervalgrids.jl

"An AbstractIntervalGrid is a grid that is defined on an interval, i.e. it is connected."
abstract type AbstractIntervalGrid{T} <: AbstractGrid1d{T}
end

# Some default implementations for interval grids follow
leftendpoint(g::AbstractIntervalGrid) = g.a
rightendpoint(g::AbstractIntervalGrid) = g.b
length(g::AbstractIntervalGrid) = g.n

# Perhaps we should add a stepsize field, for better efficiency?
# Now the stepsize is recomputed with every call to getindex.

"An equispaced grid has equispaced points, and therefore it has a stepsize."
abstract type AbstractEquispacedGrid{T} <: AbstractIntervalGrid{T}
end

range(g::AbstractEquispacedGrid) = range(leftendpoint(g), stepsize(g), length(g))

unsafe_getindex(g::AbstractEquispacedGrid, i) = g.a + (i-1)*stepsize(g)

similar_grid(g::AbstractEquispacedGrid, a, b) =
    similar_grid(g, a, b, promote_type(eltype(g), typeof((b-a)/length(g))))

# Equispaced grids already support rescaling - avoid the construction of a LinearMappedGrid,
# but make sure to retain the type of the original grid.
rescale(g::AbstractEquispacedGrid, a, b) = similar_grid(g, a, b)

mapped_grid(g::AbstractEquispacedGrid, map::AffineMap) =
    similar_grid(g, applymap(map, leftendpoint(g)), applymap(map, rightendpoint(g)))

"""
An equispaced grid with n points on an interval [a,b], including the endpoints.
It has stepsize (b-a)/(n-1).
"""
struct EquispacedGrid{T} <: AbstractEquispacedGrid{T}
    n   ::  Int
    a   ::  T
    b   ::  T

    EquispacedGrid{T}(n::Int, a = -one(T), b = one(T)) where {T} = (@assert a < b; new(n, a, b))
end

EquispacedGrid(n, ::Type{T} = Float64) where {T} = EquispacedGrid{T}(n)

EquispacedGrid(n, a, b, ::Type{T} = typeof((b-a)/n)) where {T} = EquispacedGrid{T}(n, a, b)

similar_grid(g::EquispacedGrid, a, b, ::Type{T} = eltype(g)) where {T} = EquispacedGrid{T}(length(g), a, b)

has_extension(::EquispacedGrid) = true

resize(g::EquispacedGrid, n::Int) = EquispacedGrid(n, g.a, g.b)

extend(g::EquispacedGrid, factor::Int) = resize(g, factor*g.n-1)

stepsize(g::EquispacedGrid) = (g.b-g.a)/(g.n-1)

# Support conversion from a LinSpace in julia Base
# (What about more general ranges?)
EquispacedGrid(x::LinSpace) = EquispacedGrid(length(x), first(x), last(x))


"""
A periodic equispaced grid is an equispaced grid that omits the right endpoint.
It has stepsize (b-a)/n.
"""
struct PeriodicEquispacedGrid{T} <: AbstractEquispacedGrid{T}
    n   ::  Int
    a   ::  T
    b   ::  T

    PeriodicEquispacedGrid{T}(n, a = -one(T), b = one(T)) where {T} = (@assert a < b; new(n, a, b))
end

PeriodicEquispacedGrid(n::Int, ::Type{T} = Float64) where {T} = PeriodicEquispacedGrid{T}(n)

PeriodicEquispacedGrid(n::Int, a, b, ::Type{T} = typeof((b-a)/n)) where {T} = PeriodicEquispacedGrid{T}(n, a, b)

similar_grid(g::PeriodicEquispacedGrid, a, b, T = eltype(g)) = PeriodicEquispacedGrid{T}(length(g), a, b)

has_extension(::PeriodicEquispacedGrid) = true

resize(g::PeriodicEquispacedGrid{T}, n::Int) where {T} = PeriodicEquispacedGrid(n, g.a, g.b)

extend(g::PeriodicEquispacedGrid{T}, factor::Int) where {T} = resize(g, factor*g.n)

stepsize(g::PeriodicEquispacedGrid) = (g.b-g.a)/g.n

# We need this basic definition, otherwise equality does not seem to hold when T is BigFloat...
==(g1::PeriodicEquispacedGrid, g2::PeriodicEquispacedGrid) =
    (g1.n == g2.n) && (g1.a == g2.a) && (g1.b==g2.b)


"""
A dyadic periodic equispaced grid is an equispaced grid that omits the right
endpoint and has length `n = 2^l`.
It has stepsize `(b-a)/n`.
"""
struct DyadicPeriodicEquispacedGrid{T} <: AbstractEquispacedGrid{T}
    l   ::  Int
    a   ::  T
    b   ::  T

    DyadicPeriodicEquispacedGrid{T}(l, a = zero(T), b = one(T)) where {T} = (@assert a < b; new(l, a, b))
end

dyadic_length(g::DyadicPeriodicEquispacedGrid) = g.l

length(g::DyadicPeriodicEquispacedGrid) = 1<<dyadic_length(g)

DyadicPeriodicEquispacedGrid(l, ::Type{T} = Float64) where {T} = DyadicPeriodicEquispacedGrid{T}(l)

DyadicPeriodicEquispacedGrid(l, a, b, ::Type{T} = typeof((b-a)/l)) where {T} = DyadicPeriodicEquispacedGrid{T}(l, a, b)

PeriodicEquispacedGrid(g::DyadicPeriodicEquispacedGrid{T}) where {T} = PeriodicEquispacedGrid{T}(length(g), g.a, g.b)

similar_grid(g::DyadicPeriodicEquispacedGrid, a, b, T) = DyadicPeriodicEquispacedGrid{T}(g.l, a, b)

has_extension(::DyadicPeriodicEquispacedGrid) = true

resize(g::DyadicPeriodicEquispacedGrid, n::Int) = DyadicPeriodicEquispacedGrid(n, g.a, g.b)

extend(g::DyadicPeriodicEquispacedGrid, factor::Int) = resize(g, factor*g.l)

stepsize(g::DyadicPeriodicEquispacedGrid) = (g.b-g.a)/length(g)

# We need this basic definition, otherwise equality does not seem to hold when T is BigFloat...
==(g1::DyadicPeriodicEquispacedGrid, g2::DyadicPeriodicEquispacedGrid) =
    (g1.l == g2.l) && (g1.a == g2.a) && (g1.b==g2.b)
==(g1::DyadicPeriodicEquispacedGrid, g2::PeriodicEquispacedGrid) =
    (length(g1) == length(g2)) && (g1.a == g2.a) && (g1.b==g2.b)

"""
A MidpointEquispaced grid is an equispaced grid with grid points in the centers of the equispaced
subintervals. In other words, this is a DCT-II grid.
It has stepsize `(b-a)/n`.
"""
struct MidpointEquispacedGrid{T} <: AbstractEquispacedGrid{T}
    n   ::  Int
    a   ::  T
    b   ::  T

    MidpointEquispacedGrid{T}(n, a = -one(T), b = one(T)) where {T} = (@assert a < b; new(n, a, b))
end

MidpointEquispacedGrid(n, ::Type{T} = Float64) where {T} = MidpointEquispacedGrid{T}(n)

MidpointEquispacedGrid(n, a, b, ::Type{T} = typeof((b-a)/n)) where {T} = MidpointEquispacedGrid{T}(n, a, b)

similar_grid(g::MidpointEquispacedGrid, a, b, T) = MidpointEquispacedGrid{T}(length(g), a, b)

resize(g::MidpointEquispacedGrid, n::Int) = MidpointEquispacedGrid(n, g.a, g.b)

unsafe_getindex{T}(g::MidpointEquispacedGrid{T}, i) = g.a + (i-one(T)/2)*stepsize(g)

stepsize(g::MidpointEquispacedGrid) = (g.b-g.a)/g.n


struct ChebyshevNodeGrid{T} <: AbstractIntervalGrid{T}
    n   ::  Int
end

const ChebyshevGrid = ChebyshevNodeGrid
const ChebyshevPoints = ChebyshevNodeGrid

ChebyshevNodeGrid(n::Int, ::Type{T} = Float64) where {T} = ChebyshevNodeGrid{T}(n)


leftendpoint(g::ChebyshevNodeGrid{T}) where {T} = -one(T)
rightendpoint(g::ChebyshevNodeGrid{T}) where {T} = one(T)

# The minus sign is added to avoid having to flip the inputs to the dct. More elegant fix required.
unsafe_getindex(g::ChebyshevNodeGrid{T}, i) where {T} = T(-1)*cos((i-1/2) * T(pi) / (g.n) )


struct ChebyshevExtremaGrid{T} <: AbstractIntervalGrid{T}
    n   ::  Int
end

ChebyshevPointsOfTheSecondKind = ChebyshevExtremaGrid

ChebyshevExtremaGrid(n::Int, ::Type{T} = Float64) where {T} = ChebyshevExtremaGrid{T}(n)

leftendpoint(g::ChebyshevExtremaGrid{T}) where {T} = -one(T)
rightendpoint(g::ChebyshevExtremaGrid{T}) where {T} = one(T)

# Likewise, the minus sign is added to avoid having to flip the inputs to the dct. More elegant fix required.
unsafe_getindex(g::ChebyshevExtremaGrid{T}, i) where {T} = i == 0 ? T(0) : cos((i-1)*T(pi) / (g.n-1) )
