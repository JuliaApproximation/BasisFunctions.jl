
"An AbstractIntervalGrid is a grid that is defined on an interval, i.e. it is connected."
abstract type AbstractIntervalGrid{T} <: AbstractGrid1d{T,1}
end

instantiate(::Type{T}, n::Int, ::Type{ELT})  where {T<:AbstractIntervalGrid,ELT} = T(n,ELT(0),ELT(1))

# Some default implementations for interval grids follow
leftendpoint(g::AbstractIntervalGrid) = g.a
rightendpoint(g::AbstractIntervalGrid) = g.b
support(g::AbstractIntervalGrid) = Interval(leftendpoint(g), rightendpoint(g))

size(g::AbstractIntervalGrid) = (g.n,)



# Perhaps we should add a stepsize field, for better efficiency?
# Now the stepsize is recomputed with every call to getindex.
# Alternatively, we should just wrap around a native Julia range

"An equispaced grid has equispaced points, and therefore it has a stepsize."
abstract type AbstractEquispacedGrid{T} <: AbstractIntervalGrid{T}
end

range(g::AbstractEquispacedGrid) = range(leftendpoint(g), stepsize(g), length(g))

unsafe_getindex(g::AbstractEquispacedGrid, i) = g.a + (i-1)*stepsize(g)

similar_equispacedgrid(g::AbstractEquispacedGrid, a, b) =
    similar_equispacedgrid(g, a, b, promote_type(eltype(g), typeof((b-a)/length(g))))

# Equispaced grids already support rescaling - avoid the construction of a LinearMappedGrid,
# but make sure to retain the type of the original grid.
rescale(g::AbstractEquispacedGrid, a, b) = similar_equispacedgrid(g, a, b)

mapped_grid(g::AbstractEquispacedGrid, map::AffineMap) =
    similar_equispacedgrid(g, applymap(map, leftendpoint(g)), applymap(map, rightendpoint(g)))

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

EquispacedGrid(n::Int) = EquispacedGrid{Float64}(n)

EquispacedGrid(n::Int, a, b) = EquispacedGrid{typeof((b-a)/n)}(n, a, b)

EquispacedGrid(n::Int, d::AbstractInterval{T}) where {T} = EquispacedGrid{T}(n, infimum(d), supremum(d))

name(g::EquispacedGrid) = "Equispaced grid"

similargrid(g::EquispacedGrid, ::Type{T}, n::Int) where {T} = EquispacedGrid{T}(n, convert(T, g.a), convert(T, g.b))

similar_equispacedgrid(g::EquispacedGrid, a, b, ::Type{T} = eltype(g)) where {T} = EquispacedGrid{T}(length(g), a, b)

hasextension(::EquispacedGrid) = true

extend(g::EquispacedGrid, factor::Int) = resize(g, factor*g.n-1)


stepsize(g::EquispacedGrid) = (g.b-g.a)/(g.n-1)

# Support conversion from a LinSpace in julia Base
# (What about more general ranges?)
# TODO: update, LinRange no longer exists
convert(::Type{EquispacedGrid}, x::LinRange) = EquispacedGrid(length(x), first(x), last(x))


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

PeriodicEquispacedGrid(n::Int) = PeriodicEquispacedGrid{Float64}(n)

PeriodicEquispacedGrid(n::Int, a, b) = PeriodicEquispacedGrid{typeof((b-a)/n)}(n, a, b)

PeriodicEquispacedGrid(n::Int, d::AbstractInterval{T}) where {T} = PeriodicEquispacedGrid{T}(n, infimum(d), supremum(d))

name(g::PeriodicEquispacedGrid) = "Periodic equispaced grid"

similar_equispacedgrid(g::PeriodicEquispacedGrid, a, b, T = eltype(g)) = PeriodicEquispacedGrid{T}(length(g), a, b)

similargrid(g::PeriodicEquispacedGrid, ::Type{T}, n::Int) where {T} = PeriodicEquispacedGrid{T}(n, convert(T, g.a), convert(T, g.b))

stepsize(g::PeriodicEquispacedGrid) = (g.b-g.a)/g.n

# We need this basic definition, otherwise equality does not seem to hold when T is BigFloat...
==(g1::PeriodicEquispacedGrid, g2::PeriodicEquispacedGrid) =
    (g1.n == g2.n) && (g1.a == g2.a) && (g1.b==g2.b)

hasextension(::PeriodicEquispacedGrid) = true

extend(g::PeriodicEquispacedGrid, factor::Int) = resize(g, factor*g.n)


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

MidpointEquispacedGrid(n::Int) = MidpointEquispacedGrid{Float64}(n)

MidpointEquispacedGrid(n::Int, a, b) = MidpointEquispacedGrid{typeof((b-a)/n)}(n, a, b)

MidpointEquispacedGrid(n::Int, d::AbstractInterval{T}) where {T} = MidpointEquispacedGrid{T}(n, infimum(d), supremum(d))

name(g::MidpointEquispacedGrid) = "Equispaced midpoints grid"

similargrid(g::MidpointEquispacedGrid, ::Type{T}, n::Int) where {T} = MidpointEquispacedGrid{T}(n, convert(T, g.a), convert(T, g.b))

similar_equispacedgrid(g::MidpointEquispacedGrid, a, b, ::Type{T} = eltype(g)) where {T} = MidpointEquispacedGrid{T}(length(g), a, b)

unsafe_getindex(g::MidpointEquispacedGrid{T}, i) where {T} = g.a + (i-one(T)/2)*stepsize(g)

stepsize(g::MidpointEquispacedGrid) = (g.b-g.a)/g.n


"A Fourier grid is a periodic equispaced grid on the interval [0,1]."
struct FourierGrid{T} <: AbstractEquispacedGrid{T}
    n   ::  Int
end

FourierGrid(n::Int) = FourierGrid{Float64}(n)

name(g::FourierGrid) = "Periodic Fourier grid"

leftendpoint(g::FourierGrid{T}) where {T} = zero(T)
rightendpoint(g::FourierGrid{T}) where {T} = one(T)
support(g::FourierGrid{T}) where {T} = UnitInterval{T}()

similargrid(g::FourierGrid, ::Type{T}, n::Int) where {T} =
    FourierGrid{T}(n)

stepsize(g::FourierGrid{T}) where {T} = one(T)/length(g)

unsafe_getindex(g::FourierGrid, i) = (i-1)*stepsize(g)

has_extension(::FourierGrid) = true

extend(g::FourierGrid, factor::Int) = resize(g, factor*g.n)

mapped_grid(g::FourierGrid, map::AffineMap) = MappedGrid(g, map)

function rescale(g::FourierGrid, a, b)
	m = interval_map(leftendpoint(g), rightendpoint(g), a, b)
	mapped_grid(g, m)
end


struct ChebyshevNodes{T} <: AbstractIntervalGrid{T}
    n   ::  Int
end

const ChebyshevGrid = ChebyshevNodes
const ChebyshevPoints = ChebyshevNodes

ChebyshevNodes(n::Int) = ChebyshevNodes{Float64}(n)
ChebyshevNodes(n::Int, a, b) = rescale(ChebyshevNodes{typeof((b-a)/n)}(n), a, b)

similargrid(g::ChebyshevNodes, ::Type{T}, n::Int) where {T} = ChebyshevNodes{T}(n)

leftendpoint(g::ChebyshevNodes{T}) where {T} = -one(T)
rightendpoint(g::ChebyshevNodes{T}) where {T} = one(T)

# The minus sign is added to avoid having to flip the inputs to the dct. More elegant fix required.
unsafe_getindex(g::ChebyshevNodes{T}, i) where {T} = T(-1)*cos((i-T(1)/2) * T(pi) / (g.n) )

name(g::ChebyshevNodes) = "Chebyshev nodes"


struct ChebyshevExtremae{T} <: AbstractIntervalGrid{T}
    n   ::  Int
end

ChebyshevPointsOfTheSecondKind = ChebyshevExtremae

ChebyshevExtremae(n::Int) = ChebyshevExtremae{Float64}(n)
ChebyshevExtremae(n::Int, a, b) = rescale(ChebyshevExtremae{typeof((b-a)/n)}(n), a, b)

similargrid(g::ChebyshevExtremae, ::Type{T}, n::Int) where {T} = ChebyshevExtremae{T}(n)

leftendpoint(g::ChebyshevExtremae{T}) where {T} = -one(T)
rightendpoint(g::ChebyshevExtremae{T}) where {T} = one(T)

# TODO: flip the values so that they are sorted
unsafe_getindex(g::ChebyshevExtremae{T}, i) where {T} = i == 0 ? T(0) : cos((i-1)*T(pi) / (g.n-1) )

name(g::ChebyshevExtremae) = "Chebyshev extremae"

string(g::AbstractIntervalGrid) = name(g) * " of length $(length(g)) on [$(leftendpoint(g)), $(rightendpoint(g))], ELT = $(eltype(g))"
