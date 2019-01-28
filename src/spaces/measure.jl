
"""
The abstract supertype of all measures.
"""
abstract type Measure{T}
end

domaintype(m::Measure{T}) where {T} = T

weight(m::Measure{T}, x::T) where {T} = weight1(m, x)

weight(m::Measure{T}, x) where {T} = weight1(m, convert(T, x))

function weight1(m::Measure{T}, x) where {T}
    x ∈ support(m) ? unsafe_weight(m, x) : zero(T)
end

weightfunction(m::Measure) = x->weight(m, x)
unsafe_weightfunction(m::Measure) = x->unsafe_weight(m, x)

codomaintype(m::Measure{T}) where {T} = subeltype(T)


"A measure on a general domain with a general weight function `dσ = w(x) dx`."
struct GenericWeightMeasure{T} <: Measure{T}
    support          ::  Domain{T}
    weightfunction
end

unsafe_weight(m::GenericWeightMeasure{T}, x) where {T} = weight.weightfunction(x)

support(m::GenericWeightMeasure, x) = m.support


"Supertype of all Lebesgue measures."
abstract type LebesgueMeasure{T} <: Measure{T}
end

unsafe_weight(m::LebesgueMeasure{T}, x) where {T} = one(T)


"Lebesgue measure supported on a general domain."
struct GenericLebesgueMeasure{T} <: LebesgueMeasure{T}
    support  ::  Domain{T}
end

support(m::GenericLebesgueMeasure) = m.support


"The Legendre measure is the Lebesgue measure on `[-1,1]`."
struct LegendreMeasure{T} <: LebesgueMeasure{T}
end

support(m::LegendreMeasure{T}) where {T} = ChebyshevInterval{T}()


"The Fourier measure is the Lebesgue measure on `[0,1]`."
struct FourierMeasure{T} <: LebesgueMeasure{T}
end

support(m::FourierMeasure{T}) where {T} = UnitInterval{T}()

lebesguemeasure(domain::UnitInterval{T}) where {T} = FourierMeasure{T}()
lebesguemeasure(domain::ChebyshevInterval{T}) where {T} = LegendreMeasure{T}()
lebesguemeasure(domain::Domain{T}) where {T} = GenericLebesgueMeasure{T}(domain)


"""
The `Chebyshev` or `ChebyshevT` measure is the measure on `[-1,1]` with the
Chebyshev weight `w(x) = 1/√(1-x^2)`.
"""
struct ChebyshevTMeasure{T} <: Measure{T}
end

const ChebyshevMeasure = ChebyshevTMeasure

support(m::ChebyshevTMeasure{T}) where {T} = ChebyshevInterval{T}()

unsafe_weight(m::ChebyshevTMeasure, x) = 1/sqrt(1-x^2)


"""
The ChebyshevU measure is the measure on `[-1,1]` with the Chebyshev weight
of the second kind `w(x) = √(1-x^2).`
"""
struct ChebyshevUMeasure{T} <: Measure{T}
end

support(m::ChebyshevUMeasure{T}) where {T} = ChebyshevInterval{T}()

unsafe_weight(m::ChebyshevUMeasure, x) = sqrt(1-x^2)


"""
The Jacobi measure is the measure on `[-1,1]` with the classical Jacobi weight
`w(x) = (1-x)^α (1+x)^β`.
"""
struct JacobiMeasure{T} <: Measure{T}
    α   ::  T
    β   ::  T
end

support(m::JacobiMeasure{T}) where {T} = ChebyshevInterval{T}()

unsafe_weight(m::JacobiMeasure, x) = (1-x)^m.α * (1+x)^m.β


"""
The Laguerre measure is the measure on `[0,∞)` with the classical generalized
Laguerre weight `w(x) = x^α exp(-x)`.
"""
struct LaguerreMeasure{T} <: Measure{T}
    α   ::  T
end

support(m::LaguerreMeasure{T}) where {T} = HalfLine{T}()

unsafe_weight(m::LaguerreMeasure, x) = x^m.α * exp(-x)



"""
The Hermite measure is the measure on `[0,∞)` with the classical generalized
Hermite weight `w(x) = x^α exp(-x)`.
"""
struct HermiteMeasure{T} <: Measure{T}
end

support(m::HermiteMeasure{T}) where {T} = DomainSets.FullSpace{T}()

unsafe_weight(m::HermiteMeasure, x) = exp(-x^2)


"""
The abstract supertype of discrete measures.
"""
abstract type DiscreteMeasure{T} <: Measure{T}
end

function unsafe_weight(m::DiscreteMeasure{T}, x) where {T}
    @warn "You might want to use `unsafe_discrete_weight`"
    convert(T, NaN)
end

"A Dirac function at a point `x`."
struct DiracMeasure{T} <: DiscreteMeasure{T}
    x   ::  T
end

support(m::DiracMeasure) = Point(m.x)

point(m::DiracMeasure) = m.x

unsafe_discrete_weight(m::DiracMeasure, i::Int) where {T} = one(T)

abstract type AbstractDiracCombMeasure{T} <: DiscreteMeasure{T}
end

support(m::AbstractDiracCombMeasure) = WrappedDomain(grid(m))

grid(m::AbstractDiracCombMeasure) = m.equispaced

struct DiractCombMeasure{T,EG<:AbstractEquispacedGrid} <: AbstractDiracCombMeasure{T}
    equispaced      :: EG

    DiracCombMeasure(eg::AbstractEquispacedGrid) = new{eltype(eg),typeof(eg)}(eg)
end

unsafe_discrete_weight(m::DiractCombMeasure, i::Int) where {T} = one(T)

struct WeightedDiractCombMeasure{T,EG<:AbstractEquispacedGrid,W<:AbstractArray} <: AbstractDiracCombMeasure{T}
    equispaced      :: EG
    weights         :: W

    function WeightedDiractCombMeasure(eg::AbstractEquispacedGrid{T}, w::AbstractArray=Ones{T}(length(eq))) where T
        @assert length(eq) == length(w)
        new{eltype(eg),typeof(eg),typeof(w)}(eg, w)
    end
end

unsafe_discrete_weight(m::WeightedDiractCombMeasure, i::Int) = m.weights[i]


######################################################
# Generating new measures from existing measures
######################################################

function restrict(m::LebesgueMeasure{T}, d::Domain{T}) where {T}
    @assert issubset(d, support(m))
    GenericLebesgueMeasure(d)
end

function restrict(m::LebesgueMeasure{T}, d::UnitInterval{T}) where {T}
    @assert issubset(d, support(m))
    FourierMeasure{T}()
end


struct SubMeasure{M,D,T} <: Measure{T}
    measure     ::  M
    domain      ::  D
end

SubMeasure(measure::Measure{T}, domain::Domain) where {T} =
    SubMeasure{typeof(measure),typeof(domain),T}(measure,domain)

supermeasure(measure::SubMeasure) = measure.measure
support(measure::SubMeasure) = measure.domain

unsafe_weight(m::SubMeasure, x) = unsafe_weight(supermeasure(m), x)

restrict(measure::Measure, domain::Domain) = SubMeasure(measure, domain)



struct MappedMeasure{MAP,M,T} <: Measure{T}
    map     ::  MAP
    measure ::  M
end

MappedMeasure(map, measure::Measure{T}) where {T} =
    MappedMeasure{typeof(map),typeof(measure),T}(map, measure)

mapping(m::MappedMeasure) = m.map

supermeasure(m::MappedMeasure) = m.measure

apply_map(measure::Measure, map) = MappedMeasure(map, measure)
apply_map(measure::MappedMeasure, map) = MappedMeasure(map*mapping(measure), supermeasure(measure))

support(m::MappedMeasure) = mapping(m) * support(supermeasure(m))

unsafe_weight(m::MappedMeasure, x) = unsafe_weight(supermeasure(m), inv(mapping(m))*x) / (jacobian(mapping(m))*x)



struct ProductMeasure{M,T} <: Measure{T}
    measures ::  M
end

product_domaintype(measures::Measure...) = Tuple{map(domaintype, measures)...}

function ProductMeasure(measures...)
    T = product_domaintype(measures...)
    ProductMeasure{typeof(measures),T}(measures)
end

iscomposite(m::ProductMeasure) = true
elements(m::ProductMeasure) = m.measures
element(m::ProductMeasure, i) = m.measures[i]

support(m::ProductMeasure) = cartesianproduct(map(support, elements(m)))

unsafe_weight(m::ProductMeasure, x) = prod(map(unsafe_weight, elements(m), x))
