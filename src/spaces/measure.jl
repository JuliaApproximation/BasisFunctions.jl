
"""
The abstract supertype of all measures.
"""
abstract type AbstractMeasure{T}
end


isprobabilitymeasure(m::AbstractMeasure; options...) = error("isprobabilitymeasure not implemented for measure $(typeof(m)).")
applymeasure(m::AbstractMeasure, f::Function; options...) = default_applymeasure(m, f; options...)

domaintype(m::AbstractMeasure{T}) where {T} = T
subdomaintype(m::AbstractMeasure) = subeltype(domaintype(m))
rangetype(m::AbstractMeasure{T}) where {T} = T
codomaintype(m::AbstractMeasure{T}) where {T} = subeltype(T)

"""
The abstract supertype of all continuous measures.

I.e., the `weight` is a function and the `support` has mass.
"""
abstract type Measure{T} <: AbstractMeasure{T}
end

weight(m::Measure{T}, x::T) where {T} = weight1(m, x)
weight(m::Measure{T}, x) where {T} = weight1(m, convert(T, x))

function default_applymeasure(measure::Measure, f::Function; options...)
    @debug  "Applying measure $(typeof(measure)) numerically" maxlog=3
    integral(f, measure; options...)
end

function weight1(m::Measure{T}, x) where {T}
    x ∈ support(m) ? unsafe_weight(m, x) : zero(rangetype(m))
end

weightfunction(m::Measure) = x->weight(m, x)
unsafe_weightfunction(m::Measure) = x->unsafe_weight(m, x)
iscomposite(m::Measure) = false


"""
The abstract supertype of discrete measures.

I.e., the `weight` is a vector and the `support` contains a `grid`.
"""
abstract type DiscreteMeasure{T} <: AbstractMeasure{T}
end

grid(m::DiscreteMeasure) = m.grid
support(m::DiscreteMeasure) = grid(m)#DomainSets.WrappedDomain(grid(m)) the support is no domain, but it is a set, i.e., a vector
weights(m::DiscreteMeasure) = m.weights
discrete_weight(m::DiscreteMeasure, i) = (@boundscheck checkbounds(m, i); unsafe_discrete_weight(m, i))
checkbounds(m::DiscreteMeasure, i) = checkbounds(grid(m), i)
unsafe_discrete_weight(m::DiscreteMeasure, i) where {T} = Base.unsafe_getindex(m.weights, i)
isprobabilitymeasure(m::DiscreteMeasure) = sum(m.weights) ≈ 1
function default_applymeasure(measure::DiscreteMeasure, f::Function; options...)
    integral(f, measure; options...)
end

"Supertype of all Lebesgue measures."
abstract type LebesgueMeasure{T} <: Measure{T}
end

unsafe_weight(m::LebesgueMeasure{T}, x) where {T} = one(T)

"A measure on a general domain with a general weight function `dσ = w(x) dx`."
struct GenericWeightMeasure{T} <: Measure{T}
    support          ::  Domain{T}
    weightfunction
end

name(m::GenericWeightMeasure) = "Measure with generic weight function"

unsafe_weight(m::GenericWeightMeasure{T}, x) where {T} = m.weightfunction(x)

support(m::GenericWeightMeasure) = m.support

strings(m::GenericWeightMeasure) = (name(m), (string(m.support),), (string(m.weightfunction),))


struct WholeLebesgueMeasure{T} <: LebesgueMeasure{T}
end

support(m::WholeLebesgueMeasure{T}) where {T} = DomainSets.FullSpace{T}()

name(m::WholeLebesgueMeasure) = "Lebesgue measure on the full space"


"Lebesgue measure supported on a general domain."
struct GenericLebesgueMeasure{T} <: LebesgueMeasure{T}
    support  ::  Domain{T}
end

support(m::GenericLebesgueMeasure) = m.support

name(m::GenericLebesgueMeasure) = "Lebesgue measure"

"The Legendre measure is the Lebesgue measure on `[-1,1]`."
struct LegendreMeasure{T} <: LebesgueMeasure{T}
end

LegendreMeasure() = LegendreMeasure{Float64}()

support(m::LegendreMeasure{T}) where {T} = ChebyshevInterval{T}()

name(m::LegendreMeasure) = "Legendre measure"

isprobabilitymeasure(::LegendreMeasure) = false

"The Fourier measure is the Lebesgue measure on `[0,1]`."
struct FourierMeasure{T} <: LebesgueMeasure{T}
end

FourierMeasure() = FourierMeasure{Float64}()

support(m::FourierMeasure{T}) where {T} = UnitInterval{T}()

name(m::FourierMeasure) = "Fourier (Lebesgue) measure"

isprobabilitymeasure(::FourierMeasure) = true

lebesguemeasure(domain::UnitInterval{T}) where {T} = FourierMeasure{T}()
lebesguemeasure(domain::ChebyshevInterval{T}) where {T} = LegendreMeasure{T}()
lebesguemeasure(domain::DomainSets.FullSpace{T}) where {T} = WholeLebesgueMeasure{T}()
lebesguemeasure(domain::Domain{T}) where {T} = GenericLebesgueMeasure{T}(domain)


"""
The `Chebyshev` or `ChebyshevT` measure is the measure on `[-1,1]` with the
Chebyshev weight `w(x) = 1/√(1-x^2)`.
"""
struct ChebyshevTMeasure{T} <: Measure{T}
end
ChebyshevTMeasure() = ChebyshevTMeasure{Float64}()

const ChebyshevMeasure = ChebyshevTMeasure

support(m::ChebyshevTMeasure{T}) where {T} = ChebyshevInterval{T}()

name(m::ChebyshevTMeasure) = "Chebyshev measure of the first kind"

unsafe_weight(m::ChebyshevTMeasure, x) = 1/sqrt(1-x^2)

isprobabilitymeasure(::ChebyshevTMeasure) = false# is pi

"""
The ChebyshevU measure is the measure on `[-1,1]` with the Chebyshev weight
of the second kind `w(x) = √(1-x^2).`
"""
struct ChebyshevUMeasure{T} <: Measure{T}
end
ChebyshevUMeasure() = ChebyshevUMeasure{Float64}()

support(m::ChebyshevUMeasure{T}) where {T} = ChebyshevInterval{T}()

name(m::ChebyshevUMeasure) = "Chebyshev measure of the second kind"

unsafe_weight(m::ChebyshevUMeasure, x) = sqrt(1-x^2)

isprobabilitymeasure(::ChebyshevUMeasure) = false # is pi/2

"""
The Jacobi measure is the measure on `[-1,1]` with the classical Jacobi weight
`w(x) = (1-x)^α (1+x)^β`.
"""
struct JacobiMeasure{T} <: Measure{T}
    α   ::  T
    β   ::  T
end

support(m::JacobiMeasure{T}) where {T} = ChebyshevInterval{T}()

name(m::JacobiMeasure) = "Jacobi measure (α = $(m.α), β = $(m.β))"

unsafe_weight(m::JacobiMeasure, x) = (1-x)^m.α * (1+x)^m.β

isprobabilitymeasure(::JacobiMeasure) = false


"""
The Laguerre measure is the measure on `[0,∞)` with the classical generalized
Laguerre weight `w(x) = x^α exp(-x)`.
"""
struct LaguerreMeasure{T} <: Measure{T}
    α   ::  T
end

support(m::LaguerreMeasure{T}) where {T} = HalfLine{T}()

name(m::LaguerreMeasure) = m.α == 0 ? "Laguerre measure" : "Generalized Laguerre measure (α = $(m.α))"

unsafe_weight(m::LaguerreMeasure, x) = x^m.α * exp(-x)

isprobabilitymeasure(m::LaguerreMeasure) = m.α == 0


"""
The Hermite measure is the measure on `[0,∞)` with the classical generalized
Hermite weight `w(x) = x^α exp(-x)`.
"""
struct HermiteMeasure{T} <: Measure{T}
end

support(m::HermiteMeasure{T}) where {T} = DomainSets.FullSpace{T}()

name(m::HermiteMeasure) = "Hermite measure"

unsafe_weight(m::HermiteMeasure, x) = exp(-x^2)

isprobabilitymeasure(::HermiteMeasure) = false


######################################################
# Generating new measures from existing measures
######################################################

struct MappedMeasure{MAP,M,T} <: Measure{T}
    map     ::  MAP
    measure ::  M
end

MappedMeasure(map, measure::Measure{T}) where {T} =
    MappedMeasure{typeof(map),typeof(measure),T}(map, measure)
mappedmeasure(map, measure::Measure) =
    MappedMeasure(map, measure)

name(m::MappedMeasure) = "Mapped measure"

mapping(m::MappedMeasure) = m.map

supermeasure(m::MappedMeasure) = m.measure

apply_map(measure::Measure, map) = MappedMeasure(map, measure)
apply_map(measure::MappedMeasure, map) = MappedMeasure(map*mapping(measure), supermeasure(measure))

support(m::MappedMeasure) = mapping(m) * support(supermeasure(m))

unsafe_weight(m::MappedMeasure, x) = unsafe_weight(supermeasure(m), inv(mapping(m))*x) / (jacobian(mapping(m))*x)

strings(m::MappedMeasure) = (name(m), strings(mapping(m)), strings(supermeasure(m)))



struct ProductMeasure{M,T} <: Measure{T}
    measures ::  M
end


product_domaintype(measures::Measure...) = Tuple{map(domaintype, measures)...}

function ProductMeasure(measures::Measure...)
    T = product_domaintype(measures...)
    ProductMeasure{typeof(measures),T}(measures)
end
productmeasure(measures::Measure...) = ProductMeasure(measures...)

iscomposite(m::ProductMeasure) = true
elements(m::ProductMeasure) = m.measures
element(m::ProductMeasure, i) = m.measures[i]
isprobabilitymeasure(m::ProductMeasure) = reduce(&, map(isprobabilitymeasure, elements(m)))

support(m::ProductMeasure) = cartesianproduct(map(support, elements(m)))

weight1(m::ProductMeasure, x) = prod(map(weight1, elements(m), x))

function stencilarray(m::ProductMeasure)
    A = Any[]
    push!(A, element(m,1))
    for i = 2:length(elements(m))
        push!(A," ⊗ ")
        push!(A, element(m,i))
    end
    A
end

#############################
# Compatibility of measures
#############################

# By default, measures are compatible only when they are equal.
iscompatible(m1::M, m2::M) where {M <: Measure} = m1==m2
iscompatible(m1::Measure, m2::Measure) = false

#############################
# Discrete measures
#############################
include("discretemeasure.jl")
