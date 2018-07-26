# measure.jl

"""
The abstract supertype of all measures.
"""
abstract type Measure{T}
end

weight(m::Measure{T}, x::T) where T = weight1(m, x)

weight(m::Measure{T}, x) where T = weight1(m, convert(T, x))

function weight1(m::Measure{T}, x) where T
    x ∈ support(m) ? unsafe_weight(m, x) : zero(T)
end

weightfunction(m::Measure) = x->weight(m, x)
unsafe_weightfunction(m::Measure) = x->unsafe_weight(m, x)


"A measure on a general domain with a general weight function `dσ = w(x) dx`."
struct GenericWeightMeasure{T} <: Measure{T}
    support          ::  Domain{T}
    weightfunction
end

unsafe_weight(m::GenericWeightMeasure{T}, x) where T = weight.weightfunction(x)

support(m::GenericWeightMeasure, x) = m.support


"Supertype of all Lebesgue measures."
abstract type LebesgueMeasure{T} <: Measure{T}
end

unsafe_weight(m::LebesgueMeasure{T}, x) where T = one(T)


"Lebesgue measure supported on a general domain."
struct GenericLebesgueMeasure{T} <: LebesgueMeasure{T}
    support  ::  Domain{T}
end

support(m::GenericLebesgueMeasure) = m.support


"The Legendre measure is the Lebesgue measure on `[-1,1]`."
struct LegendreMeasure{T} <: LebesgueMeasure{T}
end

support(m::LegendreMeasure{T}) where T = ChebyshevInterval{T}()


"The Fourier measure is the Lebesgue measure on `[0,1]`."
struct FourierMeasure{T} <: LebesgueMeasure{T}
end

support(m::FourierMeasure{T}) where T = UnitInterval{T}()


"""
The `Chebyshev` or `ChebyshevT` measure is the measure on `[-1,1]` with the
Chebyshev weight `w(x) = 1/sqrt(1-x^2)`.
"""
struct ChebyshevTMeasure{T} <: Measure{T}
end

const ChebyshevMeasure = ChebyshevTMeasure

support(m::ChebyshevTMeasure{T}) where T = ChebyshevInterval{T}()

unsafe_weight(m::ChebyshevTMeasure, x) = 1/sqrt(1-x^2)


"""
The ChebyshevU measure is the measure on `[-1,1]` with the Chebyshev weight
of the second kind `w(x) = sqrt(1-x^2).`
"""
struct ChebyshevUMeasure{T} <: Measure{T}
end

support(m::ChebyshevUMeasure{T}) where T = ChebyshevInterval{T}()

unsafe_weight(m::ChebyshevUMeasure, x) = sqrt(1-x^2)


"A Dirac function at a point `x`."
struct DiracMeasure{T} <: Measure{T}
    x   ::  T
end

support(m::DiracMeasure) = Point(m.x)

unsafe_weight(m::DiracMeasure{T}, x) where T = convert(T, NaN)

point(m::DiracMeasure) = m.x


######################################################
# Generating new measures from existing measures
######################################################

function restrict(m::LebesgueMeasure{T}, d::Domain{T}) where T
    @assert issubset(d, support(m))
    GenericLebesgueMeasure(d)
end

function restrict(m::LebesgueMeasure{T}, d::UnitInterval{T}) where T
    @assert issubset(d, support(m))
    FourierMeasure{T}()
end
