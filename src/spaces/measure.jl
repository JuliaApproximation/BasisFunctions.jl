
innerproduct(f, g, measure; options...) = integral(x->conj(f(x))*g(x), measure)

applymeasure(μ::Measure, f::Function; options...) = integral(f, μ)


"A measure on a general domain with a general weight function `dσ = w(x) dx`."
struct GenericWeight{T} <: Weight{T}
    support          ::  Domain{T}
    weightfunction
end

name(m::GenericWeight) = "Measure with generic weight function"

unsafe_weightfun(m::GenericWeight, x) = m.weightfunction(x)

support(m::GenericWeight) = m.support

strings(m::GenericWeight) = (name(m), (string(m.support),), (string(m.weightfunction),))


name(m::LebesgueDomain) = "Lebesgue measure"
name(m::LegendreWeight) = "Legendre weight"
name(m::JacobiWeight) = "Jacobi weight (α = $(m.α), β = $(m.β))"
name(m::LaguerreWeight) = m.α == 0 ? "Laguerre weight" : "Generalized Laguerre measure (α = $(m.α))"
name(m::HermiteWeight) = "Hermite weight"
name(m::ChebyshevTWeight) = "Chebyshev weight of the first kind"
name(m::ChebyshevUWeight) = "Chebyshev weight of the second kind"

const FourierWeight = DomainIntegrals.LebesgueUnit
name(m::FourierWeight) = "Fourier (Lebesgue) measure"




######################################################
# Generating new measures from existing measures
######################################################

using DomainIntegrals: MappedWeight, ProductWeight

import DomainIntegrals: supermeasure, mappedmeasure

name(m::MappedWeight) = "Mapped measure"
strings(m::MappedWeight) = (name(m), strings(forward_map(m)), strings(supermeasure(m)))


apply_map(μ::Measure, m) = MappedWeight(m, μ)

function stencilarray(m::ProductWeight)
    A = Any[]
    push!(A, component(m,1))
    for i = 2:length(components(m))
        push!(A," ⊗ ")
        push!(A, component(m,i))
    end
    A
end

#############################

#############################
# Compatibility of measures
#############################

# By default, measures are compatible only when they are equal.
iscompatible(m1::M, m2::M) where {M <: Measure} = m1==m2
iscompatible(m1::Weight, m2::Weight) = false
