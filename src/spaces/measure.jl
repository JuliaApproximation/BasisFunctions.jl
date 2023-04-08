
defaultmeasure(f, g) = _defaultmeasure(f, g, measure(f), measure(g))
function _defaultmeasure(Φ1, Φ2, m1, m2)
    if iscompatible(m1, m2)
        m1
    else
        if iscompatible(support(Φ1),support(Φ2))
            lebesguemeasure(support(Φ1))
        else
            error("Please specify which measure to use for the combination of $(Φ1) and $(Φ2).")
        end
    end
end

innerproduct(f, g; options...) =
    innerproduct(f, g, defaultmeasure(f, g); options...)
innerproduct(f, g, measure; options...) = integral(x->conj(f(x))*g(x), measure)

applymeasure(μ::Measure, f::Function; options...) = integral(f, μ)


"A measure on a general domain with a general weight function `dσ = w(x) dx`."
struct GenericWeight{T} <: Weight{T}
    support          ::  Domain{T}
    weightfunction
end

unsafe_weightfun(m::GenericWeight, x) = m.weightfunction(x)

support(m::GenericWeight) = m.support

const FourierWeight = DomainIntegrals.LebesgueUnit




######################################################
# Generating new measures from existing measures
######################################################

using DomainIntegrals: MappedWeight, ProductWeight

import DomainIntegrals: supermeasure, mappedmeasure


apply_map(μ::Measure, m) = MappedWeight(m, μ)


#############################

#############################
# Compatibility of measures
#############################

# By default, measures are compatible only when they are equal.
iscompatible(m1::M, m2::M) where {M <: Measure} = m1==m2
iscompatible(m1::Weight, m2::Weight) = false
