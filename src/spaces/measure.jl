
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

const FourierWeight = LebesgueUnit
name(m::FourierWeight) = "Fourier (Lebesgue) measure"




######################################################
# Generating new measures from existing measures
######################################################

## Mapped measure

"A mapped weight function"
struct MappedWeight{MAP,M,T} <: Weight{T}
    map     ::  MAP
    measure ::  M
end

MappedWeight(map, measure::Weight{T}) where {T} =
    MappedWeight{typeof(map),typeof(measure),T}(map, measure)
mappedmeasure(map, measure::Weight) =
    MappedWeight(map, measure)

name(m::MappedWeight) = "Mapped measure"

forward_map(m::MappedWeight) = m.map

supermeasure(m::MappedWeight) = m.measure

apply_map(measure::Weight, map) = MappedWeight(map, measure)
apply_map(measure::MappedWeight, map) = MappedWeight(map ∘ forward_map(measure), supermeasure(measure))

support(m::MappedWeight) = forward_map(m).(support(supermeasure(m)))

unsafe_weightfun(m::MappedWeight, x) = unsafe_weightfun(supermeasure(m), inverse(forward_map(m), x)) / jacdet(forward_map(m), x)

strings(m::MappedWeight) = (name(m), strings(forward_map(m)), strings(supermeasure(m)))

# For mapped measures, we can undo the map and leave out the jacobian in the
# weight function of the measure by going to the supermeasure

using DomainIntegrals: Identity,
    process_measure_default

const id = Identity()

import DomainIntegrals: islebesguemeasure,
    process_measure



function process_measure(qs, domain::DomainSets.MappedDomain, μ::MappedWeight, sing)
    # TODO: think about what to do with the singularity here
    # -> Move MappedWeight into DomainIntegrals.jl and implement the logic there
    m1 = forward_map(μ)
    m2 = forward_map(domain)
    if iscompatible(m1,m2)
        # f -> f(m1(x))
        pre2, map2, domain2, μ2, sing =
            process_measure(qs, superdomain(domain), supermeasure(μ), sing)
        pre2, m1 ∘ map2, domain2, μ2, sing
    else
        id, id, domain, μ, sing
    end
end

function process_measure(qs, domain, μ::MappedWeight, sing)
    # f -> f(m(x))
    map1 = forward_map(μ)
    pre2, map2, domain2, μ2, sing =
        process_measure(qs, mapped_domain(map1, domain), supermeasure(μ), sing)
    pre2, map1 ∘ map2, domain2, μ2, sing
end



## Product measure

"A product weight."
struct ProductWeight{T,M} <: Weight{T}
    measures ::  M
end


product_domaintype(measures::Weight...) = Tuple{map(domaintype, measures)...}
function product_domaintype(measures::Vararg{Weight{<:Number},N}) where {N}
    T = promote_type(map(domaintype, measures)...)
    SVector{N,T}
end

function ProductWeight(measures::Weight...)
    T = product_domaintype(measures...)
    ProductWeight{T}(measures...)
end
ProductWeight{T}(measures::Weight...) where {T} =
    ProductWeight{T,typeof(measures)}(measures)


islebesguemeasure(μ::ProductWeight) = all(map(islebesguemeasure, components(μ)))

productmeasure(measures::Weight...) = ProductWeight(measures...)

components(m::ProductWeight) = m.measures
component(m::ProductWeight, i) = m.measures[i]

isnormalized(m::ProductWeight) = mapreduce(isnormalized, &, components(m))

support(m::ProductWeight{T,M}) where {T,M} = ProductDomain{T}(map(support, components(m))...)

unsafe_weightfun(m::ProductWeight, x) = mapreduce(unsafe_weightfun, *, components(m), x)

# For product domains, process the measures dimension per dimension
function process_measure(qs, domain::ProductDomain, μ::ProductWeight, sing)
    if ncomponents(domain) == ncomponents(μ)
        if ncomponents(domain) == 2
            pre1, map1, domain1, μ1, sing1 = process_measure(qs, component(domain, 1), component(μ, 1), sing)
            pre2, map2, domain2, μ2, sing2 = process_measure(qs, component(domain, 2), component(μ, 2), sing)
            prefactor = t -> pre1(t[1])*pre2(t[2])
            map = t -> SA[map1(t[1]), map2(t[2])]
            prefactor, map, ProductDomain(domain1, domain2), productmeasure(μ1, μ2), sing
        elseif ncomponents(domain) == 3
            pre1, map1, domain1, μ1, sing1 = process_measure(qs, component(domain, 1), component(μ, 1), sing)
            pre2, map2, domain2, μ2, sing2 = process_measure(qs, component(domain, 2), component(μ, 2), sing)
            pre3, map3, domain3, μ3, sing3 = process_measure(qs, component(domain, 3), component(μ, 3), sing)
            prefactor = t -> pre1(t[1])*pre2(t[2])*pre3(t[3])
            map = t -> SA[map1(t[1]), map2(t[2]), map3(t[3])]
            prefactor, map, ProductDomain(domain1, domain2, domain3), productmeasure(μ1, μ2, μ3), sing
        else
            process_measure_default(qs, domain, μ, sing)
        end
    else
        process_measure_default(qs, domain, μ, sing)
    end
end


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
# Compatibility of measures
#############################

# By default, measures are compatible only when they are equal.
iscompatible(m1::M, m2::M) where {M <: Measure} = m1==m2
iscompatible(m1::Weight, m2::Weight) = false
