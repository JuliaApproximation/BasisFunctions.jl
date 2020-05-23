
innerproduct(f, g, measure; options...) = integral(x->conj(f(x))*g(x), measure)

applymeasure(μ::AbstractMeasure, f::Function; options...) = integral(f, μ)


"A measure on a general domain with a general weight function `dσ = w(x) dx`."
struct GenericWeightMeasure{T} <: Measure{T}
    support          ::  Domain{T}
    weightfunction
end

name(m::GenericWeightMeasure) = "Measure with generic weight function"

unsafe_weight(m::GenericWeightMeasure, x) = m.weightfunction(x)

support(m::GenericWeightMeasure) = m.support

strings(m::GenericWeightMeasure) = (name(m), (string(m.support),), (string(m.weightfunction),))


name(m::DomainLebesgueMeasure) = "Lebesgue measure"
name(m::LegendreMeasure) = "Legendre measure"
name(m::JacobiMeasure) = "Jacobi measure (α = $(m.α), β = $(m.β))"
name(m::LaguerreMeasure) = m.α == 0 ? "Laguerre measure" : "Generalized Laguerre measure (α = $(m.α))"
name(m::HermiteMeasure) = "Hermite measure"
name(m::ChebyshevTMeasure) = "Chebyshev measure of the first kind"

const FourierMeasure = UnitLebesgueMeasure
name(m::FourierMeasure) = "Fourier (Lebesgue) measure"

name(m::ChebyshevUMeasure) = "Chebyshev measure of the second kind"



######################################################
# Generating new measures from existing measures
######################################################

## Mapped measure

"A mapped measure"
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
apply_map(measure::MappedMeasure, map) = MappedMeasure(map ∘ mapping(measure), supermeasure(measure))

support(m::MappedMeasure) = mapping(m).(support(supermeasure(m)))

unsafe_weight(m::MappedMeasure, x) = unsafe_weight(supermeasure(m), inverse(mapping(m), x)) / jacdet(mapping(m), x)

strings(m::MappedMeasure) = (name(m), strings(mapping(m)), strings(supermeasure(m)))

# For mapped measures, we can undo the map and leave out the jacobian in the
# weight function of the measure by going to the supermeasure

using DomainIntegrals: Identity,
    process_measure_default

const id = Identity()

import DomainIntegrals: islebesguemeasure,
    process_measure



function process_measure(qs, domain::DomainSets.MappedDomain, μ::MappedMeasure, sing)
    # TODO: think about what to do with the singularity here
    # -> Move MappedMeasure into DomainIntegrals.jl and implement the logic there
    m1 = mapping(μ)
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

function process_measure(qs, domain, μ::MappedMeasure, sing)
    # f -> f(m(x))
    map1 = mapping(μ)
    pre2, map2, domain2, μ2, sing =
        process_measure(qs, mapped_domain(map1, domain), supermeasure(μ), sing)
    pre2, map1 ∘ map2, domain2, μ2, sing
end



## Product measure

"The product measure"
struct ProductMeasure{T,M} <: Measure{T}
    measures ::  M
end


product_domaintype(measures::Measure...) = Tuple{map(domaintype, measures)...}
function product_domaintype(measures::Vararg{Measure{<:Number},N}) where {N}
    T = promote_type(map(domaintype, measures)...)
    SVector{N,T}
end

function ProductMeasure(measures::Measure...)
    T = product_domaintype(measures...)
    ProductMeasure{T}(measures...)
end
ProductMeasure{T}(measures::Measure...) where {T} =
    ProductMeasure{T,typeof(measures)}(measures)


islebesguemeasure(μ::ProductMeasure) = all(map(islebesguemeasure, elements(μ)))

productmeasure(measures::Measure...) = ProductMeasure(measures...)

elements(m::ProductMeasure) = m.measures
element(m::ProductMeasure, i) = m.measures[i]

isnormalized(m::ProductMeasure) = mapreduce(isnormalized, &, elements(m))

support(m::ProductMeasure{T,M}) where {T,M} = ProductDomain{T}(map(support, elements(m))...)

unsafe_weight(m::ProductMeasure, x) = mapreduce(unsafe_weight, *, elements(m), x)

# For product domains, process the measures dimension per dimension
function process_measure(qs, domain::ProductDomain, μ::ProductMeasure, sing)
    if numelements(domain) == numelements(μ)
        if numelements(domain) == 2
            pre1, map1, domain1, μ1, sing1 = process_measure(qs, element(domain, 1), element(μ, 1), sing)
            pre2, map2, domain2, μ2, sing2 = process_measure(qs, element(domain, 2), element(μ, 2), sing)
            prefactor = t -> pre1(t[1])*pre2(t[2])
            map = t -> SA[map1(t[1]), map2(t[2])]
            prefactor, map, ProductDomain(domain1, domain2), productmeasure(μ1, μ2), sing
        elseif numelements(domain) == 3
            pre1, map1, domain1, μ1, sing1 = process_measure(qs, element(domain, 1), element(μ, 1), sing)
            pre2, map2, domain2, μ2, sing2 = process_measure(qs, element(domain, 2), element(μ, 2), sing)
            pre3, map3, domain3, μ3, sing3 = process_measure(qs, element(domain, 3), element(μ, 3), sing)
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
