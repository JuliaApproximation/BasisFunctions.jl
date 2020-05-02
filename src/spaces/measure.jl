
innerproduct(f, g, measure; options...) = integral(x->conj(f(x))*g(x), measure)

applymeasure(m::AbstractMeasure, f::Function; options...) = default_applymeasure(m, f; options...)

function default_applymeasure(measure::Measure, f::Function; options...)
    @debug  "Applying measure $(typeof(measure)) numerically" maxlog=3
    integral(f, measure)
end

iscomposite(m::Measure) = false


"A measure on a general domain with a general weight function `dσ = w(x) dx`."
struct GenericWeightMeasure{T} <: Measure{T}
    support          ::  Domain{T}
    weightfunction
end

name(m::GenericWeightMeasure) = "Measure with generic weight function"

unsafe_weight(m::GenericWeightMeasure{T}, x) where {T} = m.weightfunction(x)

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


"The product measure"
struct ProductMeasure{T,M} <: Measure{T}
    measures ::  M
end


product_domaintype(measures::Measure...) = Tuple{map(domaintype, measures)...}

function ProductMeasure(measures::Vararg{Measure{<:Number},N}) where {N}
    T = promote_type(map(domaintype, measures)...)
    ProductMeasure{SVector{N,T}}(measures...)
end
function ProductMeasure(measures::Measure...)
    T = product_domaintype(measures...)
    ProductMeasure{T}(measures)
end
ProductMeasure{T}(measures::Measure...) where {T} =
    ProductMeasure{T,typeof(measures)}(measures)

productmeasure(measures::Measure...) = ProductMeasure(measures...)

iscomposite(m::ProductMeasure) = true
elements(m::ProductMeasure) = m.measures
element(m::ProductMeasure, i) = m.measures[i]
isnormalized(m::ProductMeasure) = reduce(&, map(isnormalized, elements(m)))

support(m::ProductMeasure{T,M}) where {T,M} = ProductDomain{T}(map(support, elements(m))...)

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
