
export BasisFunction,
    index

abstract type TypedFunction{S,T} end

domaintype(::Type{<:TypedFunction{S,T}}) where {S,T} = S
domaintype(φ::TypedFunction) = domaintype(typeof(φ))

codomaintype(::Type{<:TypedFunction{S,T}}) where {S,T} = T
codomaintype(φ::TypedFunction) = codomaintype(typeof(φ))

prectype(F::Type{<:TypedFunction}) = prectype(domaintype(F),codomaintype(F))

eltype(::Type{<:Dictionary{S,T}}) where {S,T} = TypedFunction{S,T}

promote_rule(::Type{<:TypedFunction{S1,T1}}, ::Type{<:TypedFunction{S2,T2}}) where {S1,T1,S2,T2} =
    TypedFunction{promote_type(S1,S2),promote_type(T1,T2)}

iscomposite(f::TypedFunction) = false

"The supertype of functions that can be associated with a dictionary or a family of basis functions."
abstract type AbstractBasisFunction{S,T} <: TypedFunction{S,T} end

"A `BasisFunction` is one element of a dictionary."
struct BasisFunction{S,T,D<:Dictionary{S,T},I} <: AbstractBasisFunction{S,T}
    dictionary  ::  D
    index       ::  I
end

dictionary(φ::BasisFunction) = φ.dictionary
index(φ::BasisFunction) = φ.index

support(φ::BasisFunction) = support(dictionary(φ), index(φ))
measure(φ::BasisFunction) = measure(dictionary(φ))

name(φ::BasisFunction) = _name(φ, dictionary(φ))
_name(φ::BasisFunction, dict::Dictionary) = "φ(x)   (dictionary element of $(name(dict)))"

(φ::BasisFunction)(x) = unsafe_eval_element1(dictionary(φ), index(φ), x)
(φ::BasisFunction)(x, y...) = φ(SVector(x, y...))

basisfunction(dict::Dictionary, idx) = BasisFunction(dict, idx)

function getindex(dict::Dictionary, idx)
    @boundscheck checkbounds(dict, idx)
    basisfunction(dict, idx)
end
getindex(dict::Dictionary, i, j, indices...) =
    getindex(dict, (i,j,indices...))



# Inner product with a basis function: we choose the measure associated with the dictionary
innerproduct(φ::BasisFunction, f; options...) =
    innerproduct(φ, f, measure(φ); options...)

# The inner product between two basis functions: invoke the implementation of the dictionary
innerproduct(φ::BasisFunction, ψ::BasisFunction, measure; options...) =
    innerproduct(dictionary(φ), index(φ), dictionary(ψ), index(ψ), measure; options...)

# The inner product of a basis function with another function: this is an analysis integral
# We introduce a separate function name for this for easier dispatch.
innerproduct(φ::BasisFunction, g, measure; options...) =
    analysis_integral(dictionary(φ), index(φ), g, measure; options...)

# We want to check whether the supports of the basis function and the measure differ.
# The integral may be easier to evaluate by restricting to the intersection of these
# supports. However, we only perform this optimization if the intersection does not
# lead to an IntersectionDomain (i.e., the intersection is not recognized) since
# that leads to incomputable integrals.
function analysis_integral(dict::Dictionary, idx, g, measure::AbstractMeasure; options...)
    @boundscheck checkbounds(dict, idx)
    domain1 = support(dict, idx)
    domain2 = support(measure)
    unsafe_analysis_integral1(dict, idx, g, measure, domain1, domain2, domain1 ∩ domain2; options...)
end

function analysis_integral(dict::Dictionary, idx, g, measure::DiscreteMeasure; options...)
    @boundscheck checkbounds(dict, idx)
    unsafe_analysis_integral2(dict, idx, g, measure, support(dict, idx); options...)
end

# unsafe for indexing
function unsafe_analysis_integral1(dict, idx, g, measure::Measure, d1, d2, domain; options...)
    if d1 == d2
        # -> domains are the same, don't convert the measure
        unsafe_analysis_integral2(dict, idx, g, measure; options...)
    else
        # -> do compute on the smaller domain and convert the measure
        # TODO: use DomainIntegrals.jl code that enables both measures and domains
        integral(x->conj(unsafe_eval_element(dict, idx, x))*g(x)*unsafe_weight(measure,x), domain; options...)
    end
end

@inbounds unsafe_analysis_integral1(dict, idx, g, measure, d1, d2, domain::IntersectionDomain; options...) =
    # -> disregard the intersection domain, but use eval_element to guarantee correctness
    integral(x->conj(eval_element(dict, idx, x))*g(x), measure; options...)

# unsafe for indexing and for support of integral
@inbounds unsafe_analysis_integral2(dict::Dictionary, idx, g, measure::AbstractMeasure; options...) =
    integral(x->conj(unsafe_eval_element(dict, idx, x))*g(x), measure; options...)

@inbounds unsafe_analysis_integral2(dict::Dictionary, idx, g, measure::DiscreteMeasure, domain; options...) =
    integral(x->conj(unsafe_eval_element(dict, idx, x))*g(x), measure, domain; options...)
