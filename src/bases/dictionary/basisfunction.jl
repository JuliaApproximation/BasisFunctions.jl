
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
function getindex(dict::Dictionary, i, j, indices...)
    @boundscheck checkbounds(dict, i, j, indices...)
    basisfunction(dict, (i,j,indices...))
end



# Inner product with a basis function: we choose the measure associated with the dictionary
innerproduct(φ::AbstractBasisFunction, g; options...) =
    innerproduct(φ, g, measure(φ); options...)
innerproduct(f, ψ::AbstractBasisFunction; options...) =
    innerproduct(f, ψ, measure(ψ); options...)
innerproduct(ϕ::AbstractBasisFunction, ψ::AbstractBasisFunction; options...) =
    innerproduct(ϕ, ψ, defaultmeasure(dictionary(ϕ), dictionary(ψ)); options...)

# The inner product between two basis functions: invoke the implementation of the dictionary
innerproduct(φ::AbstractBasisFunction, ψ::AbstractBasisFunction, measure; options...) =
    innerproduct(dictionary(φ), index(φ), dictionary(ψ), index(ψ), measure; options...)

# The inner product of a basis function with another function: this is an analysis integral
# We introduce a separate function name for this for easier dispatch.
innerproduct(φ::AbstractBasisFunction, g, measure; options...) =
    analysis_integral(dictionary(φ), index(φ), g, measure; options...)
innerproduct(f, ψ::AbstractBasisFunction, measure; options...) =
    conj(analysis_integral(dictionary(ψ), index(ψ), f, measure; options...))

# We want to check whether the supports of the basis function and the measure differ.
# The integral may be easier to evaluate by restricting to the intersection of these
# supports. However, we only perform this optimization if the intersection does not
# lead to an IntersectionDomain (i.e., the intersection is not recognized) since
# that leads to incomputable integrals.
function analysis_integral(dict::Dictionary, idx, g, measure::Measure; options...)
    @boundscheck checkbounds(dict, idx)
    domain1 = support(dict, idx)
    domain2 = support(measure)
    qs = quadstrategy(promote_type(prectype(dict), prectype(measure)); options...)
    unsafe_analysis_integral1(dict, idx, g, measure, domain1, domain2, domain1 ∩ domain2, qs)
end

function analysis_integral(dict::Dictionary, idx, g, measure::DiscreteWeight; options...)
    @boundscheck checkbounds(dict, idx)
    unsafe_analysis_integral2(dict, idx, g, measure, support(dict, idx))
end

# unsafe for indexing
function unsafe_analysis_integral1(dict, idx, g, measure::Weight, d1, d2, domain, qs)
    if d1 == d2
        integral(qs, x->conj(unsafe_eval_element(dict, idx, x))*g(x), measure)
    else
        # -> do compute on the smaller domain and convert the measure
        integral(qs, x->conj(unsafe_eval_element(dict, idx, x))*g(x), domain, measure)
    end
end

unsafe_analysis_integral1(dict, idx, g, measure, d1, d2, domain::IntersectionDomain, qs) =
    # -> disregard the intersection domain, but use eval_element to guarantee correctness
    integral(qs, x->conj(eval_element(dict, idx, x))*g(x), measure)

# unsafe for indexing and for support of integral
unsafe_analysis_integral2(dict::Dictionary, idx, g, measure::DiscreteWeight, domain) =
    integral(x->conj(unsafe_eval_element(dict, idx, x))*g(x), domain, measure)
