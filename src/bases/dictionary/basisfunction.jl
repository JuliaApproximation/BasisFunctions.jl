
export BasisFunction,
    index

abstract type TypedFunction{S,T} end

domaintype(::Type{<:TypedFunction{S,T}}) where {S,T} = S
domaintype(φ::TypedFunction) = domaintype(typeof(φ))

codomaintype(::Type{<:TypedFunction{S,T}}) where {S,T} = T
codomaintype(φ::TypedFunction) = codomaintype(typeof(φ))

prectype(F::Type{<:TypedFunction}) = prectype(domaintype(F),codomaintype(F))

eltype(::Type{<:Dictionary{S,T}}) where {S,T} = TypedFunction{S,T}


"A `BasisFunction` is one element of a dictionary."
struct BasisFunction{S,T,D<:Dictionary{S,T},I} <: TypedFunction{S,T}
    dictionary  ::  D
    index       ::  I
end

dictionary(φ::BasisFunction) = φ.dictionary
index(φ::BasisFunction) = φ.index

support(φ::BasisFunction) = support(dictionary(φ), index(φ))
measure(φ::BasisFunction) = measure(dictionary(φ))

name(φ::BasisFunction) = _name(φ, dictionary(φ))
_name(φ::BasisFunction, dict::Dictionary) = "Dictionary element"

(φ::BasisFunction)(x) = unsafe_eval_element1(dictionary(φ), index(φ), x)
(φ::BasisFunction)(x, y...) = φ(SVector(x, y...))


function getindex(dict::Dictionary, idx)
    @boundscheck checkbounds(dict, idx)
    BasisFunction(dict, idx)
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

analysis_integral(dict::Dictionary, idx, g, measure; options...) =
    _analysis_integral(dict, idx, g, measure, support(dict, idx), support(measure); options...)

# We take the intersection of the support of the basis function with the support of the measure.
# Perhaps one is smaller than the other.
function _analysis_integral(dict, idx, g, measure, domain1, domain2; options...)
    if domain1 == domain2
        integral(x->conj(unsafe_eval_element(dict, idx, x))*g(x), measure; options...)
    else
        integral(x->conj(unsafe_eval_element(dict, idx, x))*g(x)*unsafe_weight(measure,x), domain1 ∩ domain2; options...)
    end
end
