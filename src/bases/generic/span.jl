# span.jl

"""
The span of a dictionary is the set of all possible expansions in that
dictionary, with coefficient eltype determined uniquely by the Dictionary.

The span of a dictionary is a function space, mapping `S` to `T`. Here, `S` is
the domain type of the dictionary. The `T` is the codomain type.
"""
struct Span{S,T} <: AbstractFunctionSpace{S,T}
    dictionary  ::  Dictionary{S}
end

Span(dict::Dictionary{S}) where S = Span{S,span_codomaintype(dict)}(dict)

# What is the codomain type of a span? It depends on the type A of the
# coefficients, and on the codomain type T of the dictionary:
span_codomaintype(dict::Dictionary) =
    span_codomaintype(coefficient_type(dict), codomaintype(dict))
# - When the types are the same, that type is the result
span_codomaintype(::Type{T}, ::Type{T}) where {T <: Number} = T
# - the coefficient types are complex and the set itself is real
span_codomaintype(::Type{Complex{T}}, ::Type{T}) where {T <: Number} = Complex{T}
# - or vice-versa
span_codomaintype(::Type{T}, ::Type{Complex{T}}) where {T <: Number} = Complex{T}
# Default fallback
span_codomaintype(::Type{A}, ::Type{Z}) where {Z,A} = typeof(zero(A) * zero(Z))

coefficient_type(span::Span) = coefficient_type(dictionary(span))

# Convenient shorthand
coeftype = coefficient_type

dictionary(s::Span) = s.dictionary

function promote_domaintype(span::Span, ::Type{S}) where {S}
    newdict = promote_domaintype(dictionary(span), S)
    Span(newdict, promote_type(coefficient_type(span), coefficient_type(newdict)))
end

random_expansion(span::Span) = Expansion(dictionary(span), rand(dictionary(span)))

zero(span::Span) = Expansion(dictionary(span), zeros(dictionary(span)))

tensorproduct(s1::Span, s2::Span) = Span(tensorproduct(dictionary(s1), dictionary(s2)))
