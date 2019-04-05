
"""
The span of a dictionary is the set of all possible expansions in that
dictionary, with coefficient eltype determined uniquely by the Dictionary.

The span of a dictionary is a function space, mapping `S` to `T`. Here, `S` is
the domain type of the dictionary. The `T` is the codomain type.
"""
struct Span{S,T} <: FunctionSpace{S,T}
    dictionary  ::  Dictionary{S}
end

Span(dict::Dictionary{S}) where S = Span{S,span_codomaintype(dict)}(dict)

span_codomaintype(dict::Dictionary) =
    span_codomaintype(coefficienttype(dict), codomaintype(dict))
# - When the types are the same, that type is the result
span_codomaintype(::Type{T}, ::Type{T}) where {T <: Number} = T
# - the coefficient types are complex and the set itself is real
span_codomaintype(::Type{Complex{T}}, ::Type{T}) where {T <: Number} = Complex{T}
# - or vice-versa
span_codomaintype(::Type{T}, ::Type{Complex{T}}) where {T <: Number} = Complex{T}
# Default fallback
span_codomaintype(::Type{A}, ::Type{Z}) where {Z,A} = typeof(zero(A) * zero(Z))

coefficienttype(span::Span) = coefficienttype(dictionary(span))

# Convenient shorthand
coefficienttype = coefficienttype

dictionary(s::Span) = s.dictionary

similar(s::Span, ::Type{T}, dims) where {T} = Span(similar(dictionary(s), T, dims))

promote_domaintype(span::Span, ::Type{T}) where {T} = similar(span, T, size(span))

random_expansion(span::Span) = Expansion(dictionary(span), rand(dictionary(span)))

zero(span::Span) = Expansion(dictionary(span), zeros(dictionary(span)))

tensorproduct(s1::Span, s2::Span) = Span(tensorproduct(dictionary(s1), dictionary(s2)))

size(span::Span) = size(dictionary(span))
length(span::Span) = length(dictionary(span))

name(span::Span) = "Span of a dictionary"
