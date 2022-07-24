
"""
The span of a dictionary is the set of all possible expansions in that
dictionary, with coefficient eltype determined uniquely by the dictionary.

The span of a dictionary is a function space, mapping `S` to `T`. Here, `S` is
the domain type of the dictionary and `T` is the codomain type.
"""
struct Span{S,T} <: FunctionSpace{S,T}
    dictionary  ::  Dictionary{S}
end

Span(dict::Dictionary{S}) where S = Span{S,span_codomaintype(dict)}(dict)

span_codomaintype(dict::Dictionary) =
    span_codomaintype(coefficienttype(dict), codomaintype(dict))
span_codomaintype(dict::Dictionary, coefficients) =
    span_codomaintype(eltype(coefficients), codomaintype(dict))
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

==(s1::Span, s2::Span) = false
==(s1::Span{S,T}, s2::Span{S,T}) where {S,T} =
    equalspan(s1, s2, dictionary(s1), dictionary(s2))
equalspan(s1, s2, dict1, dict2) = dict1 == dict2

size(span::Span) = size(dictionary(span))
length(span::Span) = length(dictionary(span))

show(io::IO, mime::MIME"text/plain", s::Span) = composite_show(io, mime, s)
Display.displaystencil(s::Span) = ["Span(", dictionary(s), ")"]
show(io::IO, s::Span) = print(io, "Space : $(domaintype(s)) → $(codomaintype(s)) (span of $(repr(dictionary(s))))")
