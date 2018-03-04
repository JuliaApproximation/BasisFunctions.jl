# span.jl

"""
The span of a dictionary is the set of all possible expansions in that
dictionary, with coefficients having eltype `A`.

The span of a dictionary is a function space, mapping `S` to `T`. Here, `S` is
the domain type of the dictionary. The codomain type `T` of the span results from
the multiplication of values of type `A` with the a value from the codomain type
of the dictionary.
"""
struct Span{A,S,T,D <: Dictionary} <: FunctionSpace{S,T}
    dictionary  ::  D
end

const Span1d{A,S,T,D <: Dictionary1d} = Span{A,S,T,D}
const Span2d{A,S,T,D <: Dictionary2d} = Span{A,S,T,D}
const Span3d{A,S,T,D <: Dictionary3d} = Span{A,S,T,D}
const Span4d{A,S,T,D <: Dictionary4d} = Span{A,S,T,D}


# What is the codomain type of a span? It depends on the type A of the
# coefficients, and on the codomain type T of the dictionary:
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


# We use a default coefficient type (as returned by the dict) when none is given
Span(dict::Dictionary) = Span(dict, coefficient_type(dict))

Span(dict::Dictionary{S,T}, ::Type{A}) where {A,S,T} =
    Span{A,S,span_codomaintype(A,T),typeof(dict)}(dict)

# Use this constructor to explicitly set coefficient type, domain and
# codomain type
Span{A,S,T}(dict::Dictionary) where {A,S,T} = Span{A,S,T,typeof(dict)}(dict)


coefficient_type(::Type{Span{A,S,T,D}}) where {A,S,T,D} = A
coefficient_type(::Type{S}) where {S <: Span} = coefficient_type(supertype(D))
coefficient_type(::Span{A,S,T,D}) where {A,S,T,D} = A
# Convenient shorthand
coeftype = coefficient_type

domaintype(::Type{Span{A,S,T,D}}) where {A,S,T,D} = S
domaintype(::Type{S}) where {S <: Span} = domaintype(supertype(D))
domaintype(::Span{A,S,T,D}) where {A,S,T,D} = S

codomaintype(::Type{Span{A,S,T,D}}) where {A,S,T,D} = T
codomaintype(::Type{S}) where {S <: Span} = codomaintype(supertype(D))
codomaintype(::Span{A,S,T,D}) where {A,S,T,D} = T

dictionary(s::Span) = s.dictionary


# The difference between similar_span and promote_coeftype is that similar_span
# sets the coefficient type of the new span to the given argument, whereas
# promote_coeftype sets the coefficient type to promote_type of the old and new
# coefficient types.
similar_span(span::Span{A}, ::Type{A}) where {A} = span
similar_span(span::Span{A}, ::Type{B}) where {A,B} = Span(dictionary(span), B)

# For convenience, one can also create a span with the same coefficient type
# but with a different function set
similar_span(span::Span, dict::Dictionary) = Span(dict, coefficient_type(span))

promote_coefficient_type(span::Span{A}, ::Type{A}) where {A} = span
promote_coefficient_type(span::Span{A}, ::Type{B}) where {A,B} = Span(dictionary(span), promote_type(A,B))

# Convenient shorthand
promote_coeftype = promote_coefficient_type

function promote_domaintype(span::Span, ::Type{S}) where {S}
    newdict = promote_domaintype(dictionary(span), S)
    Span(newdict, promote_type(coefficient_type(span), coefficient_type(newdict)))
end

########################
# Interface delegation
########################

# We provide a part of the interface of a Dictionary for the Span, by delegating
# to the underlying dictionary.

elements(span::Span) = map(s -> Span(s, coefficient_type(span)), elements(dictionary(span)))
element(span::Span, i) = Span(element(dictionary(span), i), coefficient_type(span))

isreal(span::Span) = isreal(coefficient_type(span)) && isreal(dictionary(span))

for op in (:length, :size, :dimension, :grid)
    @eval $op(span::Span) = $op(dictionary(span))
end

compatible_grid(span::Span, grid::AbstractGrid) = compatible_grid(dictionary(span), grid)

for op in (:has_transform, :has_extension, :has_derivative, :has_antiderivative,
    :has_grid)
    @eval $op(span::Span) = $op(dictionary(span))
end

has_transform(s1::Span, s2::Span) = has_transform(dictionary(s1), dictionary(s2))
has_transform(s1::Span, g::AbstractGrid) = has_transform(dictionary(s1), g)

resize(span::Span, n) = Span(resize(dictionary(span), n), coefficient_type(span))

zeros(span::Span) = zeros(coefficient_type(span), dictionary(span))

function ones(span::Span)
    A = coefficient_type(span)
    c = zeros(span)
    for i in eachindex(c)
        c[i] = one(A)
    end
    c
end

# Generate a random value of type T
random_value(::Type{T}) where {T <: Number} = convert(T, rand())
random_value(::Type{Complex{T}}) where {T <: Real} = T(rand()) + im*T(rand())
random_value(::Type{T}) where {T} = rand() * one(T)

# Compute a random expansion
function rand(span::Span)
    A = coefficient_type(span)
    c = zeros(span)
    for i in eachindex(c)
        c[i] = random_value(A)
    end
    c
end

random_expansion(span::Span) = Expansion(dictionary(span), rand(span))

zero(span::Span) = Expansion(dictionary(span), zeros(span))

complex(span::Span) = promote_coefficient_type(span, complex(coefficient_type(span)))

real(span::Span) = similar_span(span, real(coefficient_type(span)))

eltype(span::Span) = error("The eltype of a span is not supported. Perhaps you meant coefficient_type?")

linearize_coefficients(span::Span, coef_native) = linearize_coefficients(dictionary(span), coef_native)
linearize_coefficients!(span::Span, coef_linear, coef_native) = linearize_coefficients!(dictionary(span), coef_linear, coef_native)

delinearize_coefficients(span::Span, coef_linear) = linearize_coefficients(dictionary(span), coef_linear)
delinearize_coefficients!(span::Span, coef_native, coef_linear) = delinearize_coefficients!(dictionary(span), coef_native, coef_linear)

tensorproduct(s1::Span{A}, s2::Span{A}) where {A} = Span(tensorproduct(dictionary(s1), dictionary(s2)), A)
tensorproduct(s1::Span{A}, s2::Span{B}) where {A,B} = Span(tensorproduct(dictionary(s1), dictionary(s2)), promote_type(A,B))

for op in (:extend, :restrict)
    @eval $op(s::Span) = Span($op(dictionary(s)), coefficient_type(s))
end

native_index(s::Span, idx) = native_index(dictionary(s), idx)
multilinear_index(s::Span, idx) = multilinear_index(dictionary(s), idx)
linear_index(s::Span, idxn) = linear_index(dictionary(s), idxn)

# A concrete Dictionary has spaces associated with derivatives or antiderivatives of a certain order,
# and it should implement the following introspective functions:
# derivative_space(s::MyDictionary, order) = ...
# antiderivative_space(s::MyDictionary, order) = ...
# where order is either an Int (in 1D) or a tuple of Int's (in higher dimensions).

# The default order is 1 for 1d sets:
derivative_space(s::Span1d) = derivative_space(s, 1)
antiderivative_space(s::Span1d) = antiderivative_space(s, 1)

# Catch tuples with just one element and convert to Int
derivative_space(s::Span1d, order::Tuple{Int}) = derivative_space(s, order[1])
antiderivative_space(s::Span1d, order::Tuple{Int}) = antiderivative_space(s, order[1])

# This is a candidate for a better implementation. How does one generate a
# unit vector in a tuple?
# ASK is this indeed a better implementation?
dimension_tuple(n, dim) = ntuple(k -> (k==dim? 1: 0), n)

# Convenience function to differentiate in a given dimension
derivative_space(s::Dictionary; dim=1) = derivative_space(s, dimension_tuple(dimension(s), dim))
antiderivative_space(s::Dictionary; dim=1) = antiderivative_space(s, dimension_tuple(dimension(s), dim))

==(s1::Span, s2::Span) = (coefficient_type(s1) == coefficient_type(s2)) && (dictionary(s1) == dictionary(s2))
