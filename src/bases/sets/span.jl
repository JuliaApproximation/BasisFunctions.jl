# span.jl

"""
The span of a function set is the set of all possible expansions in that set,
where the coefficients have eltype `A`.
"""
struct Span{A,F} <: FunctionSpace
    set ::  F
end

const Span1d{A,F <: FunctionSet1d} = Span{A,F}
const Span2d{A,F <: FunctionSet2d} = Span{A,F}

Span(set::FunctionSet, ::Type{A} = coefficient_type(set)) where {A} = Span{A,typeof(set)}(set)

set(s::Span) = s.set

span(set::FunctionSet, ::Type{A} = coefficient_type(set)) where {A} = Span(set, A)

coefficient_type(::Span{A,F}) where {A,F} = A

coeftype = coefficient_type

# The difference between similar_span and promote_coeftype is that similar_span
# sets the coefficient type of the new span to the given argument, whereas
# promote_coeftype sets the coefficient type to promote_type of the old and new
# coefficient types.
similar_span(span::Span{A,F}, ::Type{A}) where {A,F} = span
similar_span(span::Span{A,F}, ::Type{B}) where {A,B,F} = Span(set(span), B)

# For convenience, one can also create a span with the same coefficient type
# but with a different function set
similar_span(span::Span, set::FunctionSet) = Span(set, coeftype(span))

promote_coeftype(span::Span{A,F}, ::Type{A}) where {A,F} = span
promote_coeftype(span::Span{A,F}, ::Type{B}) where {A,B,F} = Span(set(span), promote_type(A,B))

function promote_domaintype(span::Span, ::Type{S}) where {S}
    newset = promote_domaintype(set(span), S)
    Span(newset, promote_type(coeftype(span), coeftype(newset)))
end

# What is the rangetype of a span? It depends on the type of the coefficients,
# and on the rangetype of the set.
rangetype(span::Span) = _rangetype(coefficient_type(span), rangetype(set(span)))
# - When the types are the same, it is the result
_rangetype(::Type{T}, ::Type{T}) where {T <: Number} = T
# - the coefficient types are complex and the set itself is real
_rangetype(::Type{Complex{T}}, ::Type{T}) where {T <: Number} = Complex{T}
# Default fallback
_rangetype(::Type{A}, ::Type{Z}) where {Z,A} = typeof(zero(A) * zero(Z))

# For convenience
rangetype(set::FunctionSet, coefficients) = rangetype(set, eltype(coefficients))
rangetype(set::FunctionSet, ::Type{A}) where {A} = rangetype(Span(set, A))

rangetype(span1::Span, span2::Span) = promote_eltype(rangetype(span1), rangetype(span2))

elements(span::Span) = map(s -> Span(s, coeftype(span)), elements(set(span)))
element(span::Span, i) = Span(element(set(span), i), coeftype(span))

isreal(span::Span{A,F}) where {A,F} = isreal(A) && isreal(set(span))

for op in (:length, :size, :dimension, :domaintype, :grid)
    @eval $op(span::Span) = $op(set(span))
end

compatible_grid(span::Span, grid::AbstractGrid) = compatible_grid(set(span), grid)

for op in (:has_transform, :has_extension, :has_derivative, :has_antiderivative,
    :has_grid)
    @eval $op(span::Span) = $op(set(span))
end

has_transform(s1::Span, s2::Span) = has_transform(set(s1), set(s2))
has_transform(s1::Span, g::AbstractGrid) = has_transform(set(s1), g)

resize(span::Span, n) = Span(resize(set(span), n), coeftype(span))

zeros(span::Span) = zeros(coefficient_type(span), set(span))

function ones(span::Span)
    c = zeros(span)
    for i in eachindex(c)
        c[i] = one(coeftype(span))
    end
    c
end

# Generate a random value of type T
random_value(::Type{T}) where {T <: Number} = convert(T, rand())
random_value(::Type{Complex{T}}) where {T <: Real} = T(rand()) + im*T(rand())
random_value(::Type{T}) where {T} = rand() * one(T)

# Compute a random expansion
function rand(span::Span)
    c = zeros(span)
    for i in eachindex(c)
        c[i] = random_value(coeftype(span))
    end
    c
end

random_expansion(span::Span) = SetExpansion(set(span), rand(span))

zero(span::Span) = SetExpansion(set(span), zeros(span))

complex(span::Span) = promote_coeftype(span, complex(coeftype(span)))

real(span::Span) = similar_span(span, real(coeftype(span)))

eltype(span::Span) = error("The eltype of a span is not supported. Perhaps you meant coeftype?")

linearize_coefficients(span::Span, coef_native) = linearize_coefficients(set(span), coef_native)
linearize_coefficients!(span::Span, coef_linear, coef_native) = linearize_coefficients!(set(span), coef_linear, coef_native)

delinearize_coefficients(span::Span, coef_linear) = linearize_coefficients(set(span), coef_linear)
delinearize_coefficients!(span::Span, coef_native, coef_linear) = delinearize_coefficients!(set(span), coef_native, coef_linear)

tensorproduct(s1::Span{A}, s2::Span{A}) where {A} = span(tensorproduct(set(s1), set(s2)), A)
tensorproduct(s1::Span{A}, s2::Span{B}) where {A,B} = span(tensorproduct(set(s1), set(s2)), promote_type(A,B))

for op in (:extend, :restrict)
    @eval $op(s::Span) = Span($op(set(s)), coeftype(s))
end

native_index(s::Span, idx) = native_index(set(s), idx)
multilinear_index(s::Span, idx) = multilinear_index(set(s), idx)
linear_index(s::Span, idxn) = linear_index(set(s), idxn)

# A concrete FunctionSet has spaces associated with derivatives or antiderivatives of a certain order,
# and it should implement the following introspective functions:
# derivative_space(s::MyFunctionSet, order) = ...
# antiderivative_space(s::MyFunctionSet, order) = ...
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
derivative_space(s::FunctionSet; dim=1) = derivative_space(s, dimension_tuple(dimension(s), dim))
antiderivative_space(s::FunctionSet; dim=1) = antiderivative_space(s, dimension_tuple(dimension(s), dim))

==(s1::Span, s2::Span) = (coeftype(s1) == coeftype(s2)) && (set(s1) == set(s2))
