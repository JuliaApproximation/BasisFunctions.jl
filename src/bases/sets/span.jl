# span.jl

"""
The span of a function set is the set of all possible expansions in that set,
where the coefficients have eltype `A`.
"""
struct Span{A,F} <: FunctionSpace
    set ::  F
end

const Span1d{A,F <: FunctionSet1d} = Span{A,F}

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

elements(span::Span) = map(s -> Span(s, coeftype(span)), elements(set(span)))
element(span::Span, i) = Span(element(set(span, i)), coeftype(span))


for op in (:length, :size, :ndims, :domaintype, :grid)
    @eval $op(span::Span) = $op(set(span))
end

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
    @eval $op(s::Span) = Span(op(set(s)), coeftype(s))
end
