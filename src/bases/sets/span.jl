# span.jl

"""
The span of a function set is the set of all possible expansions in that set,
where the coefficients have eltype `A`.
"""
struct Span{A,F}
    set ::  F
end

Span(set::FunctionSet, ::Type{A} = coefficient_type(set)) where {A} = Span{A,typeof(set)}(set)

set(s::Span) = s.set

coefficient_type(::Span{A,F}) where {A,F} = A

coeftype = coefficient_type

# The difference between similar_span and promote_coeftype is that similar_span
# sets the coefficient type of the new span to the given argument, whereas
# promote_coeftype sets the coefficient type to promote_type of the old and new
# coefficient types.
similar_span(span::Span{A,F}, ::Type{A}) where {A,F} = span
similar_span(span::Span{A,F}, ::Type{B}) where {A,B,F} = Span(set(span), B)

promote_coeftype(span::Span{A,F}, ::Type{A}) where {A,F} = span
promote_coeftype(span::Span{A,F}, ::Type{B}) where {A,B,F} = Span(set(span), promote_type(A,B))

domaintype(s::Span) = domaintype(set(s))

# What is the rangetype of a span? It depends on the type of the coefficients,
# and on the rangetype of the set.
rangetype(span::Span) = _rangetype(coefficient_type(span), rangetype(set(span)))
# - When the types are the same, it is the result
_rangetype(::Type{T}, ::Type{T}) where {T <: Number} = T
# - the coefficient types are complex and the set itself is real
_rangetype(::Type{Complex{T}}, ::Type{T}) where {T <: Number} = Complex{T}
# Default fallback
_rangetype(::Type{A}, ::Type{Z}) where {Z,A} = typeof(zero(A) * zero(Z))

elements(span::Span) = map(s -> Span(s, coeftype(span)), elements(set(span)))
element(span::Span, i) = Span(element(set(span, i)), coeftype(span))


for op in (:length, :size, :ndims)
    @eval $op(span::Span) = $op(set(span))
end

zeros(span::Span) = zeros(coefficient_type(span), set(span))

function ones(span::Span)
    c = zeros(span)
    for i in eachindex(c)
        c[i] = one(coeftype(span))
    end
    c
end

zero(span::Span) = Expansion(set(span), zeros(span))

complex(span::Span) = promote_coeftype(span, complex(coeftype(span)))

real(span::Span) = similar_span(span, real(coeftype(span)))

eltype(span::Span) = error("The eltype of a span is not supported. Perhaps you meant coeftype?")

linearize_coefficients(span::Span, coef_native) = linearize_coefficients(set(span), coef_native)
linearize_coefficients!(span::Span, coef_linear, coef_native) = linearize_coefficients!(set(span), coef_linear, coef_native)

delinearize_coefficients(span::Span, coef_linear) = linearize_coefficients(set(span), coef_linear)
delinearize_coefficients!(span::Span, coef_native, coef_linear) = delinearize_coefficients!(set(span), coef_native, coef_linear)
