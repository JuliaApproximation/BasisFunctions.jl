# differentiation.jl

##################################################
# Generic differentiation and antidifferentiation
##################################################

"""
The differentiation operator of a dictionary maps an expansion in a dictionary to an
expansion of its derivative. The result may be an expansion in a different dictionary.
A dictionary can support different differentiation operators, with different
result dicts. For example, an expansion of Chebyshev polynomials up to degree n
may map to polynomials up to degree n, or to polynomials up to degree n-1.
"""
struct Differentiation{SRC <: Span,DEST <: Span,T} <: AbstractOperator{T}
    src     ::  SRC
    dest    ::  DEST
    order   ::  Int
end

Differentiation(src::Span, dest::Span, order) =
    Differentiation{typeof(src),typeof(dest),op_eltype(src,dest)}(src, dest, order)


order(op::Differentiation) = op.order


"""
The differentation_operator function returns an operator that can be used to differentiate
a function in the dictionary, with the result as an expansion in a second dictionary.
"""
function differentiation_operator(s1::Span, s2::Span, order; options...)
    @assert has_derivative(s1)
    Differentiation(s1, s2, order)
end

# Default if no order is specified
function differentiation_operator(s1::Span1d, order=1; options...)
    s2 = derivative_space(s1, order; options...)
    differentiation_operator(s1, s2, order; options...)
end

differentiation_operator(s1::Span; dim=1, options...) =
    differentiation_operator(s1, dimension_tuple(dimension(dictionary(s1)), dim))

function differentiation_operator(s1::Span, order; options...)
    s2 = derivative_space(s1, order)
    differentiation_operator(s1, s2, order; options...)
end


"""
The antidifferentiation operator of a dictionary maps an expansion in the dictionary to an
expansion of its antiderivative. The result may be an expansion in a different
dictionary. A dictionary can have different antidifferentiation operators,
with different result dictionaries.
"""
struct AntiDifferentiation{SRC <: Span,DEST <: Span,T} <: AbstractOperator{T}
    src     ::  SRC
    dest    ::  DEST
    order   ::  Int
end

AntiDifferentiation(src::Span, dest::Span, order) =
    AntiDifferentiation{typeof(src),typeof(dest),op_eltype(src,dest)}(src, dest, order)

order(op::AntiDifferentiation) = op.order

# The default antidifferentiation implementation is differentiation with a negative order (such as for Fourier)
# If the destination contains a DC coefficient, it is zero by default.

"""
The antidifferentiation_operator function returns an operator that can be used to find the antiderivative
of a function in the dictionary, with the result an expansion in a second dictionary.
"""
function antidifferentiation_operator(s1::Span, s2::Span, order; options...)
    @assert has_antiderivative(s1)
    AntiDifferentiation(s1, s2, order)
end

# Default if no order is specified
function antidifferentiation_operator(s1::Span1d, order=1; options...)
    s2 = antiderivative_space(s1, order)
    antidifferentiation_operator(s1, s2, order; options...)
end

antidifferentiation_operator(s1::Span; dim=1, options...) = antidifferentiation_operator(s1, dimension_tuple(dimension(dictionary(s1)), dim))

function antidifferentiation_operator(s1::Span, order; options...)
    s2 = antiderivative_space(s1, order)
    antidifferentiation_operator(s1, s2, order; options...)
end
