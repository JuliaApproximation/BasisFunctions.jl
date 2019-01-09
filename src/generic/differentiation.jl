
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
struct Differentiation{SRC <: Dictionary,DEST <: Dictionary,T} <: DictionaryOperator{T}
    src     ::  SRC
    dest    ::  DEST
    order   ::  Int
end

Differentiation(src::Dictionary, dest::Dictionary, order) =
    Differentiation{typeof(src),typeof(dest),op_eltype(src,dest)}(src, dest, order)


order(op::Differentiation) = op.order


"""
The differentation_operator function returns an operator that can be used to differentiate
a function in the dictionary, with the result as an expansion in a second dictionary.
"""
function differentiation_operator(s1::Dictionary, s2::Dictionary, order; options...)
    @assert has_derivative(s1)
    Differentiation(s1, s2, order)
end

# Default if no order is specified
function differentiation_operator(s1::Dictionary1d, order=1; options...)
    s2 = derivative_dict(s1, order; options...)
    differentiation_operator(s1, s2, order; options...)
end

differentiation_operator(s1::Dictionary; dim=1, options...) =
    differentiation_operator(s1, dimension_tuple(Val(dimension(s1)), dim))

function differentiation_operator(s1::Dictionary, order; options...)
    s2 = derivative_dict(s1, order)
    differentiation_operator(s1, s2, order; options...)
end


"""
The antidifferentiation operator of a dictionary maps an expansion in the dictionary to an
expansion of its antiderivative. The result may be an expansion in a different
dictionary. A dictionary can have different antidifferentiation operators,
with different result dictionaries.
"""
struct AntiDifferentiation{SRC <: Dictionary,DEST <: Dictionary,T} <: DictionaryOperator{T}
    src     ::  SRC
    dest    ::  DEST
    order   ::  Int
end

AntiDifferentiation(src::Dictionary, dest::Dictionary, order) =
    AntiDifferentiation{typeof(src),typeof(dest),op_eltype(src,dest)}(src, dest, order)

order(op::AntiDifferentiation) = op.order

# The default antidifferentiation implementation is differentiation with a negative order (such as for Fourier)
# If the destination contains a DC coefficient, it is zero by default.

"""
The antidifferentiation_operator function returns an operator that can be used to find the antiderivative
of a function in the dictionary, with the result an expansion in a second dictionary.
"""
function antidifferentiation_operator(s1::Dictionary, s2::Dictionary, order; options...)
    @assert has_antiderivative(s1)
    AntiDifferentiation(s1, s2, order)
end

# Default if no order is specified
function antidifferentiation_operator(s1::Dictionary1d, order=1; options...)
    s2 = antiderivative_dict(s1, order)
    antidifferentiation_operator(s1, s2, order; options...)
end

antidifferentiation_operator(s1::Dictionary; dim=1, options...) = antidifferentiation_operator(s1, dimension_tuple(Val(dimension(s1)), dim))

function antidifferentiation_operator(s1::Dictionary, order; options...)
    s2 = antiderivative_dict(s1, order)
    antidifferentiation_operator(s1, s2, order; options...)
end

"""
The pseudodifferential_operator function returns an operator representing a function of the
derivative operator.
"""
struct PseudodifferentialOperator{SRC <: Dictionary,DEST <: Dictionary, T} <: DictionaryOperator{T}
    src     ::  SRC
    dest    ::  DEST
    symbol  ::  Function
end

PseudodifferentialOperator(src::Dictionary, dest::Dictionary, symbol::Function) =
    PseudodifferentialOperator{typeof(src),typeof(dest),op_eltype(src,dest)}(src, dest, symbol)

symbol(op::PseudodifferentialOperator) = op.symbol

function pseudodifferential_operator(s1::Dictionary, s2::Dictionary, symbol::Function; options...)
    @assert has_derivative(s1)
    PseudodifferentialOperator(s1, s2, symbol)
end
