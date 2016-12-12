# differentiation.jl

##################################################
# Generic differentiation and antidifferentiation
##################################################

"""
The differentiation operator of a set maps an expansion in the set to an expansion of its derivative.
The result of this operation may be an expansion in a different set. A function set can have different
differentiation operators, with different result sets.
For example, an expansion of Chebyshev polynomials up to degree n may map to polynomials up to degree n,
or to polynomials up to degree n-1.
"""
immutable Differentiation{SRC <: FunctionSet,DEST <: FunctionSet,ELT} <: AbstractOperator{ELT}
    src     ::  SRC
    dest    ::  DEST
    order   ::  Int
end

Differentiation(src::FunctionSet, dest::FunctionSet, order) =
    Differentiation{typeof(src),typeof(dest),op_eltype(src,dest)}(src, dest, order)


order(op::Differentiation) = op.order


"""
The differentation_operator function returns an operator that can be used to differentiate
a function in the function set, with the result as an expansion in a second set.
"""
function differentiation_operator(s1::FunctionSet, s2::FunctionSet, order; options...)
    @assert has_derivative(s1)
    Differentiation(s1, s2, order)
end

# Default if no order is specified
function differentiation_operator(s1::FunctionSet1d, order=1; options...)
    s2 = derivative_set(s1, order; options...)
    differentiation_operator(s1, s2, order; options...)
end
differentiation_operator(s1::FunctionSet; dim=1, options...) = differentiation_operator(s1, dimension_tuple(ndims(s1), dim))

function differentiation_operator(s1::FunctionSet, order; options...)
    s2 = derivative_set(s1, order)
    differentiation_operator(s1, s2, order; options...)
end


"""
The antidifferentiation operator of a set maps an expansion in the set to an expansion of its
antiderivative. The result of this operation may be an expansion in a different set. A function set
can have different antidifferentiation operators, with different result sets.
"""
immutable AntiDifferentiation{SRC <: FunctionSet,DEST <: FunctionSet,ELT} <: AbstractOperator{ELT}
    src     ::  SRC
    dest    ::  DEST
    order   ::  Int
end

AntiDifferentiation(src::FunctionSet, dest::FunctionSet, order) =
    AntiDifferentiation{typeof(src),typeof(dest),op_eltype(src,dest)}(src, dest, order)

order(op::AntiDifferentiation) = op.order

# The default antidifferentiation implementation is differentiation with a negative order (such as for Fourier)
# If the destination contains a DC coefficient, it is zero by default.

"""
The antidifferentiation_operator function returns an operator that can be used to find the antiderivative
of a function in the function set, with the result an expansion in a second set.
"""
function antidifferentiation_operator(s1::FunctionSet, s2::FunctionSet, order; options...)
    @assert has_antiderivative(s1)
    AntiDifferentiation(s1, s2, order)
end

# Default if no order is specified
function antidifferentiation_operator(s1::FunctionSet1d, order=1; options...)
    s2 = antiderivative_set(s1, order)
    antidifferentiation_operator(s1, s2, order; options...)
end

antidifferentiation_operator(s1::FunctionSet; dim=1, options...) = antidifferentiation_operator(s1, dimension_tuple(ndims(s1), dim))

function antidifferentiation_operator(s1::FunctionSet, order; options...)
    s2 = antiderivative_set(s1, order)
    antidifferentiation_operator(s1, s2, order; options...)
end
