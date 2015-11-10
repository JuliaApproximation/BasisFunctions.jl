# generic_operators.jl


############################
# Extension and restriction
############################


# An extension operator is an operator that can be used to extend a representation in a set s1 to a
# larger set s2. The default extension operator is of type Extension with s1 and s2 as source and
# destination.

immutable Extension{SRC,DEST} <: AbstractOperator{SRC,DEST}
    src     ::  SRC
    dest    ::  DEST
end

# Default extension operator
extension_operator(s1::AbstractFunctionSet, s2::AbstractFunctionSet) = Extension(s1, s2)


# A restriction operator does the opposite of what the extension operator does.
# Loss of accuracy may result from the restriction.

immutable Restriction{SRC,DEST} <: AbstractOperator{SRC,DEST}
    src     ::  SRC
    dest    ::  DEST
end

# Default restriction operator
restriction_operator(s1::AbstractFunctionSet, s2::AbstractFunctionSet) = Restriction(s1, s2)



# Default implementation of an extension uses zero-padding of coef_src to coef_dest
function apply!(op::Extension, dest, src, coef_dest, coef_src)
    # We do too much work here, since we put all entries of coef_dest to zero. Fix later.
    fill!(coef_dest, 0)

    for i in eachindex(coef_src, coef_dest)
        coef_dest[i] = coef_src[i]
    end
end



# Default implementation of a restriction selects coef_dest from the start of coef_src
function apply!(op::Restriction, dest, src, coef_dest, coef_src)
    for i in eachindex(coef_dest, coef_src)
        coef_dest[i] = coef_src[i]
    end
end


#################
# Approximation
#################



# The approximation_operator function returns an operator that can be used to approximate
# a function in the function set. The operator maps a grid to a set of coefficients.
approximation_operator(s::AbstractFunctionSet) = println("Don't know how to approximate a function using a " * name(s))



####################
# Differentiation
####################

# The differentiation operator of a set maps an expansion in the set to an expansion of its derivative.
# The result of this operation may be an expansion in a different set. A function set can have different
# differentiation operators, with different result sets.
# For example, an expansion of Chebyshev polynomials up to degree n may map to polynomials up to degree n,
# or to polynomials up to degree n-1.

immutable Differentiation{SRC,DEST} <: AbstractOperator{SRC,DEST}
    src     ::  SRC
    dest    ::  DEST
    var     ::  Int
    order   ::  Int
end

variable(op::Differentiation) = op.var

order(op::Differentiation) = op.order


# The differentation_operator function returns an operator that can be used to differentiate
# a function in the function set.
differentiation_operator(s1::AbstractFunctionSet, s2::AbstractFunctionSet = s1, var = 1, order = 1) = Differentiation(s1, s2, var, order)


# A shortcut routine to compute the derivative of an expansion in a basis that is closed under differentiation
differentiate(src::AbstractBasis, coef) = apply(differentiation_operator(src), coef)



