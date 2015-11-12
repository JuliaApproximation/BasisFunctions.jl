# generic_operators.jl


# In this file we define default implementations for the following generic functions:
# - extension_operator
# - restriction_operator
# - interpolation_operator
# - approximation_operator
# - transform_operator
# - differentiation_operator
# - evaluation_operator

# These operators are also defined for TensorProductSet's.
# TODO: We may have to rethink some of them, perhaps we can remove a few.


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

    for i in eachindex(coef_src)
        coef_dest[i] = coef_src[i]
    end
end



# Default implementation of a restriction selects coef_dest from the start of coef_src
function apply!(op::Restriction, dest, src, coef_dest, coef_src)
    for i in eachindex(coef_dest)
        coef_dest[i] = coef_src[i]
    end
end



#################
# Approximation
#################



# The approximation_operator function returns an operator that can be used to approximate
# a function in the function set. The operator maps a grid to a set of coefficients.
approximation_operator(s::AbstractFunctionSet) = println("Don't know how to approximate a function using a " * name(s))


# A function set can implement the apply! method of a suitable TransformOperator for any known transform.
# Example: a discrete transform from a set of samples on a grid to a set of expansion coefficients.
immutable TransformOperator{SRC,DEST} <: AbstractOperator{SRC,DEST}
    src     ::  SRC
    dest    ::  DEST
end

# The default transform from src to dest is a TransformOperator. This may be overridden for specific source and destinations.
transform_operator(src, dest) = TransformOperator(src, dest)

# Convenience functions: automatically convert a grid to a DiscreteGridSpace
transform_operator(src::AbstractGrid, dest::AbstractFunctionSet) = transform_operator(DiscreteGridSpace(src), dest)
transform_operator(src::AbstractFunctionSet, dest::AbstractGrid) = transform_operator(src, DiscreteGridSpace(dest))

ctranspose(op::TransformOperator) = transform_operator(dest(op), src(op))

## The transform is invariant under a linear map.
#apply!(op::TransformOperator, src::LinearMappedSet, dest::LinearMappedSet, coef_dest, coef_src) =
#    apply!(op, set(src), set(dest), coef_dest, coef_src)



# Compute the interpolation matrix of the given basis on the given grid.
function interpolation_matrix(b::AbstractBasis, g::AbstractGrid, T = eltype(b))
    a = Array(T, length(g), length(b))
    interpolation_matrix!(b, g, a)
    a
end

function interpolation_matrix!{N,T}(b::AbstractBasis{N,T}, g::AbstractGrid{N,T}, a::AbstractArray)
    n = size(a,1)
    m = size(a,2)
    @assert n == length(g)
    @assert m == length(b)

    x_i = Array(T,N)
    for j = 1:m
        for i = 1:n
            a[i,j] = call(b, j, x_i...)
        end
    end
end


function interpolation_matrix!{T}(b::AbstractBasis1d{T}, g::AbstractGrid1d{T}, a::AbstractArray)
    n = size(a,1)
    m = size(a,2)
    @assert n == length(g)
    @assert m == length(b)

    for j = 1:m
        for i = 1:n
            a[i,j] = call(b, j, g[i])
        end
    end
end


interpolation_operator(b::AbstractBasis) = SolverOperator(grid(b), b, qrfact(interpolation_matrix(b, grid(b))))

# Evaluation works for any set that has a grid(set) associated with it.
evaluation_operator(s::AbstractFunctionSet) = MatrixOperator(s, grid(s), interpolation_matrix(s, grid(s)))


# The default approximation for a basis is interpolation
approximation_operator(b::AbstractBasis) = interpolation_operator(b)





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

Differentiation{SRC <: AbstractFunctionSet, DEST <: AbstractFunctionSet}(src::SRC, dest::DEST = src, var = 1, order = 1) = Differentiation{SRC,DEST}(src, dest, 1, 1)

variable(op::Differentiation) = op.var

order(op::Differentiation) = op.order


# The differentation_operator function returns an operator that can be used to differentiate
# a function in the function set.
differentiation_operator(s1::AbstractFunctionSet, s2::AbstractFunctionSet = s1, var = 1, order = 1) = Differentiation(s1, s2, var, order)


# A shortcut routine to compute the derivative of an expansion in a basis that is closed under differentiation
differentiate(src::AbstractBasis, coef) = apply(differentiation_operator(src), coef)




#####################################
# Operators for tensor product sets
#####################################


for op in (:extension_operator, :restriction_operator, :approximation_operator, 
    :interpolation_operator, :evaluation_operator, :differentiation_operator,
    :transform_operator)
    @eval $op{TS1,TS2,SN,LEN}(s1::TensorProductSet{TS1,SN,LEN}, s2::TensorProductSet{TS2,SN,LEN}) = 
        TensorProductOperator([$op(set(s1,i),set(s2, i)) for i in 1:LEN]...)
end

