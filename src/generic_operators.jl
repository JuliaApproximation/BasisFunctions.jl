# generic_operators.jl


# In this file we define the interface for the following generic functions:
# - extension_operator
# - restriction_operator
# - interpolation_operator
# - approximation_operator
# - transform_operator
# - differentiation_operator
# - evaluation_operator
# - normalization_operator
# - transform_normalization_operator

# These operators are also defined for TensorProductSet's.


############################
# Extension and restriction
############################



immutable Extension{SRC,DEST} <: AbstractOperator{SRC,DEST}
    src     ::  SRC
    dest    ::  DEST
end

"""
An extension operator is an operator that can be used to extend a representation in a set s1 to a
representation in a larger set s2. The default extension operator is of type Extension with s1 and
s2 as source and destination.
"""
extension_operator(s1::FunctionSet, s2::FunctionSet) = Extension(s1, s2)

"""
Return a suitable length to extend to, for example one such that the corresponding grids are nested
and function evaluations can be shared. The default is twice the length of the current set.
"""
extension_size(s::FunctionSet) = 2*length(s)

immutable Restriction{SRC,DEST} <: AbstractOperator{SRC,DEST}
    src     ::  SRC
    dest    ::  DEST
end


"""
A restriction operator does the opposite of what the extension operator does: it restricts
a representation in a set s1 to a representation in a smaller set s2. Loss of accuracy may result
from the restriction. The default restriction_operator is of type Restriction with sets s1 and 
s2 as source and destination.
"""
restriction_operator(s1::FunctionSet, s2::FunctionSet) = Restriction(s1, s2)


ctranspose(op::Extension) = restriction_operator(dest(op), src(op))

ctranspose(op::Restriction) = extension_operator(dest(op), src(op))




#################
# Normalization
#################


immutable NormalizationOperator{SRC,DEST} <: AbstractOperator{SRC,DEST}
    src     ::  SRC
    dest    ::  DEST
end

normalization_operator(src::FunctionSet) = NormalizationOperator(src, normalize(src))

is_inplace{O <: NormalizationOperator}(::Type{O}) = True

is_diagonal{O <: NormalizationOperator}(::Type{O}) = True




#################
# Approximation
#################


"""
A function set can implement the apply! method of a suitable TransformOperator for any known transform.
Example: a discrete transform from a set of samples on a grid to a set of expansion coefficients.
"""
immutable TransformOperator{SRC,DEST} <: AbstractOperator{SRC,DEST}
    src     ::  SRC
    dest    ::  DEST
end

# The default transform from src to dest is a TransformOperator. This may be overridden for specific source and destinations.
transform_operator(src, dest) = TransformOperator(src, dest)

# Convenience functions: automatically convert a grid to a DiscreteGridSpace
transform_operator(src::AbstractGrid, dest::FunctionSet) = transform_operator(DiscreteGridSpace(src), dest)
transform_operator(src::FunctionSet, dest::AbstractGrid) = transform_operator(src, DiscreteGridSpace(dest))

ctranspose(op::TransformOperator) = transform_operator(dest(op), src(op))

## The transform is invariant under a linear map.
#apply!(op::TransformOperator, src::LinearMappedSet, dest::LinearMappedSet, coef_dest, coef_src) =
#    apply!(op, set(src), set(dest), coef_dest, coef_src)



# Compute the interpolation matrix of the given basis on the given grid.
function interpolation_matrix(s::FunctionSet, g::AbstractGrid)
    T = promote_type(eltype(s), eltype(g))
    a = Array(T, length(g), length(s))
    interpolation_matrix!(a, s, g)
end

function interpolation_matrix(s::FunctionSet, xs::AbstractVector{AbstractVector})
    T = promote_type(eltype(s), eltype(xs))
    a = Array(T, length(xs), length(s))
    interpolation_matrix!(a, s, xs)
end

function interpolation_matrix!(a::AbstractMatrix, s::FunctionSet, g)
    n,m = size(a)
    @assert n == length(g)
    @assert m == length(s)

    for j = 1:m
        for i = 1:n
            a[i,j] = call(s, j, g[i])
        end
    end
    a
end



function interpolation_operator(s::FunctionSet)
    if has_transform(s)
        transform_normalization_operator(s) * transform_operator(grid(s), s)
    else
        SolverOperator(grid(s), s, qrfact(interpolation_matrix(s, grid(s))))
    end
end

function interpolate{N}(s::FunctionSet{N}, xs::AbstractVector{AbstractVector}, f)
    A = interpolation_matrix(s, xs)
    B = [f(x...) for x in xs]
    SetExpansion(s, A\B)
end


# Evaluation works for any set that has a grid(set) associated with it.
evaluation_operator(s::FunctionSet) = MatrixOperator(s, grid(s), interpolation_matrix(s, grid(s)))


"""
The approximation_operator function returns an operator that can be used to approximate
a function in the function set. This operator maps a grid to a set of coefficients.
"""
approximation_operator(b::FunctionSet) = interpolation_operator(b)
# The default approximation for a basis is interpolation


sample{N,T}(g::AbstractGrid{N,T}, f::Function, ELT = T) = ELT[f(x...) for x in g]

(*)(op::AbstractOperator, f::Function) = op * sample(grid(src(op)), f, eltype(op))

approximate(s::FunctionSet, f::Function) = SetExpansion(s, approximation_operator(s) * f)




####################
# Differentiation
####################

"""
The differentiation operator of a set maps an expansion in the set to an expansion of its derivative.
The result of this operation may be an expansion in a different set. A function set can have different
differentiation operators, with different result sets.
For example, an expansion of Chebyshev polynomials up to degree n may map to polynomials up to degree n,
or to polynomials up to degree n-1.
"""
immutable Differentiation{SRC,DEST} <: AbstractOperator{SRC,DEST}
    src     ::  SRC
    dest    ::  DEST
    var     ::  Int
    order   ::  Int
end

Differentiation{SRC <: FunctionSet, DEST <: FunctionSet}(src::SRC, dest::DEST = src, var = 1, order = 1) = Differentiation{SRC,DEST}(src, dest, 1, 1)

variable(op::Differentiation) = op.var

order(op::Differentiation) = op.order


"""
The differentation_operator function returns an operator that can be used to differentiate
a function in the function set, with the result as an expansion in a second set.
"""
differentiation_operator(s1::FunctionSet, s2::FunctionSet = s1, var::Int = 1, order::Int = 1) = Differentiation(s1, s2, var, order)

# With this definition below, the user may specify a single set and a variable, with or without an order
differentiation_operator(s1::FunctionSet, var::Int, order::Int...) = differentiation_operator(s1, s1, var, order...)


"""
A shortcut routine to compute the derivative of an expansion in a basis that is closed under differentiation.
"""
differentiate(src::AbstractBasis, coef) = apply(differentiation_operator(src), coef)


#####################################
# Operators for tensor product sets
#####################################

# TODO: add differentiation operator

for op in (:extension_operator, :restriction_operator, :transform_operator)
    @eval $op{TS1,TS2,SN,LEN}(s1::TensorProductSet{TS1,SN,LEN}, s2::TensorProductSet{TS2,SN,LEN}) = 
        TensorProductOperator([$op(set(s1,i),set(s2, i)) for i in 1:LEN]...)
end

for op in (:interpolation_operator, :evaluation_operator, :approximation_operator,
    :normalization_operator, :transform_normalization_operator)
    @eval $op{TS,SN,LEN}(s::TensorProductSet{TS,SN,LEN}) = 
        TensorProductOperator([$op(set(s,i)) for i in 1:LEN]...)
end


# These are not very elegant. We need a fix for it.
for op in (:extension_operator, :restriction_operator, :transform_operator, 
    :differentiation_operator, :normalization_operator)
    @eval $op{TS1,TS2,SN,LEN}(s1::TensorProductSet{TS1,SN,LEN}, s2::TensorProductSet{TS2,SN,LEN},ELT::DataType) = 
        TensorProductOperator(ELT,[$op(set(s1,i),set(s2, i)) for i in 1:LEN]...)
end

for op in (:interpolation_operator, :evaluation_operator, :approximation_operator,
    :normalization_operator, :transform_normalization_operator)
    @eval $op{TS,SN,LEN}(s::TensorProductSet{TS,SN,LEN}, ELT::DataType) = 
        TensorProductOperator(ELT,[$op(set(s,i)) for i in 1:LEN]...)
end

for op in (:extension_operator, :restriction_operator, :transform_operator)
    @eval $op{N,T}(s1::FunctionSet{N,T}, s2::FunctionSet{N,T}, ELT::DataType) = 
        $op(s1,s2) 
end

for op in (:interpolation_operator, :evaluation_operator, :approximation_operator,
    :normalization_operator)
    @eval $op{N,T}(s::FunctionSet{N,T}, ELT::DataType) = $op(s)
end
