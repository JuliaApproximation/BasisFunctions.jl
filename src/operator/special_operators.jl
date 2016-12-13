# special_operators.jl

"""
A CoefficientScalingOperator scales a single coefficient.
"""
immutable CoefficientScalingOperator{ELT} <: AbstractOperator{ELT}
    src     ::  FunctionSet
    dest    ::  FunctionSet
    index   ::  Int
    scalar  ::  ELT

    function CoefficientScalingOperator(src, dest, index, scalar)
        @assert length(src) == length(dest)
        new(src, dest, index, scalar)
    end
end

function CoefficientScalingOperator(src::FunctionSet, dest::FunctionSet, index::Int, scalar::Number)
    ELT = promote_type(eltype(src), eltype(dest), typeof(scalar))
    CoefficientScalingOperator{ELT}(src, dest, index, scalar)
end

CoefficientScalingOperator(src::FunctionSet, index::Int, scalar::Number) =
    CoefficientScalingOperator(src, src, index, scalar)

index(op::CoefficientScalingOperator) = op.index

scalar(op::CoefficientScalingOperator) = op.scalar

op_promote_eltype{ELT,S}(op::CoefficientScalingOperator{ELT}, ::Type{S}) =
    CoefficientScalingOperator{S}(promote_eltype(src(op),S), promote_eltype(dest(op),S), op.index, S(op.scalar))

is_inplace(::CoefficientScalingOperator) = true
is_diagonal(::CoefficientScalingOperator) = true

ctranspose(op::CoefficientScalingOperator) =
    CoefficientScalingOperator(dest(op), src(op), index(op), conj(scalar(op)))

inv(op::CoefficientScalingOperator) =
    CoefficientScalingOperator(dest(op), src(op), index(op), 1/scalar(op))

function matrix!(op::CoefficientScalingOperator, a)
    a[:] = 0
    for i in 1:min(size(a,1),size(a,2))
        a[i,i] = 1
    end
    a[op.index,op.index] = op.scalar
    a
end

function apply_inplace!(op::CoefficientScalingOperator, coef_srcdest)
    coef_srcdest[op.index] *= op.scalar
    coef_srcdest
end

function diagonal(op::CoefficientScalingOperator)
    diag = ones(eltype(op), length(src(op)))
    diag[index(op)] = scalar(op)
    diag
end


"""
A WrappedOperator has a source and destination, as well as an embedded operator with its own
source and destination. The coefficients of the source of the WrappedOperator are passed on
unaltered as coefficients of the source of the embedded operator. The resulting coefficients
of the embedded destination are returned as coefficients of the wrapping destination.

The purpose of this operator is to make sure that source and destinations sets of an
operator are correct, for example if a derived set returns an operator of the embedded set.
This operator can be wrapped to make sure it has the right source and destination sets, i.e.
its source and destination would correspond to the derived set, and not to the embedded set.
"""
immutable WrappedOperator{OP,ELT} <: AbstractOperator{ELT}
    src     ::  FunctionSet
    dest    ::  FunctionSet
    op      ::  OP

    function WrappedOperator(src, dest, op)
        @assert size(op,1) == length(dest)
        @assert size(op,2) == length(src)

        new(src, dest, op)
    end
end

WrappedOperator(src, dest, op::AbstractOperator) =
    WrappedOperator{typeof(op),eltype(op)}(src, dest, op)

"""
The function wrap_operator returns an operator with the given source and destination,
and with the action of the given operator. Depending on the operator, the result is
a WrappedOperator, but sometimes that can be avoided.
"""
function wrap_operator(w_src, w_dest, op::AbstractOperator)
    if (w_src == src(op)) && (w_dest == dest(op))
        op
    else
        WrappedOperator(w_src, w_dest, op)
    end
end

# No need to wrap a wrapped operator
wrap_operator(src, dest, op::WrappedOperator) = wrap_operator(src, dest, operator(op))

# No need to wrap an IdentityOperator, we can just change src and dest
# Same for a few other operators
wrap_operator(src, dest, op::IdentityOperator) = IdentityOperator(src, dest)
wrap_operator(src, dest, op::DiagonalOperator) = DiagonalOperator(src, dest, diagonal(op))
wrap_operator(src, dest, op::ScalingOperator) = ScalingOperator(src, dest, scalar(op))
wrap_operator(src, dest, op::ZeroOperator) = ZeroOperator(src, dest)

op_promote_eltype{OP,ELT,S}(op::WrappedOperator{OP,ELT}, ::Type{S}) =
    WrappedOperator(promote_eltype(src(op), S), promote_eltype(dest(op), S), promote_eltype(op.op, S))

operator(op::WrappedOperator) = op.op

for property in [:is_inplace, :is_diagonal]
	@eval $property(op::WrappedOperator) = $property(operator(op))
end

apply_inplace!(op::WrappedOperator, coef_srcdest) = apply_inplace!(op.op, coef_srcdest)

apply!(op::WrappedOperator, coef_dest, coef_src) = apply!(op.op, coef_dest, coef_src)

inv(op::WrappedOperator) = wrap_operator(dest(op), src(op), inv(op.op))

ctranspose(op::WrappedOperator) = wrap_operator(dest(op), src(op), ctranspose(op.op))

matrix!(op::WrappedOperator, a) = matrix!(op.op, a)

simplify(op::WrappedOperator) = op.op


"""
An IndexRestrictionOperator selects a subset of coefficients based on their indices.
"""
immutable IndexRestrictionOperator{I,ELT} <: AbstractOperator{ELT}
    src         ::  FunctionSet
    dest        ::  FunctionSet
    subindices  ::  I

    function IndexRestrictionOperator(src, dest, subindices)
        @assert length(dest) == length(subindices)
        @assert length(src) >= length(dest)
        new(src, dest, subindices)
    end
end

function IndexRestrictionOperator(src, dest, subindices)
    ELT = promote_type(eltype(src), eltype(dest))
    IndexRestrictionOperator{typeof(subindices),ELT}(src, dest, subindices)
end

subindices(op::IndexRestrictionOperator) = op.subindices

is_diagonal(::IndexRestrictionOperator) = true

function apply!(op::IndexRestrictionOperator, coef_dest, coef_src)
    for (i,j) in enumerate(subindices(op))
        coef_dest[i] = coef_src[j]
    end
    coef_dest
end

op_promote_eltype{I,ELT,S}(op::IndexRestrictionOperator{I,ELT}, ::Type{S}) =
    IndexRestrictionOperator{I,S}(promote_eltype(op.src, S), promote_eltype(op.dest, S), subindices(op))


"""
An IndexExtensionOperator embeds coefficients in a larger set based on their indices.
"""
immutable IndexExtensionOperator{I,ELT} <: AbstractOperator{ELT}
    src         ::  FunctionSet
    dest        ::  FunctionSet
    subindices  ::  I

    function IndexExtensionOperator(src, dest, subindices)
        @assert length(src) == length(subindices)
        @assert length(dest) >= length(src)
        new(src, dest, subindices)
    end
end

function IndexExtensionOperator(src, dest, subindices)
    ELT = promote_type(eltype(src), eltype(dest))
    IndexExtensionOperator{typeof(subindices),ELT}(src, dest, subindices)
end

subindices(op::IndexExtensionOperator) = op.subindices

is_diagonal(::IndexExtensionOperator) = true

function apply!(op::IndexExtensionOperator, coef_dest, coef_src)
    for (i,j) in enumerate(subindices(op))
        coef_dest[j] = coef_src[i]
    end
    coef_dest
end

op_promote_eltype{I,ELT,S}(op::IndexExtensionOperator{I,ELT}, ::Type{S}) =
    IndexExtensionOperator{I,S}(promote_eltype(op.src, S), promote_eltype(op.dest, S), subindices(op))

ctranspose(op::IndexRestrictionOperator) =
    IndexExtensionOperator(dest(op), src(op), subindices(op))

ctranspose(op::IndexExtensionOperator) =
    IndexRestrictionOperator(dest(op), src(op), subindices(op))


"""
A MultiplicationOperator is defined by a (matrix-like) object that multiplies
coefficients. The multiplication is in-place if type parameter INPLACE is true,
otherwise it is not in-place.

An alias MatrixOperator is provided, for which type parameter ARRAY equals
Array{ELT,2}. In this case, multiplication is done using A_mul_B!.
"""
immutable MultiplicationOperator{ARRAY,INPLACE,ELT} <: AbstractOperator{ELT}
    src     ::  FunctionSet
    dest    ::  FunctionSet
    object  ::  ARRAY

    function MultiplicationOperator(src, dest, object)
        # @assert size(object,1) == length(dest)
        # @assert size(object,2) == length(src)
        new(src, dest, object)
    end
end

object(op::MultiplicationOperator) = op.object

# An MatrixOperator is defined by an actual matrix, i.e. the parameter
# ARRAY is Array{T,2}.
typealias MatrixOperator{ELT} MultiplicationOperator{Array{ELT,2},false,ELT}

function MultiplicationOperator(src::FunctionSet, dest::FunctionSet, object; inplace = false)
    ELT = promote_type(eltype(object), op_eltype(src,dest))
    MultiplicationOperator{typeof(object),inplace,ELT}(src, dest, object)
end

MultiplicationOperator{T <: Number}(matrix::AbstractMatrix{T}) =
    MultiplicationOperator(Rn{T}(size(matrix,2)), Rn{T}(size(matrix,1)), matrix)

MultiplicationOperator{T <: Number}(matrix::AbstractMatrix{Complex{T}}) =
    MultiplicationOperator(Cn{T}(size(matrix,2)), Cn{T}(size(matrix,1)), matrix)

# Provide aliases for when the object is an actual matrix.
MatrixOperator{T}(matrix::Array{T,2}) = MultiplicationOperator(matrix)

function MatrixOperator{T}(src::FunctionSet, dest::FunctionSet, matrix::Array{T,2})
    @assert size(matrix, 1) == length(dest)
    @assert size(matrix, 2) == length(src)
    MultiplicationOperator(src, dest, matrix)
end

is_inplace{ARRAY,INPLACE}(op::MultiplicationOperator{ARRAY,INPLACE}) = INPLACE

# General definition
function apply!{ARRAY}(op::MultiplicationOperator{ARRAY,false}, coef_dest, coef_src)
    # Note: this is very likely to allocate memory in the right hand side
    coef_dest[:] = op.object * coef_src
    coef_dest
end

# In-place definition
apply_inplace!{ARRAY}(op::MultiplicationOperator{ARRAY,true}, coef_srcdest) =
    op.object * coef_srcdest


# Definition in terms of A_mul_B
apply!{ELT,T}(op::MultiplicationOperator{Array{ELT,2},false,ELT}, coef_dest::AbstractArray{T,1}, coef_src::AbstractArray{T,1}) =
    A_mul_B!(coef_dest, object(op), coef_src)

# Be forgiving for matrices: if the coefficients are multi-dimensional, reshape to a linear array first.
apply!{ELT,T,N1,N2}(op::MultiplicationOperator{Array{ELT,2},false,ELT}, coef_dest::AbstractArray{T,N1}, coef_src::AbstractArray{T,N2}) =
    apply!(op, reshape(coef_dest, length(coef_dest)), reshape(coef_src, length(coef_src)))


matrix{ELT}(op::MultiplicationOperator{Array{ELT,2},false,ELT}) = op.object

matrix!{ELT}(op::MultiplicationOperator{Array{ELT,2},false,ELT}, a::Array) = (a[:] = op.object)


ctranspose(op::MultiplicationOperator) = ctranspose_multiplication(op, object(op))

# This can be overriden for types of objects that do not support ctranspose
ctranspose_multiplication(op::MultiplicationOperator, object) =
    MultiplicationOperator(dest(op), src(op), ctranspose(object))

inv(op::MultiplicationOperator) = inv_multiplication(op, object(op))

# This can be overriden for types of objects that do not support inv
inv_multiplication(op::MultiplicationOperator, object) = MultiplicationOperator(dest(op), src(op), inv(object))

inv_multiplication{ELT}(op::MultiplicationOperator{Array{ELT,2},false,ELT}, matrix) = SolverOperator(dest(op), src(op), qrfact(matrix))



# Intercept calls to dimension_operator with a MultiplicationOperator and transfer
# to dimension_operator_multiplication. The latter can be overridden for specific
# object types.
# This is used e.g. for multivariate FFT's and friends
dimension_operator(src, dest, op::MultiplicationOperator, dim; options...) =
    dimension_operator_multiplication(src, dest, op, dim, object(op); options...)

# Default if no more specialized definition is available: make DimensionOperator
dimension_operator_multiplication(src, dest, op::MultiplicationOperator, dim, object; viewtype = VIEW_DEFAULT, options...) =
    DimensionOperator(src, dest, op, dim, viewtype)



"""
A SolverOperator wraps around a solver that is used when the SolverOperator is applied. The solver
should implement the \ operator.
Examples include a QR or SVD factorization, or a dense matrix.
"""
immutable SolverOperator{Q,ELT} <: AbstractOperator{ELT}
    src     ::  FunctionSet
    dest    ::  FunctionSet
    solver  ::  Q
end

function SolverOperator(src::FunctionSet, dest::FunctionSet, solver)
    # Note, we do not require eltype(solver) to be implemented, so we can't infer the type of the solver.
    ELT = op_eltype(src, dest)
    SolverOperator{typeof(solver),ELT}(src, dest, solver)
end

# TODO: does this allocate memory? Are there (operator-specific) ways to avoid that?
function apply!(op::SolverOperator, coef_dest, coef_src)
    coef_dest[:] = op.solver \ coef_src
    coef_dest
end


"""
A FunctionOperator applies a given function to the set of coefficients and
returns the result.
"""
immutable FunctionOperator{F,ELT} <: AbstractOperator{ELT}
    src     ::  FunctionSet
    dest    ::  FunctionSet
    fun     ::  F
end

function FunctionOperator(src::FunctionSet, dest::FunctionSet, fun)
    ELT = op_eltype(src, dest)
    FunctionOperator{typeof(fun),ELT}(src, dest, fun)
end

# Warning: this very likely allocates memory
apply!(op::FunctionOperator, coef_dest, coef_src) = apply_fun!(op, op.fun, coef_dest, coef_src)

function apply_fun!(op::FunctionOperator, fun, coef_dest, coef_src)
    coef_dest[:] = fun(coef_src)
end


# An operator to flip the signs of the coefficients at uneven positions. Used in Chebyshev normalization.
immutable UnevenSignFlipOperator{ELT} <: AbstractOperator{ELT}
    src     ::  FunctionSet
    dest    ::  FunctionSet
end

UnevenSignFlipOperator(src::FunctionSet, dest = src) =
    UnevenSignFlipOperator{op_eltype(src,dest)}(src, dest)


is_inplace(::UnevenSignFlipOperator) = true
is_diagonal(::UnevenSignFlipOperator) = true

ctranspose(op::UnevenSignFlipOperator) = op
inv(op::UnevenSignFlipOperator) = op

function apply_inplace!(op::UnevenSignFlipOperator, coef_srcdest)
    l = 1
    for i in eachindex(coef_srcdest)
        coef_srcdest[i] *= l
        l = -l
    end
    coef_srcdest
end

diagonal{ELT}(op::UnevenSignFlipOperator{ELT}) = ELT[-(-1)^i for i in 1:length(src(op))]


"A linear combination of operators: val1 * op1 + val2 * op2."
immutable OperatorSum{OP1 <: AbstractOperator,OP2 <: AbstractOperator,ELT,S} <: AbstractOperator{ELT}
    op1         ::  OP1
    op2         ::  OP2
    val1        ::  ELT
    val2        ::  ELT
    scratch     ::  S

    function OperatorSum(op1, op2, val1, val2, scratch)
        # We don't enforce that source and destination of op1 and op2 are the same, but at least
        # their sizes must match.
        @assert size(src(op1)) == size(src(op2))
        @assert size(dest(op1)) == size(dest(op2))

        new(op1, op2, val1, val2, scratch)
    end
end

function OperatorSum(op1::AbstractOperator, op2::AbstractOperator, val1::Number, val2::Number)
    ELT = promote_type(eltype(op1), eltype(op2), typeof(val1), typeof(val2))
    scratch = zeros(ELT,dest(op1))
    OperatorSum{typeof(op1),typeof(op2),ELT,typeof(scratch)}(op1, op2, ELT(val1), ELT(val2), scratch)
end

src(op::OperatorSum) = src(op.op1)

dest(op::OperatorSum) = dest(op.op1)

ctranspose(op::OperatorSum) = OperatorSum(ctranspose(op.op1), ctranspose(op.op2), conj(op.val1), conj(op.val2))

is_diagonal(op::OperatorSum) = is_diagonal(op.op1) && is_diagonal(op.op2)


apply_inplace!(op::OperatorSum, dest, src, coef_srcdest) =
    apply_sum_inplace!(op, op.op1, op.op2, coef_srcdest)

function apply_sum_inplace!(op::OperatorSum, op1::AbstractOperator, op2::AbstractOperator, coef_srcdest)
    scratch = op.scratch

    apply!(op1, scratch, coef_srcdest)
    apply!(op2, coef_srcdest)

    for i in eachindex(coef_srcdest)
        coef_srcdest[i] = op.val1 * scratch[i] + op.val2 * coef_srcdest[i]
    end
    coef_srcdest
end

apply!(op::OperatorSum, dest, src, coef_dest, coef_src) = apply_sum!(op, op.op1, op.op2, coef_dest, coef_src)

function apply_sum!(op::OperatorSum, op1::AbstractOperator, op2::AbstractOperator, coef_dest, coef_src)
    scratch = op.scratch

    apply!(op1, scratch, coef_src)
    apply!(op2, coef_dest, coef_src)

    for i in eachindex(coef_dest)
        coef_dest[i] = op.val1 * scratch[i] + op.val2 * coef_dest[i]
    end
    coef_dest
end

(+)(op1::AbstractOperator, op2::AbstractOperator) = OperatorSum(op1, op2, 1, 1)
(-)(op1::AbstractOperator, op2::AbstractOperator) = OperatorSum(op1, op2, 1, -1)

"""
An operator that calls linearize on a native representation of a set, returning
a Vector with the length of the set. For example, one can not apply a matrix
to a non-arraylike representation, hence the representation has to be linearized
first.
"""
immutable LinearizationOperator{ELT} <: AbstractOperator{ELT}
    src         ::  FunctionSet
    dest        ::  FunctionSet
end

LinearizationOperator(src::FunctionSet, ELT = eltype(src)) =
    LinearizationOperator{ELT}(src, DiscreteSet{ELT}(length(src)))

apply!(op::LinearizationOperator, coef_dest, coef_src) =
    linearize_coefficients!(coef_dest, coef_src)

is_diagonal(op::LinearizationOperator) = true


"The inverse of a LinearizationOperator."
immutable DelinearizationOperator{ELT} <: AbstractOperator{ELT}
    src         ::  FunctionSet
    dest        ::  FunctionSet
end

DelinearizationOperator(dest::FunctionSet, ELT = eltype(dest)) =
    LinearizationOperator{ELT}(DiscreteSet{ELT}(length(src)), dest)

apply!(op::DelinearizationOperator, coef_dest, coef_src) =
    delinearize_coefficients!(coef_dest, coef_src)

is_diagonal(op::DelinearizationOperator) = true
