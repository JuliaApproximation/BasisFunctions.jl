# special_operators.jl




"""
A CoefficientScalingOperator scales a single coefficient.
"""
immutable CoefficientScalingOperator{ELT} <: AbstractOperator{ELT}
    src     ::  FunctionSet
    dest    ::  FunctionSet
    index   ::  Int
    scalar  ::  ELT
end

function CoefficientScalingOperator(src::FunctionSet, dest::FunctionSet, index::Int, scalar::Number)
    ELT = promote_type(eltype(src), eltype(dest), typeof(scalar))
    CoefficientScalingOperator{ELT}(src, dest, index, scalar)
end

CoefficientScalingOperator(src::FunctionSet, index::Int, scalar::Number) =
    CoefficientScalingOperator(src, src, index, scalar)

promote_eltype{ELT,S}(op::CoefficientScalingOperator{ELT}, ::Type{S}) =
    CoefficientScalingOperator{S}(op.src, op.dest, op.index, S(op.scalar))

index(op::CoefficientScalingOperator) = op.index

scalar(op::CoefficientScalingOperator) = op.scalar

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



"""
A WrappedOperator has a source and destination, as well as an embedded operator with its own
source and destination. The coefficients of the source of the WrappedOperator are passed on
unaltered as coefficients of the source of the embedded operator. The resulting coefficients
of the embedded destination are returned as coefficients of the wrapping destination.
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

wrap_operator(src, dest, op::AbstractOperator) = WrappedOperator(src, dest, op)

wrap_operator(src, dest, op::WrappedOperator) = wrap_operator(src, dest, operator(op))

promote_eltype{OP,ELT,S}(op::WrappedOperator{OP,ELT}, ::Type{S}) =
    WrappedOperator(op.src, op.dest, promote_eltype(op.op, S))

operator(op::WrappedOperator) = op.op

for property in [:is_inplace, :is_diagonal]
	@eval $property(op::WrappedOperator) = $property(operator(op))
end

apply_inplace!(op::WrappedOperator, coef_srcdest) = apply_inplace!(op.op, coef_srcdest)

apply!(op::WrappedOperator, coef_dest, coef_src) = apply!(op.op, coef_dest, coef_src)

inv(op::WrappedOperator) = WrappedOperator(dest(op), src(op), inv(op.op))

ctranspose(op::WrappedOperator) = WrappedOperator(dest(op), src(op), ctranspose(op.op))

matrix!(op::WrappedOperator, a) = matrix!(op.op, a)

simplify(op::WrappedOperator) = op.op

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
apply!{T}(op::MatrixOperator, coef_dest::AbstractArray{T,1}, coef_src::AbstractArray{T,1}) =
    A_mul_B!(coef_dest, object(op), coef_src)

# Be forgiving for matrices: if the coefficients are multi-dimensional, reshape to a linear array first.
apply!{T,N1,N2}(op::MatrixOperator, coef_dest::AbstractArray{T,N1}, coef_src::AbstractArray{T,N2}) =
    apply!(op, reshape(coef_dest, length(coef_dest)), reshape(coef_src, length(coef_src)))


matrix(op::MatrixOperator) = op.object

matrix!(op::MatrixOperator, a::Array) = (a[:] = op.object)


ctranspose(op::MultiplicationOperator) = ctranspose_multiplication(op, object(op))

# This can be overriden for types of objects that do not support ctranspose
ctranspose_multiplication(op::MultiplicationOperator, object) =
    MultiplicationOperator(dest(op), src(op), ctranspose(object))

inv(op::MultiplicationOperator) = inv_multiplication(op, object(op))

# This can be overriden for types of objects that do not support inv
inv_multiplication(op::MultiplicationOperator, object) = MultiplicationOperator(dest(op), src(op), inv(object))

inv_multiplication(op::MatrixOperator, matrix) = SolverOperator(dest(op), src(op), qr(matrix))



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
    ELT = promote_type(eltype(solver), op_eltype(src, dest))
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



# An index scaling operator, used to generate weights for the polynomial scaling algorithm.
immutable IdxnScalingOperator{ELT} <: AbstractOperator{ELT}
    src     ::  FunctionSet
    order   ::  Int
    scale   ::  Function
end

IdxnScalingOperator(src::FunctionSet; order=1, scale = default_scaling_function) =
    IdxnScalingOperator{eltype(src)}(src, order, scale)

dest(op::IdxnScalingOperator) = src(op)

default_scaling_function(i) = 10.0^-4+(abs(i))+abs(i)^2+abs(i)^3
default_scaling_function(i,j) = 1+(abs(i)^2+abs(j)^2)

is_inplace(::IdxnScalingOperator) = true
is_diagonal(::IdxnScalingOperator) = true

ctranspose(op::IdxnScalingOperator) = DiagonalOperator(src(op), conj(diagonal(op)))
function apply_inplace!(op::IdxnScalingOperator, dest, src, coef_srcdest)
    ELT = eltype(op)
    for i in eachindex(coef_srcdest)
        coef_srcdest[i] *= op.scale(ELT(native_index(op.src,i)))^op.order
    end
    coef_srcdest
end

function apply_inplace!{TS1,TS2}(op::IdxnScalingOperator, dest::TensorProductSet{Tuple{TS1,TS2}}, src, coef_srcdest)
    ELT = eltype(op)
    for i in eachindex(coef_srcdest)
        indices = ind2sub(size(dest),i)
        coef_srcdest[i]*=op.scale(ELT(native_index(TS1,indices[1])),ELT(native_index(TS2,indices[2])))^op.order
    end
    coef_srcdest
end
inv(op::IdxnScalingOperator) = IdxnScalingOperator(op.src, order=op.order*-1, scale=op.scale)





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
