# special_operators.jl

"""
A CoefficientScalingOperator scales a single coefficient.
"""
struct CoefficientScalingOperator{T} <: AbstractOperator{T}
    src     ::  Span
    dest    ::  Span
    index   ::  Int
    scalar  ::  T

    function CoefficientScalingOperator{T}(src, dest, index, scalar) where {T}
        @assert length(src) == length(dest)
        new(src, dest, index, scalar)
    end
end

CoefficientScalingOperator(src::Span, index::Int, scalar::Number) =
    CoefficientScalingOperator(src, src, index, scalar)

CoefficientScalingOperator(src::Span, dest::Span, index::Int, scalar::Number) =
    CoefficientScalingOperator(eltype(scalar), src, dest, index, scalar)

function CoefficientScalingOperator(::Type{T}, src::Span, dest::Span, index::Int, scalar::Number) where {T}
    S, D, A = op_eltypes(src, dest, T)
    CoefficientScalingOperator{A}(promote_coeftype(src, S), promote_coeftype(dest, D), index, scalar)
end

similar_operator(op::CoefficientScalingOperator, ::Type{S}, src, dest) where {S} =
    CoefficientScalingOperator(S, src, dest, index(op), scalar(op))

index(op::CoefficientScalingOperator) = op.index

scalar(op::CoefficientScalingOperator) = op.scalar

is_inplace(::CoefficientScalingOperator) = true
is_diagonal(::CoefficientScalingOperator) = true

ctranspose(op::CoefficientScalingOperator) =
    CoefficientScalingOperator(dest(op), src(op), index(op), conj(scalar(op)))

inv(op::CoefficientScalingOperator) =
    CoefficientScalingOperator(dest(op), src(op), index(op), inv(scalar(op)))

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
struct WrappedOperator{OP,T} <: AbstractOperator{T}
    src     ::  Span
    dest    ::  Span
    op      ::  OP

    function WrappedOperator{OP,T}(src, dest, op) where {OP,T}
        @assert size(op,1) == length(dest)
        @assert size(op,2) == length(src)
        new(src, dest, op)
    end
end

WrappedOperator(src::Span, dest::Span, op) = WrappedOperator(eltype(op), src, dest, op)

function WrappedOperator(::Type{T}, src::Span, dest::Span, op) where {T}
    S, D, A = op_eltypes(src, dest, T)
    WrappedOperator{typeof(op),A}(promote_coeftype(src, S), promote_coeftype(dest, D), op)
end

operator(op::WrappedOperator) = op.op

function similar_operator(op::WrappedOperator, ::Type{S}, op_src, op_dest) where {S}
    subop = operator(op)
    WrappedOperator(op_src, op_dest, similar_operator(subop, S, src(subop), dest(subop)))
end

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
struct IndexRestrictionOperator{I,T} <: AbstractOperator{T}
    src         ::  Span
    dest        ::  Span
    subindices  ::  I

    function IndexRestrictionOperator{I,T}(src, dest, subindices) where {I,T}
        # Verify the lenght of subindices, but only if its length is defined
        if Base.iteratorsize(subindices) != Base.SizeUnknown()
            @assert length(dest) == length(subindices)
        end
        @assert length(src) >= length(dest)
        new(src, dest, subindices)
    end
end

IndexRestrictionOperator(src::Span, dest::Span, subindices) =
    IndexRestrictionOperator(op_eltype(src, dest), src, dest, subindices)

function IndexRestrictionOperator(::Type{T}, src, dest, subindices) where {T}
    S, D, A = op_eltypes(src, dest, T)
    IndexRestrictionOperator{typeof(subindices),A}(promote_coeftype(src, S), promote_coeftype(dest, D), subindices)
end

subindices(op::IndexRestrictionOperator) = op.subindices

similar_operator(op::IndexRestrictionOperator, ::Type{S}, src, dest) where {S} =
    IndexRestrictionOperator(S, src, dest, subindices(op))

is_diagonal(::IndexRestrictionOperator) = true

apply!(op::IndexRestrictionOperator, coef_dest, coef_src) = apply!(op, coef_dest, coef_src, subindices(op))

function apply!(op::IndexRestrictionOperator, coef_dest, coef_src, subindices)
    for (i,j) in enumerate(subindices)
        coef_dest[i] = coef_src[j]
    end
    coef_dest
end



"""
An IndexExtensionOperator embeds coefficients in a larger set based on their indices.
"""
struct IndexExtensionOperator{I,T} <: AbstractOperator{T}
    src         ::  Span
    dest        ::  Span
    subindices  ::  I

    function IndexExtensionOperator{I,T}(src, dest, subindices) where {I,T}
        @assert length(src) == length(subindices)
        @assert length(dest) >= length(src)
        new(src, dest, subindices)
    end
end

IndexExtensionOperator(src::Span, dest::Span, subindices) =
    IndexExtensionOperator(op_eltype(src, dest), src, dest, subindices)

function IndexExtensionOperator(::Type{T}, src, dest, subindices) where {T}
    S, D, A = op_eltypes(src, dest, T)
    IndexExtensionOperator{typeof(subindices),A}(promote_coeftype(src, S), promote_coeftype(dest, D), subindices)
end

subindices(op::IndexExtensionOperator) = op.subindices

similar_operator(op::IndexExtensionOperator, ::Type{S}, src, dest) where {S} =
    IndexExtensionOperator(S, src, dest, subindices(op))


is_diagonal(::IndexExtensionOperator) = true

function apply!(op::IndexExtensionOperator, coef_dest, coef_src)
    fill!(coef_dest, zero(eltype(op)))
    for (i,j) in enumerate(subindices(op))
        coef_dest[j] = coef_src[i]
    end
    coef_dest
end

ctranspose(op::IndexRestrictionOperator) =
    IndexExtensionOperator(dest(op), src(op), subindices(op))

ctranspose(op::IndexExtensionOperator) =
    IndexRestrictionOperator(dest(op), src(op), subindices(op))


"""
A MultiplicationOperator is defined by a (matrix-like) object that multiplies
coefficients. The multiplication is in-place if type parameter INPLACE is true,
otherwise it is not in-place.

An alias MatrixOperator is provided, for which type parameter ARRAY equals
Array{T,2}. In this case, multiplication is done using A_mul_B!.
"""
struct MultiplicationOperator{ARRAY,INPLACE,T} <: AbstractOperator{T}
    src     ::  Span
    dest    ::  Span
    object  ::  ARRAY

    function MultiplicationOperator{ARRAY,INPLACE,T}(src, dest, object) where {ARRAY,INPLACE,T}
        # @assert size(object,1) == length(dest)
        # @assert size(object,2) == length(src)
        new(src, dest, object)
    end
end

object(op::MultiplicationOperator) = op.object

# An MatrixOperator is defined by an actual matrix, i.e. the parameter
# ARRAY is Array{T,2}.
const MatrixOperator{T} = MultiplicationOperator{Array{T,2},false,T}

MultiplicationOperator(src::Span, dest::Span, object; inplace = false) =
    MultiplicationOperator(op_eltype(src, dest), src, dest, object; inplace = inplace)

function MultiplicationOperator(::Type{T}, src::Span, dest::Span, object; inplace = false) where {T}
    S, D, A = op_eltypes(src, dest, T)
    MultiplicationOperator{typeof(object),inplace,A}(promote_coeftype(src, S), promote_coeftype(dest, D), object)
end

MultiplicationOperator{T <: Number}(matrix::AbstractMatrix{T}) =
    MultiplicationOperator(Span(DiscreteSet(size(matrix, 2)), T), Span(DiscreteSet(size(matrix, 1)), T), matrix)

# Provide aliases for when the object is an actual matrix.
MatrixOperator(matrix::Matrix) = MultiplicationOperator(matrix)

function MatrixOperator(src::Span, dest::Span, matrix::Matrix)
    @assert size(matrix, 1) == length(dest)
    @assert size(matrix, 2) == length(src)
    MultiplicationOperator(src, dest, matrix)
end

similar_operator(op::MultiplicationOperator, ::Type{S}, src, dest) where {S} =
    MultiplicationOperator(S, src, dest, object(op))

is_inplace(op::MultiplicationOperator{ARRAY,INPLACE}) where {ARRAY,INPLACE} = INPLACE

# General definition
function apply!(op::MultiplicationOperator{ARRAY,false}, coef_dest, coef_src) where {ARRAY}
    # Note: this is very likely to allocate memory in the right hand side
    coef_dest[:] = op.object * coef_src
    coef_dest
end

# In-place definition
apply_inplace!(op::MultiplicationOperator{ARRAY,true}, coef_srcdest) where {ARRAY} =
    op.object * coef_srcdest


# Definition in terms of A_mul_B
apply!(op::MultiplicationOperator{Array{ELT1,2},false,ELT2}, coef_dest::AbstractArray{T,1}, coef_src::AbstractArray{T,1}) where {ELT1,ELT2,T} =
    A_mul_B!(coef_dest, object(op), coef_src)

# Be forgiving for matrices: if the coefficients are multi-dimensional, reshape to a linear array first.
apply!(op::MultiplicationOperator{Array{ELT1,2},false,ELT2}, coef_dest::AbstractArray{T,N1}, coef_src::AbstractArray{T,N2}) where {ELT1,ELT2,T,N1,N2} =
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

inv_multiplication(op::MatrixOperator, matrix) = SolverOperator(dest(op), src(op), qrfact(matrix))



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
struct SolverOperator{Q,T} <: AbstractOperator{T}
    src     ::  Span
    dest    ::  Span
    solver  ::  Q
end

SolverOperator(src::Span, dest::Span, solver) = SolverOperator(op_eltype(src, dest), src, dest, solver)

function SolverOperator(::Type{T}, src::Span, dest::Span, solver) where {T}
    # Note, we do not require eltype(solver) to be implemented, so we can't infer the type of the solver.
    S, D, A = op_eltypes(src, dest, T)
    SolverOperator{typeof(solver),A}(promote_coeftype(src, S), promote_coeftype(dest, D), solver)
end

# TODO: does this allocate memory? Are there (operator-specific) ways to avoid that?
function apply!(op::SolverOperator, coef_dest, coef_src)
    coef_dest[:] = op.solver \ coef_src
    coef_dest
end

ctranspose(op::SolverOperator) = warn("not implemented")


"""
A FunctionOperator applies a given function to the set of coefficients and
returns the result.
"""
struct FunctionOperator{F,T} <: AbstractOperator{T}
    src     ::  Span
    dest    ::  Span
    fun     ::  F
end

FunctionOperator(src::Span, dest::Span, fun) = FunctionOperator(op_eltype(src, dest), src, dest, fun)

function FunctionOperator(::Type{T}, src::Span, dest::Span, fun) where {T}
    S, D, A = op_eltypes(src, dest, T)
    FunctionOperator{typeof(fun),A}(promote_coeftype(src, S), promote_coeftype(dest, D), fun)
end

# Warning: this very likely allocates memory
apply!(op::FunctionOperator, coef_dest, coef_src) = apply_fun!(op, op.fun, coef_dest, coef_src)

function apply_fun!(op::FunctionOperator, fun, coef_dest, coef_src)
    coef_dest[:] = fun(coef_src)
end

ctranspose(op::FunctionOperator) = ctranspose_function(op, op.fun)

# This can be overriden for types of functions that do not support ctranspose
ctranspose_function(op::FunctionOperator, fun) =
    FunctionOperator(dest(op), src(op), ctranspose(fun))

inv(op::FunctionOperator) = inv_function(op, op.fun)

# This can be overriden for types of functions that do not support inv
inv_function(op::FunctionOperator, fun) = FunctionOperator(dest(op), src(op), inv(fun))


# An operator to flip the signs of the coefficients at uneven positions. Used in Chebyshev normalization.
struct UnevenSignFlipOperator{T} <: AbstractOperator{T}
    src     ::  Span
    dest    ::  Span
end

UnevenSignFlipOperator(src::Span, dest::Span = src) =
    UnevenSignFlipOperator(op_eltype(src,dest), src, dest)

function UnevenSignFlipOperator(::Type{T}, src::Span, dest::Span) where {T}
    S, D, A = op_eltypes(src, dest, T)
    UnevenSignFlipOperator{A}(promote_coeftype(src, S), promote_coeftype(dest, D))
end


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

diagonal{T}(op::UnevenSignFlipOperator{T}) = T[-(-1)^i for i in 1:length(src(op))]


"A linear combination of operators: val1 * op1 + val2 * op2."
struct OperatorSum{OP1 <: AbstractOperator,OP2 <: AbstractOperator,T,S} <: AbstractOperator{T}
    op1         ::  OP1
    op2         ::  OP2
    val1        ::  T
    val2        ::  T
    scratch     ::  S

    function OperatorSum{OP1,OP2,T,S}(op1::OP1, op2::OP2, val1::T, val2::T, scratch::S) where {OP1 <: AbstractOperator,OP2 <: AbstractOperator,T,S}
        # We don't enforce that source and destination of op1 and op2 are the same, but at least
        # their sizes must match.
        @assert size(src(op1)) == size(src(op2))
        @assert size(dest(op1)) == size(dest(op2))

        new(op1, op2, val1, val2, scratch)
    end
end

function OperatorSum(op1::AbstractOperator, op2::AbstractOperator, val1::Number, val2::Number)
    T = promote_type(eltype(op1), eltype(op2), typeof(val1), typeof(val2))
    c_op1 = promote_eltype(op1, T)
    c_op2 = promote_eltype(op2, T)
    scratch = zeros(dest(c_op1))
    OperatorSum{typeof(c_op1),typeof(c_op2),T,typeof(scratch)}(c_op1, c_op2, T(val1), T(val2), scratch)
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
struct LinearizationOperator{T} <: AbstractOperator{T}
    src         ::  Span
    dest        ::  Span
end

LinearizationOperator(src::Span) =
    LinearizationOperator{coeftype(src)}(src, Span(DiscreteSet(length(src)), coeftype(src)))

similar_operator(::LinearizationOperator, ::Type{S}, src) where {S} = LinearizationOperator(promote_coeftype(src, S))

apply!(op::LinearizationOperator, coef_dest, coef_src) =
    linearize_coefficients!(coef_dest, coef_src)

is_diagonal(op::LinearizationOperator) = true


"The inverse of a LinearizationOperator."
struct DelinearizationOperator{T} <: AbstractOperator{T}
    src         ::  Span
    dest        ::  Span
end

DelinearizationOperator(dest::Span) =
    DelinearizationOperator{coeftype(dest)}(Span(DiscreteSet(length(dest)), coeftype(dest)), src)

similar_operator(::DelinearizationOperator, ::Type{S}, src) where {S} = DelinearizationOperator(promote_coeftype(src, S))

apply!(op::DelinearizationOperator, coef_dest, coef_src) =
    delinearize_coefficients!(coef_dest, coef_src)

is_diagonal(op::DelinearizationOperator) = true
