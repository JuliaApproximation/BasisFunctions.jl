
"""
A MultiplicationOperator is defined by a (matrix-like) object that multiplies
coefficients. The multiplication is in-place if type parameter INPLACE is true,
otherwise it is not in-place.
"""
struct MultiplicationOperator{T,ARRAY,INPLACE} <: DictionaryOperator{T}
    src     ::  Dictionary
    dest    ::  Dictionary
    object  ::  ARRAY

    function MultiplicationOperator{T,ARRAY,INPLACE}(src, dest, object) where {T,ARRAY,INPLACE}
        # @assert size(object,1) == length(dest)
        # @assert size(object,2) == length(src)
        new(src, dest, object)
    end
end

object(op::MultiplicationOperator) = op.object

MultiplicationOperator(args...; options...) =
    MultiplicationOperator{deduce_eltype(args...)}(args...; options...)

MultiplicationOperator{T}(src::Dictionary, dest::Dictionary, object; inplace = false) where {T} =
    MultiplicationOperator{T,typeof(object),inplace}(src, dest, object)

MultiplicationOperator(matrix::AbstractMatrix{T}) where {T <: Number} =
    MultiplicationOperator(DiscreteVectorDictionary{T}(size(matrix, 2)), DiscreteVectorDictionary{T}(size(matrix, 1)), matrix)

similar_operator(op::MultiplicationOperator{S,ARRAY,INPLACE}, src, dest) where {S,ARRAY,INPLACE} =
    MultiplicationOperator(src, dest, object(op); inplace=INPLACE)

unsafe_wrap_operator(src, dest, op::MultiplicationOperator) =
    similar_operator(op, src, dest)

isinplace(op::MultiplicationOperator{T,ARRAY,INPLACE}) where {T,ARRAY,INPLACE} = INPLACE

# We pass the object as an additional variable so we can dispatch on it.
# We only intercept non-inplace operators, as the in-place operators are
# intercept higher up where apply_inplace! ends up being called instead
apply!(op::MultiplicationOperator{T,ARRAY,false}, coef_dest, coef_src) where {T,ARRAY} =
    _apply!(op, coef_dest, coef_src, op.object)

# General definition
function _apply!(op::MultiplicationOperator{T,ARRAY,false}, coef_dest, coef_src, object) where {T,ARRAY}
    # Note: this is very likely to allocate memory in the right hand side
    coef_dest[:] = object * coef_src
    coef_dest
end


# Definition in terms of mul! for vectors
_apply!(op::MultiplicationOperator{ELT2,Array{ELT1,2},false}, coef_dest::AbstractVector{T}, coef_src::AbstractVector{T}, object) where {ELT1,ELT2,T} =
    mul!(coef_dest, object, coef_src)

# Be forgiving for matrices: if the coefficients are multi-dimensional, reshape to a linear array first.
_apply!(op::MultiplicationOperator{ELT2,Array{ELT1,2},false}, coef_dest::AbstractArray{T,N1}, coef_src::AbstractArray{T,N2}, object) where {ELT1,ELT2,T,N1,N2} =
    apply!(op, reshape(coef_dest, length(coef_dest)), reshape(coef_src, length(coef_src)))

# In-place definition
# We pass the object as an additional variable so we can dispatch on it
apply_inplace!(op::MultiplicationOperator{T,ARRAY,true}, coef_srcdest) where {T,ARRAY} =
    _apply_inplace!(op, coef_srcdest, op.object)

_apply_inplace!(op::MultiplicationOperator{T,ARRAY,true}, coef_srcdest, object) where {T,ARRAY} =
    object * coef_srcdest


# TODO: introduce unsafe_matrix here, and make a copy otherwise
# matrix(op::MatrixOperator) = op.object
#
# matrix!(op::MatrixOperator, a::Array) = (a[:] = op.object)


adjoint(op::MultiplicationOperator) = adjoint_multiplication(op, object(op))
# This can be overriden for types of objects that do not support adjoint
adjoint_multiplication(op::MultiplicationOperator{T,A,INPLACE}, object) where {T,A,INPLACE} =
    MultiplicationOperator(dest(op), src(op), adjoint(object); inplace=INPLACE)

adjoint_multiplication(op::MultiplicationOperator, object::Array{T,2}) where {T} =
    # We copy the adjoint in order to avoid storing an Adjoint type
    MultiplicationOperator(dest(op), src(op), copy(adjoint(object)))

inv(op::MultiplicationOperator) = inv_multiplication(op, object(op))

# This can be overriden for types of objects that do not support inv
inv_multiplication(op::MultiplicationOperator{T,A,INPLACE}, object) where {T,A,INPLACE} =
    MultiplicationOperator(dest(op), src(op), inv(object); inplace=INPLACE)

Display.displaysymbol(op::MultiplicationOperator) = _symbol(op, object(op))
_symbol(op::MultiplicationOperator, object) = "M"
_symbol(op::MultiplicationOperator, object::FFTW.Plan) = "FFT"






"""
A FunctionOperator applies a given function to the set of coefficients and
returns the result.
"""
struct FunctionOperator{T,F} <: DictionaryOperator{T}
    src     ::  Dictionary
    dest    ::  Dictionary
    fun     ::  F
end

FunctionOperator(args...) = FunctionOperator{deduce_eltype(args...)}(args...)

FunctionOperator{T}(src, dest, fun) where {T} =
    FunctionOperator{T,typeof(fun)}(src, dest, fun)

similar_operator(op::FunctionOperator{T,F}, src, dest) where {T,F} =
    FunctionOperator{promote_type(T,operatoreltype(src,dest)),F}(src, dest, op.fun)

unsafe_wrap_operator(src, dest, op::FunctionOperator) =
    similar_operator(op, src, dest)

# Warning: this very likely allocates memory
apply!(op::FunctionOperator, coef_dest, coef_src) = apply_fun!(op, op.fun, coef_dest, coef_src)

function apply_fun!(op::FunctionOperator, fun, coef_dest, coef_src)
    coef_dest[:] = fun(coef_src)
end

adjoint(op::FunctionOperator) = adjoint_function(op, op.fun)
# This can be overriden for types of functions that do not support adjoint
adjoint_function(op::FunctionOperator, fun) =
    FunctionOperator(dest(op), src(op), adjoint(fun))

inv(op::FunctionOperator) = inv_function(op, op.fun)

# This can be overriden for types of functions that do not support inv
inv_function(op::FunctionOperator, fun) = FunctionOperator(dest(op), src(op), inv(fun))

string(op::FunctionOperator) = "Function " * string(op.fun)



"A linear combination of operators: val1 * op1 + val2 * op2."
struct OperatorSum{T,OP1 <: DictionaryOperator,OP2 <: DictionaryOperator,S} <: DictionaryOperator{T}
    op1         ::  OP1
    op2         ::  OP2
    val1        ::  T
    val2        ::  T
    scratch     ::  S

    function OperatorSum{T,OP1,OP2,S}(op1::OP1, op2::OP2, val1::T, val2::T, scratch::S) where {OP1 <: DictionaryOperator,OP2 <: DictionaryOperator,T,S}
        # We don't enforce that source and destination of op1 and op2 are the same, but at least
        # their sizes and coefficient types must match.
        @assert size(src(op1)) == size(src(op2))
        @assert size(dest(op1)) == size(dest(op2))
        @assert coefficienttype(src(op1)) == coefficienttype(src(op2))
        @assert coefficienttype(dest(op1)) == coefficienttype(dest(op2))

        new(op1, op2, val1, val2, scratch)
    end
end

function OperatorSum{T}(op1::DictionaryOperator, op2::DictionaryOperator, val1::Number, val2::Number) where {T}
    scratch = zeros(T, dest(op1))
    OperatorSum{T,typeof(op1),typeof(op2),typeof(scratch)}(op1, op2, T(val1), T(val2), scratch)
end

OperatorSum(op1::DictionaryOperator, op2::DictionaryOperator, val1::Number, val2::Number;
            T=promote_type(eltype(op1), eltype(op2), typeof(val1), typeof(val2))) =
    OperatorSum{T}(op1, op2, val1, val2)

src(op::OperatorSum) = src(op.op1)

dest(op::OperatorSum) = dest(op.op1)

adjoint(op::OperatorSum) = OperatorSum(adjoint(op.op1), adjoint(op.op2), conj(op.val1), conj(op.val2))

pinv(op::OperatorSum) = OperatorSum(pinv(op.op1), pinv(op.op2), op.val1, op.val2)

isdiag(op::OperatorSum) = isdiag(op.op1) && isdiag(op.op2)


apply_inplace!(op::OperatorSum, coef_srcdest) =
    apply_sum_inplace!(op, op.op1, op.op2, coef_srcdest)

function apply_sum_inplace!(op::OperatorSum, op1::DictionaryOperator, op2::DictionaryOperator, coef_srcdest)
    scratch = op.scratch

    apply!(op1, scratch, coef_srcdest)
    apply!(op2, coef_srcdest)

    for i in eachindex(coef_srcdest)
        coef_srcdest[i] = op.val1 * scratch[i] + op.val2 * coef_srcdest[i]
    end
    coef_srcdest
end

apply!(op::OperatorSum, coef_dest, coef_src) = apply_sum!(op, op.op1, op.op2, coef_dest, coef_src)

function apply_sum!(op::OperatorSum, op1::DictionaryOperator, op2::DictionaryOperator, coef_dest, coef_src)
    scratch = op.scratch

    apply!(op1, scratch, coef_src)
    apply!(op2, coef_dest, coef_src)

    for i in eachindex(coef_dest)
        coef_dest[i] = op.val1 * scratch[i] + op.val2 * coef_dest[i]
    end
    coef_dest
end

components(op::OperatorSum) = (op.op1,op.op2)

(+)(op1::DictionaryOperator, op2::DictionaryOperator) = OperatorSum(op1, op2, 1, 1)
(-)(op1::DictionaryOperator, op2::DictionaryOperator) = OperatorSum(op1, op2, 1, -1)

(+)(I1::UniformScaling, op2::DictionaryOperator) = (+)(ScalingOperator(I1.λ, src(op2), dest(op2)), op2)
(-)(I1::UniformScaling, op2::DictionaryOperator) = (-)(ScalingOperator(I1.λ, src(op2), dest(op2)), op2)
(+)(op1::DictionaryOperator, I2::UniformScaling) = (+)(op1, ScalingOperator(I2.λ, src(op1), dest(op1)))
(-)(op1::DictionaryOperator, I2::UniformScaling) = (-)(op1, ScalingOperator(I2.λ, src(op1), dest(op1)))


Display.combinationsymbol(op::OperatorSum) = Display.Symbol('+')
Display.displaystencil(op::OperatorSum) = composite_displaystencil(op)


"""
An operator that calls linearize on a native representation of a set, returning
a Vector with the length of the set. For example, one can not apply a matrix
to a non-arraylike representation, hence the representation has to be linearized
first.
"""
struct LinearizationOperator{T} <: DictionaryOperator{T}
    src         ::  Dictionary
    dest        ::  Dictionary
end

LinearizationOperator(dicts::Dictionary...) =
    LinearizationOperator{operatoreltype(dicts...)}(dicts...)

LinearizationOperator{T}(src::Dictionary) where {T} =
    LinearizationOperator{T}(src, DiscreteVectorDictionary{coefficienttype(src)}(length(src)))

similar_operator(::LinearizationOperator{T}, src, dest) where {T} =
    LinearizationOperator{promote_type(T,operatoreltype(src,dest))}(src, dest)

apply!(op::LinearizationOperator, coef_dest, coef_src) =
    linearize_coefficients!(coef_dest, coef_src)

isdiag(op::LinearizationOperator) = true

Base.adjoint(op::LinearizationOperator{T}) where {T} =
    DelinearizationOperator{T}(dest(op), src(op))
Base.inv(op::LinearizationOperator{T}) where {T} =
    DelinearizationOperator{T}(dest(op), src(op))


"The inverse of a LinearizationOperator."
struct DelinearizationOperator{T} <: DictionaryOperator{T}
    src         ::  Dictionary
    dest        ::  Dictionary
end

DelinearizationOperator(dicts::Dictionary...) =
    DelinearizationOperator{operatoreltype(dicts...)}(dicts...)

DelinearizationOperator{T}(dest::Dictionary) where {T} =
    DelinearizationOperator{T}(DiscreteVectorDictionary{T}(length(dest)), dest)

similar_operator(::DelinearizationOperator{T}, src, dest) where {T} =
    DelinearizationOperator{promote_type(T,operatoreltype(src,dest))}(src, dest)

apply!(op::DelinearizationOperator, coef_dest, coef_src) =
    delinearize_coefficients!(coef_dest, coef_src)

isdiag(op::DelinearizationOperator) = true

Base.adjoint(op::DelinearizationOperator{T}) where {T} =
    LinearizationOperator{T}(dest(op), src(op))
Base.inv(op::DelinearizationOperator{T}) where {T} =
    LinearizationOperator{T}(dest(op), src(op))


const AlternatingSignOperator{T} = DiagonalOperator{T,AlternatingSigns{T}}

AlternatingSignOperator(src::Dictionary) = AlternatingSignOperator{operatoreltype(src)}(src)

function AlternatingSignOperator{T}(src::Dictionary) where {T}
    diag = Diagonal(AlternatingSigns{T}(length(src)))
    DiagonalOperator{T}(diag, src, src)
end


const CoefficientScalingOperator{T} = DiagonalOperator{T,ScaledEntry{T}}

CoefficientScalingOperator(src::Dictionary, index::Int, scalar) =
	CoefficientScalingOperator{promote_type(typeof(scalar),operatoreltype(src))}(src, index, scalar)

function CoefficientScalingOperator{T}(src::Dictionary, index::Int, scalar) where {T}
    diag = Diagonal(ScaledEntry{T}(length(src), index, scalar))
	DiagonalOperator{T}(diag, src, src)
end
