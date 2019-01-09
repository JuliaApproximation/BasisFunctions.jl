

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
struct WrappedOperator{OP,T} <: DerivedOperator{T}
    src     ::  Dictionary
    dest    ::  Dictionary
    op      ::  OP

    function WrappedOperator{OP,T}(src, dest, op) where {OP,T}
        @assert size(op,1) == length(dest)
        @assert size(op,2) == length(src)
        # Make sure that the wrapped operator has a more general element type
        # than the combination of src and dest
        @assert promote_type(T,op_eltype(src,dest)) == T
        new(src, dest, op)
    end
end

src(op::WrappedOperator) = op.src
dest(op::WrappedOperator) = op.dest

WrappedOperator{T}(src::Dictionary, dest::Dictionary, op) where {T} =
    WrappedOperator{typeof(op),T}(src,dest, op)

WrappedOperator(src::Dictionary, dest::Dictionary, op) =
    WrappedOperator{eltype(op)}(src, dest, op)

superoperator(op::WrappedOperator) = op.op

function similar_operator(op::WrappedOperator, op_src, op_dest)
    subop = superoperator(op)
    WrappedOperator(op_src, op_dest, subop)
end


## function stencil(op::WrappedOperator, S)
##     if haskey(S,op)
##         return op
##     end
##     A = Any[]
##     push!(A,"W(")
##     s = stencil(op.op)
##     if isa(s,AbstractOperator)
##         push!(A,s)
##     else
##         for i=1:length(s)
##             push!(A,s[i])
##         end
##         A = recurse_stencil(op.op,A,S)
##     end
##     push!(A,")")
##     A
## end


"""
The function wrap_operator returns an operator with the given source and destination,
and with the action of the given operator. Depending on the operator, the result is
a WrappedOperator, but sometimes that can be avoided.
"""
function wrap_operator(w_src, w_dest, op)
    # We do some consistency checks
    @assert size(w_src) == size(src(op))
    @assert size(w_dest) == size(dest(op))
    @assert promote_type(eltype(op),op_eltype(w_src,w_dest)) == eltype(op)
    unsafe_wrap_operator(w_src, w_dest, op)
end

# No need to wrap a wrapped operator
unsafe_wrap_operator(src, dest, op::WrappedOperator) = wrap_operator(src, dest, superoperator(op))

# default fallback routine
function unsafe_wrap_operator(w_src, w_dest, op::DictionaryOperator)
    if (w_src == src(op)) && (w_dest == dest(op))
        op
    else
        WrappedOperator(w_src, w_dest, op)
    end
end


inv(op::WrappedOperator) = wrap_operator(dest(op), src(op), inv(superoperator(op)))

adjoint(op::WrappedOperator) = wrap_operator(dest(op), src(op), adjoint(superoperator(op)))

simplify(op::WrappedOperator) = superoperator(op)


"""
An IndexRestrictionOperator selects a subset of coefficients based on their indices.
"""
struct IndexRestrictionOperator{T,I} <: DictionaryOperator{T}
    src         ::  Dictionary
    dest        ::  Dictionary
    subindices  ::  I

    function IndexRestrictionOperator{T,I}(src, dest, subindices) where {T,I}
        # Verify the lenght of subindices, but only if its length is defined
        if IteratorSize(subindices) != Base.SizeUnknown()
            @assert length(dest) == length(subindices)
        end
        @assert length(src) >= length(dest)
        new(src, dest, subindices)
    end
end

IndexRestrictionOperator{T}(src::Dictionary, dest::Dictionary, subindices) where {T} =
    IndexRestrictionOperator{T,typeof(subindices)}(src, dest, subindices)

IndexRestrictionOperator(src::Dictionary, dest::Dictionary, subindices) =
    IndexRestrictionOperator{op_eltype(src,dest)}(src, dest, subindices)

subindices(op::IndexRestrictionOperator) = op.subindices

similar_operator(op::IndexRestrictionOperator{T}, src, dest) where {T} =
    IndexRestrictionOperator{promote_type(T,op_eltype(src,dest))}(src, dest, subindices(op))

unsafe_wrap_operator(src, dest, op::IndexRestrictionOperator) = similar_operator(op, src, dest)

isdiagonal(::IndexRestrictionOperator) = true

apply!(op::IndexRestrictionOperator, coef_dest, coef_src) = apply!(op, coef_dest, coef_src, subindices(op))

function apply!(op::IndexRestrictionOperator, coef_dest, coef_src, subindices)
    for (i,j) in enumerate(subindices)
        coef_dest[i] = coef_src[j]
    end
    coef_dest
end

function string(op::IndexRestrictionOperator)
    # If there are many indices being selected, then the printed operator
    # takes up many many lines. The first six indices should be fine for the
    # purposes of getting the jist of the operator
    if length(op.subindices) < 6
        return "Selecting coefficients "*string(op.subindices)
    else
        return "Selecting coefficients "*string(op.subindices[1:6])*"..."
    end
end

"""
An IndexExtensionOperator embeds coefficients in a larger set based on their indices.
"""
struct IndexExtensionOperator{T,I} <: DictionaryOperator{T}
    src         ::  Dictionary
    dest        ::  Dictionary
    subindices  ::  I

    function IndexExtensionOperator{T,I}(src, dest, subindices) where {T,I}
        @assert length(src) == length(subindices)
        @assert length(dest) >= length(src)
        new(src, dest, subindices)
    end
end

IndexExtensionOperator{T}(src::Dictionary, dest::Dictionary, subindices) where {T} =
    IndexExtensionOperator{T,typeof(subindices)}(src, dest, subindices)

IndexExtensionOperator(src::Dictionary, dest::Dictionary, subindices) =
    IndexExtensionOperator{op_eltype(src,dest)}(src, dest, subindices)

subindices(op::IndexExtensionOperator) = op.subindices

similar_operator(op::IndexExtensionOperator{T}, src, dest) where {T} =
    IndexExtensionOperator{promote_type(T,op_eltype(src,dest))}(src, dest, subindices(op))

unsafe_wrap_operator(src, dest, op::IndexExtensionOperator) = similar_operator(op, src, dest)

isdiagonal(::IndexExtensionOperator) = true

function apply!(op::IndexExtensionOperator, coef_dest, coef_src)
    fill!(coef_dest, zero(eltype(op)))
    for (i,j) in enumerate(subindices(op))
        coef_dest[j] = coef_src[i]
    end
    coef_dest
end

adjoint(op::IndexRestrictionOperator{T}) where {T} =
    IndexExtensionOperator{T}(dest(op), src(op), subindices(op))
adjoint(op::IndexExtensionOperator{T}) where {T} =
    IndexRestrictionOperator{T}(dest(op), src(op), subindices(op))

string(op::IndexExtensionOperator) = "Zero padding, original elements in "*string(op.subindices)


"""
A MultiplicationOperator is defined by a (matrix-like) object that multiplies
coefficients. The multiplication is in-place if type parameter INPLACE is true,
otherwise it is not in-place.

An alias MatrixOperator is provided, for which type parameter ARRAY equals
Array{T,2}. In this case, multiplication is done using A_mul_B!.
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

# An MatrixOperator is defined by an actual matrix, i.e. the parameter
# ARRAY is Array{T,2}.
const MatrixOperator{T} = MultiplicationOperator{T,Array{T,2},false}

MultiplicationOperator{T}(src::Dictionary, dest::Dictionary, object; inplace = false) where {T} =
    MultiplicationOperator{T,typeof(object),inplace}(src, dest, object)

MultiplicationOperator(src::Dictionary, dest::Dictionary, object; inplace = false) =
    MultiplicationOperator{promote_type(eltype(object),op_eltype(src,dest))}(src::Dictionary, dest::Dictionary, object; inplace=inplace)

MultiplicationOperator(matrix::AbstractMatrix{T}) where {T <: Number} =
    MultiplicationOperator(DiscreteVectorDictionary{T}(size(matrix, 2)), DiscreteVectorDictionary{T}(size(matrix, 1)), matrix)

# Provide aliases for when the object is an actual matrix.
MatrixOperator(matrix::AbstractMatrix) = MultiplicationOperator(matrix)

function MatrixOperator(src::Dictionary, dest::Dictionary, matrix::AbstractMatrix)
    @assert size(matrix, 1) == length(dest)
    @assert size(matrix, 2) == length(src)
    MultiplicationOperator(src, dest, matrix)
end

similar_operator(op::MultiplicationOperator{T,ARRAY,INPLACE}, src, dest) where {T,ARRAY,INPLACE} =
    MultiplicationOperator{promote_type(T,op_eltype(src,dest))}(src, dest, object(op); inplace=INPLACE)

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
matrix(op::MatrixOperator) = op.object

matrix!(op::MatrixOperator, a::Array) = (a[:] = op.object)


adjoint(op::MultiplicationOperator) = adjoint_multiplication(op, object(op))
# This can be overriden for types of objects that do not support adjoint
adjoint_multiplication(op::MultiplicationOperator, object) =
    MultiplicationOperator(dest(op), src(op), adjoint(object))

adjoint_multiplication(op::MultiplicationOperator, object::Array{T,2}) where {T} =
    # We copy the adjoint in order to avoid storing an Adjoint type
    MultiplicationOperator(dest(op), src(op), copy(adjoint(object)))

inv(op::MultiplicationOperator) = inv_multiplication(op, object(op))

# This can be overriden for types of objects that do not support inv
inv_multiplication(op::MultiplicationOperator, object) = MultiplicationOperator(dest(op), src(op), inv(object))

# Use QR for matrices by default
inv_multiplication(op::MatrixOperator, matrix) = QR_solver(op)



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
A FunctionOperator applies a given function to the set of coefficients and
returns the result.
"""
struct FunctionOperator{F,T} <: DictionaryOperator{T}
    src     ::  Dictionary
    dest    ::  Dictionary
    fun     ::  F
end

FunctionOperator{T}(src, dest, fun) where {T} =
    FunctionOperator{typeof(fun),T}(src, dest, fun)

FunctionOperator(src::Dictionary, dest::Dictionary, fun) =
    FunctionOperator{op_eltype(src,dest)}(src, dest, fun)

similar_operator(op::FunctionOperator{F,T}, src, dest) where {F,T} =
    FunctionOperator{promote_type(T,op_eltype(src,dest))}(src, dest, op.fun)

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

string(op::FunctionOperator) = "Function "*string(op.fun)



"A linear combination of operators: val1 * op1 + val2 * op2."
struct OperatorSum{OP1 <: DictionaryOperator,OP2 <: DictionaryOperator,T,S} <: DictionaryOperator{T}
    op1         ::  OP1
    op2         ::  OP2
    val1        ::  T
    val2        ::  T
    scratch     ::  S

    function OperatorSum{OP1,OP2,T,S}(op1::OP1, op2::OP2, val1::T, val2::T, scratch::S) where {OP1 <: DictionaryOperator,OP2 <: DictionaryOperator,T,S}
        # We don't enforce that source and destination of op1 and op2 are the same, but at least
        # their sizes and coefficient types must match.
        @assert size(src(op1)) == size(src(op2))
        @assert size(dest(op1)) == size(dest(op2))
        @assert coefficienttype(src(op1)) == coefficienttype(src(op2))
        @assert coefficienttype(dest(op1)) == coefficienttype(dest(op2))

        new(op1, op2, val1, val2, scratch)
    end
end

function OperatorSum(op1::DictionaryOperator, op2::DictionaryOperator, val1::Number, val2::Number)
    T = promote_type(eltype(op1), eltype(op2), typeof(val1), typeof(val2))
    scratch = zeros(T, dest(op1))
    OperatorSum{typeof(op1),typeof(op2),T,typeof(scratch)}(op1, op2, T(val1), T(val2), scratch)
end

src(op::OperatorSum) = src(op.op1)

dest(op::OperatorSum) = dest(op.op1)

adjoint(op::OperatorSum) = OperatorSum(adjoint(op.op1), adjoint(op.op2), conj(op.val1), conj(op.val2))

iscomposite(op::OperatorSum) = true
isdiagonal(op::OperatorSum) = isdiagonal(op.op1) && isdiagonal(op.op2)


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

elements(op::OperatorSum) = (op.op1,op.op2)

(+)(op1::DictionaryOperator, op2::DictionaryOperator) = OperatorSum(op1, op2, 1, 1)
(-)(op1::DictionaryOperator, op2::DictionaryOperator) = OperatorSum(op1, op2, 1, -1)

function stencil(op::OperatorSum,S)
    s1=""
    if op.val1==-1
        s1="-"
    end
    s2=" + "
    if op.val2==-1
        s2=" - "
    end
    A = [s1,op.op1,s2,op.op2]
    recurse_stencil(op,A,S)
end

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

LinearizationOperator(src::Dictionary, dest = DiscreteVectorDictionary{coefficienttype(src)}(length(src));
            T = op_eltype(src, dest)) =
    LinearizationOperator{T}(src, dest)

similar_operator(::LinearizationOperator{T}, src, dest) where {T} =
    LinearizationOperator{promote_type(T,op_eltype(src,dest))}(src, dest)

apply!(op::LinearizationOperator, coef_dest, coef_src) =
    linearize_coefficients!(coef_dest, coef_src)

isdiagonal(op::LinearizationOperator) = true


"The inverse of a LinearizationOperator."
struct DelinearizationOperator{T} <: DictionaryOperator{T}
    src         ::  Dictionary
    dest        ::  Dictionary
end

DelinearizationOperator(dest::Dictionary) =
    DelinearizationOperator(DiscreteVectorDictionary{coefficienttype(dest)}(length(dest)), dest)

DelinearizationOperator(src::Dictionary, dest::Dictionary) =
    DelinearizationOperator{op_eltype(src,dest)}(src, dest)

similar_operator(::DelinearizationOperator, src, dest) =
    DelinearizationOperator(src, dest)

apply!(op::DelinearizationOperator, coef_dest, coef_src) =
    delinearize_coefficients!(coef_dest, coef_src)

isdiagonal(op::DelinearizationOperator) = true

function SparseOperator(op::DictionaryOperator; options...)
    A = sparse_matrix(op; options...)
    MultiplicationOperator(src(op), dest(op), A, inplace=false)
end


const AlternatingSignOperator{T} = DiagonalOperator{T,AlternatingSigns{T}}

AlternatingSignOperator(src) = AlternatingSignOperator{coefficienttype(src)}(src)
AlternatingSignOperator{T}(src) where {T} = AlternatingSignOperator{T}(src, src, Diagonal(AlternatingSigns{T}(length(src))))

string(op::AlternatingSignOperator) = "Alternating sign operator of length $(size(op,1))"


const CoefficientScalingOperator{T} = DiagonalOperator{T,ScaledEntry{T}}

CoefficientScalingOperator(src::Dictionary, index::Int, scalar) =
	CoefficientScalingOperator{promote_type(coefficienttype(src),typeof(scalar))}(src, index, scalar)
CoefficientScalingOperator{T}(src::Dictionary, index::Int, scalar) where {T} =
	CoefficientScalingOperator{T}(src, src, Diagonal(ScaledEntry{T}(length(src), index, scalar)))

string(op::CoefficientScalingOperator) = "Scaling of coefficient $(op.A.diag.index) by $(op.A.diag.scalar)"
