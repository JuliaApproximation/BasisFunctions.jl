# special_operators.jl

"""
A CoefficientScalingOperator scales a single coefficient.
"""
struct CoefficientScalingOperator{T} <: DictionaryOperator{T}
    src     ::  Dictionary
    dest    ::  Dictionary
    index   ::  Int
    scalar  ::  T

    function CoefficientScalingOperator{T}(src, dest, index, scalar) where {T}
        @assert length(src) == length(dest)
        new(src, dest, index, scalar)
    end
end

CoefficientScalingOperator(src::Dictionary, index::Int, scalar::Number) =
    CoefficientScalingOperator(src, src, index, scalar)

CoefficientScalingOperator(src::Dictionary, dest::Dictionary, index::Int, scalar::Number) =
    CoefficientScalingOperator{op_eltype(src,dest)}(src, dest, index, scalar)

similar_operator(op::CoefficientScalingOperator, src, dest) =
    CoefficientScalingOperator(src, dest, index(op), scalar(op))

index(op::CoefficientScalingOperator) = op.index

scalar(op::CoefficientScalingOperator) = op.scalar

is_inplace(::CoefficientScalingOperator) = true
is_diagonal(::CoefficientScalingOperator) = true

if VERSION < v"0.7-"
    ctranspose(op::CoefficientScalingOperator) =
        CoefficientScalingOperator(dest(op), src(op), index(op), conj(scalar(op)))
else
    adjoint(op::CoefficientScalingOperator) =
        CoefficientScalingOperator(dest(op), src(op), index(op), conj(scalar(op)))
end

inv(op::CoefficientScalingOperator) =
    CoefficientScalingOperator(dest(op), src(op), index(op), inv(scalar(op)))

function matrix!(op::CoefficientScalingOperator, a)
    a[:] .= 0
    for i in 1:min(size(a,1),size(a,2))
        a[i,i] = 1
    end
    a[op.index,op.index] = op.scalar
    a
end

function apply_inplace!(op::CoefficientScalingOperator, coef_srcdest)
    coef_srcdest[op.index] = coef_srcdest[op.index]*op.scalar
    coef_srcdest
end

function diagonal(op::CoefficientScalingOperator)
    diag = ones(eltype(op), length(src(op)))
    diag[index(op)] = scalar(op)
    diag
end

string(op::CoefficientScalingOperator) = "Scaling of coefficient $(op.index) by $(op.scalar)"

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
        new(src, dest, op)
    end
end

src(op::WrappedOperator) = op.src
dest(op::WrappedOperator) = op.dest

function WrappedOperator(src::Dictionary, dest::Dictionary, op)
    T1 = eltype(op)
    T2 = op_eltype(src,dest)
    @assert T1==T2
    WrappedOperator{typeof(op),T1}(src, dest, op)
end

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
    @assert op_eltype(w_src,w_dest) == eltype(op)
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
if VERSION < v"0.7-"
    ctranspose(op::WrappedOperator) = wrap_operator(dest(op), src(op), ctranspose(superoperator(op)))
else
    adjoint(op::WrappedOperator) = wrap_operator(dest(op), src(op), adjoint(superoperator(op)))
end

simplify(op::WrappedOperator) = superoperator(op)


"""
An IndexRestrictionOperator selects a subset of coefficients based on their indices.
"""
struct IndexRestrictionOperator{I,T} <: DictionaryOperator{T}
    src         ::  Dictionary
    dest        ::  Dictionary
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

IndexRestrictionOperator(src::Dictionary, dest::Dictionary, subindices) =
    IndexRestrictionOperator{typeof(subindices),op_eltype(src, dest)}(src, dest, subindices)

subindices(op::IndexRestrictionOperator) = op.subindices

similar_operator(op::IndexRestrictionOperator, src, dest) =
    IndexRestrictionOperator(src, dest, subindices(op))

unsafe_wrap_operator(src, dest, op::IndexRestrictionOperator) = similar_operator(op, src, dest)

is_diagonal(::IndexRestrictionOperator) = true

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
struct IndexExtensionOperator{I,T} <: DictionaryOperator{T}
    src         ::  Dictionary
    dest        ::  Dictionary
    subindices  ::  I

    function IndexExtensionOperator{I,T}(src, dest, subindices) where {I,T}
        @assert length(src) == length(subindices)
        @assert length(dest) >= length(src)
        new(src, dest, subindices)
    end
end

IndexExtensionOperator(src::Dictionary, dest::Dictionary, subindices) =
    IndexExtensionOperator{typeof(subindices),op_eltype(src, dest)}(src, dest, subindices)

subindices(op::IndexExtensionOperator) = op.subindices

similar_operator(op::IndexExtensionOperator, src, dest) =
    IndexExtensionOperator(src, dest, subindices(op))

unsafe_wrap_operator(src, dest, op::IndexExtensionOperator) = similar_operator(op, src, dest)

is_diagonal(::IndexExtensionOperator) = true

function apply!(op::IndexExtensionOperator, coef_dest, coef_src)
    fill!(coef_dest, zero(eltype(op)))
    for (i,j) in enumerate(subindices(op))
        coef_dest[j] = coef_src[i]
    end
    coef_dest
end
if VERSION < v"0.7-"
    ctranspose(op::IndexRestrictionOperator) =
        IndexExtensionOperator(dest(op), src(op), subindices(op))
    ctranspose(op::IndexExtensionOperator) =
        IndexRestrictionOperator(dest(op), src(op), subindices(op))
else
    adjoint(op::IndexRestrictionOperator) =
        IndexExtensionOperator(dest(op), src(op), subindices(op))
    adjoint(op::IndexExtensionOperator) =
        IndexRestrictionOperator(dest(op), src(op), subindices(op))
end

string(op::IndexExtensionOperator) = "Zero padding, original elements in "*string(op.subindices)


"""
A MultiplicationOperator is defined by a (matrix-like) object that multiplies
coefficients. The multiplication is in-place if type parameter INPLACE is true,
otherwise it is not in-place.

An alias MatrixOperator is provided, for which type parameter ARRAY equals
Array{T,2}. In this case, multiplication is done using A_mul_B!.
"""
struct MultiplicationOperator{ARRAY,INPLACE,T} <: DictionaryOperator{T}
    src     ::  Dictionary
    dest    ::  Dictionary
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

MultiplicationOperator(src::Dictionary, dest::Dictionary, object; inplace = false) =
    MultiplicationOperator{typeof(object),inplace,op_eltype(src, dest)}(src, dest, object)

MultiplicationOperator(matrix::AbstractMatrix{T}) where {T <: Number} =
    MultiplicationOperator(DiscreteVectorDictionary{T}(size(matrix, 2)), DiscreteVectorDictionary{T}(size(matrix, 1)), matrix)

# Provide aliases for when the object is an actual matrix.
MatrixOperator(matrix::Matrix) = MultiplicationOperator(matrix)

function MatrixOperator(src::Dictionary, dest::Dictionary, matrix::Matrix)
    @assert size(matrix, 1) == length(dest)
    @assert size(matrix, 2) == length(src)
    MultiplicationOperator(src, dest, matrix)
end

similar_operator(op::MultiplicationOperator, src, dest) =
    MultiplicationOperator(src, dest, object(op))

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
    mul!(coef_dest, object(op), coef_src)

# Be forgiving for matrices: if the coefficients are multi-dimensional, reshape to a linear array first.
apply!(op::MultiplicationOperator{Array{ELT1,2},false,ELT2}, coef_dest::AbstractArray{T,N1}, coef_src::AbstractArray{T,N2}) where {ELT1,ELT2,T,N1,N2} =
    apply!(op, reshape(coef_dest, length(coef_dest)), reshape(coef_src, length(coef_src)))


matrix(op::MatrixOperator) = op.object

matrix!(op::MatrixOperator, a::Array) = (a[:] = op.object)

if VERSION < v"0.7-"
    ctranspose(op::MultiplicationOperator) = ctranspose_multiplication(op, object(op))
    # This can be overriden for types of objects that do not support ctranspose
    ctranspose_multiplication(op::MultiplicationOperator, object) =
        MultiplicationOperator(dest(op), src(op), ctranspose(object))
else
    adjoint(op::MultiplicationOperator) = ctranspose_multiplication(op, object(op))
    # This can be overriden for types of objects that do not support ctranspose
    ctranspose_multiplication(op::MultiplicationOperator, object) =
        MultiplicationOperator(dest(op), src(op), adjoint(object))
end

inv(op::MultiplicationOperator) = inv_multiplication(op, object(op))

# This can be overriden for types of objects that do not support inv
inv_multiplication(op::MultiplicationOperator, object) = MultiplicationOperator(dest(op), src(op), inv(object))

if (VERSION < v"0.7-")
    inv_multiplication(op::MultiplicationOperator{Array{ELT,2},false,ELT}, matrix) where {ELT} =
        SolverOperator(dest(op), src(op), qrfact(matrix, Val{true}) )
else
    inv_multiplication(op::MultiplicationOperator{Array{ELT,2},false,ELT}, matrix) where {ELT} =
        SolverOperator(dest(op), src(op), ar(matrix, Val(true)) )
end



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
should implement the \\ operator.
Examples include a QR or SVD factorization, or a dense matrix.
"""
struct SolverOperator{Q,T} <: DictionaryOperator{T}
    src     ::  Dictionary
    dest    ::  Dictionary
    solver  ::  Q
end

SolverOperator(src::Dictionary, dest::Dictionary, solver) =
    SolverOperator{typeof(solver),op_eltype(src, dest)}(src, dest, solver)

similar_operator(op::SolverOperator, src, dest) =
    SolverOperator(src, dest, op.solver)


# TODO: does this allocate memory? Are there (operator-specific) ways to avoid that?
function apply!(op::SolverOperator, coef_dest, coef_src)
    coef_dest[:] = op.solver \ coef_src
    coef_dest
end

if VERSION < v"0.7-"
    ctranspose(op::SolverOperator) = warn("not implemented")
else
    adjoint(op::SolverOperator) = warn("not implemented")
end


"""
A FunctionOperator applies a given function to the set of coefficients and
returns the result.
"""
struct FunctionOperator{F,T} <: DictionaryOperator{T}
    src     ::  Dictionary
    dest    ::  Dictionary
    fun     ::  F
end

FunctionOperator(src::Dictionary, dest::Dictionary, fun) =
    FunctionOperator{typeof(fun),op_eltype(src, dest)}(src, dest, fun)

similar_operator(op::FunctionOperator, src, dest) =
    FunctionOperator(src, dest, op.fun)

# Warning: this very likely allocates memory
apply!(op::FunctionOperator, coef_dest, coef_src) = apply_fun!(op, op.fun, coef_dest, coef_src)

function apply_fun!(op::FunctionOperator, fun, coef_dest, coef_src)
    coef_dest[:] = fun(coef_src)
end

if VERSION < v"0.7-"
    ctranspose(op::FunctionOperator) = ctranspose_function(op, op.fun)
    # This can be overriden for types of functions that do not support ctranspose
    ctranspose_function(op::FunctionOperator, fun) =
        FunctionOperator(dest(op), src(op), ctranspose(fun))
else
    adjoint(op::FunctionOperator) = adjoint_function(op, op.fun)
    # This can be overriden for types of functions that do not support ctranspose
    adjoint_function(op::FunctionOperator, fun) =
        FunctionOperator(dest(op), src(op), adjoint(fun))
end

inv(op::FunctionOperator) = inv_function(op, op.fun)

# This can be overriden for types of functions that do not support inv
inv_function(op::FunctionOperator, fun) = FunctionOperator(dest(op), src(op), inv(fun))

string(op::FunctionOperator) = "Function "*string(op.fun)


## sruct ChebyTDiffOperator{T} <: DictionaryOperator{T}
##     src :: Dictionary
##     dest :: Dictionary
##     end


## ChebyTDiffOperator(src::Dictionary, dest::Dictionary = src) =
##     ChebyTDiffOperator(op_eltype(src,dest), src, dest)

## function ChebyTDiffOperator(::Type{T}, src::Dictionary, dest::Dictionary) where {T}
##     ChebyTDiffOperator{T}(src,dest)
## end

## similar_operator(op::ChebyTDiffOperator, ::Type{S}, src, dest) where {S} = ChebyTDiffOperator(S,src,dest)


# An operator to flip the signs of the coefficients at uneven positions. Used in Chebyshev normalization.
struct UnevenSignFlipOperator{T} <: DictionaryOperator{T}
    src     ::  Dictionary
    dest    ::  Dictionary
end

UnevenSignFlipOperator(src::Dictionary, dest::Dictionary = src) =
    UnevenSignFlipOperator{op_eltype(src,dest)}(src, dest)

similar_operator(op::UnevenSignFlipOperator, src, dest) =
    UnevenSignFlipOperator(src, dest)

is_inplace(::UnevenSignFlipOperator) = true
is_diagonal(::UnevenSignFlipOperator) = true

if VERSION < v"0.7-"
    ctranspose(op::UnevenSignFlipOperator) = op
else
    adjoint(op::UnevenSignFlipOperator) = op
end

inv(op::UnevenSignFlipOperator) = op

function apply_inplace!(op::UnevenSignFlipOperator, coef_srcdest)
    flip = false
    for i in eachindex(coef_srcdest)
        if flip
            coef_srcdest[i] = -coef_srcdest[i]
        end
        flip = !flip
    end
    coef_srcdest
end

diagonal(op::UnevenSignFlipOperator{T}) where {T} = T[-(-1)^i for i in 1:length(src(op))]



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
        @assert coeftype(src(op1)) == coeftype(src(op2))
        @assert coeftype(dest(op1)) == coeftype(dest(op2))

        new(op1, op2, val1, val2, scratch)
    end
end

function OperatorSum(op1::DictionaryOperator, op2::DictionaryOperator, val1::Number, val2::Number)
    T = promote_type(eltype(op1),eltype(op2))
    scratch = zeros(dest(op1))
    OperatorSum{typeof(op1),typeof(op2),T,typeof(scratch)}(op1, op2, T(val1), T(val2), scratch)
end

src(op::OperatorSum) = src(op.op1)

dest(op::OperatorSum) = dest(op.op1)

if VERSION < v"0.7-"
    ctranspose(op::OperatorSum) = OperatorSum(ctranspose(op.op1), ctranspose(op.op2), conj(op.val1), conj(op.val2))
else
    adjoint(op::OperatorSum) = OperatorSum(adjoint(op.op1), adjoint(op.op2), conj(op.val1), conj(op.val2))
end

is_composite(op::OperatorSum) = true
is_diagonal(op::OperatorSum) = is_diagonal(op.op1) && is_diagonal(op.op2)


apply_inplace!(op::OperatorSum, dest, src, coef_srcdest) =
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

apply!(op::OperatorSum, dest, src, coef_dest, coef_src) = apply_sum!(op, op.op1, op.op2, coef_dest, coef_src)

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

function LinearizationOperator(src::Dictionary, dest = DiscreteVectorDictionary{coeftype(src)}(length(src)))
    T = op_eltype(src,dest)
    LinearizationOperator{T}(src, dest)
end

similar_operator(::LinearizationOperator, src, dest) =
    LinearizationOperator(src, dest)

apply!(op::LinearizationOperator, coef_dest, coef_src) =
    linearize_coefficients!(coef_dest, coef_src)

is_diagonal(op::LinearizationOperator) = true


"The inverse of a LinearizationOperator."
struct DelinearizationOperator{T} <: DictionaryOperator{T}
    src         ::  Dictionary
    dest        ::  Dictionary
end

DelinearizationOperator(dest::Dictionary) =
    DelinearizationOperator(DiscreteVectorDictionary{coeftype(dest)}(length(dest)), dest)

function DelinearizationOperator(src::Dictionary, dest::Dictionary)
    T = op_eltype(src,dest)
    DelinearizationOperator{T}(src, dest)
end

similar_operator(::DelinearizationOperator, src, dest) =
    DelinearizationOperator(src, dest)

apply!(op::DelinearizationOperator, coef_dest, coef_src) =
    delinearize_coefficients!(coef_dest, coef_src)

is_diagonal(op::DelinearizationOperator) = true

function SparseOperator(op::DictionaryOperator; options...)
    A = sparse_matrix(op; options...)
    MultiplicationOperator(src(op), dest(op), A, inplace=false)
end
