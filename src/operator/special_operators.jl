# special_operators.jl


"The identity operator"
immutable IdentityOperator{ELT} <: AbstractOperator{ELT}
    src     ::  FunctionSet
    dest    ::  FunctionSet
end

IdentityOperator(src, dest = src) = IdentityOperator{op_eltype(src,dest)}(src,dest)

promote_eltype{ELT,S}(op::IdentityOperator{ELT}, ::Type{S}) =
    IdentityOperator{S}(op.src, op.dest)

is_inplace(::IdentityOperator) = true
is_diagonal(::IdentityOperator) = true

inv(op::IdentityOperator) = IdentityOperator(dest(op), src(op))

ctranspose(op::IdentityOperator) = IdentityOperator(dest(op), src(op))

function matrix!(op::IdentityOperator, a)
    a[:] = 0
    for i in 1:min(size(a,1),size(a,2))
        a[i,i] = 1
    end
    a
end

apply_inplace!(op::IdentityOperator, coef_srcdest) = coef_srcdest

# The identity operator is, well, the identity
(*)(op2::IdentityOperator, op1::IdentityOperator) = IdentityOperator(src(op1), dest(op2))
(*)(op2::AbstractOperator, op1::IdentityOperator) = op2
(*)(op2::IdentityOperator, op1::AbstractOperator) = op1


"The zero operator maps everything to zero."
immutable ZeroOperator{ELT} <: AbstractOperator{ELT}
    src     ::  FunctionSet
    dest    ::  FunctionSet
end

ZeroOperator(src, dest = src) = ZeroOperator{op_eltype(src,dest)}(src,dest)

promote_eltype{ELT,S}(op::ZeroOperator{ELT}, ::Type{S}) =
    ZeroOperator{S}(op.src, op.dest)

is_inplace(::ZeroOperator) = true
is_diagonal(::ZeroOperator) = true

ctranspose(op::ZeroOperator) = ZeroOperator(dest(op), src(op))

matrix!(op::ZeroOperator, a) = (a[:] = 0; a)

apply_inplace!(op::ZeroOperator, coef_srcdest) = coef_srcdest[:] = 0

# The zero operator annihilates all other operators
(*)(op2::ZeroOperator, op1::ZeroOperator) = ZeroOperator(src(op1), dest(op2))
(*)(op2::AbstractOperator, op1::ZeroOperator) = ZeroOperator(src(op1), dest(op2))
(*)(op2::ZeroOperator, op1::AbstractOperator) = ZeroOperator(src(op1), dest(op2))



"""
A ScalingOperator is the identity operator up to a scaling.
"""
immutable ScalingOperator{ELT} <: AbstractOperator{ELT}
    src     ::  FunctionSet
    dest    ::  FunctionSet
    scalar  ::  ELT
end

function ScalingOperator(src::FunctionSet, dest::FunctionSet, scalar::Number)
    ELT = promote_type(op_eltype(src, dest), typeof(scalar))
    ScalingOperator{ELT}(src, dest, ELT(scalar))
end

ScalingOperator(src::FunctionSet, scalar::Number) = ScalingOperator(src, src, scalar)

promote_eltype{ELT,S}(op::ScalingOperator{ELT}, ::Type{S}) =
    ScalingOperator{S}(op.src, op.dest, S(op.scalar))

is_inplace(::ScalingOperator) = true
is_diagonal(::ScalingOperator) = true

scalar(op::ScalingOperator) = op.scalar

ctranspose(op::ScalingOperator) = ScalingOperator(dest(op), src(op), conj(scalar(op)))

inv(op::ScalingOperator) = ScalingOperator(dest(op), src(op), 1/scalar(op))

function apply_inplace!(op::ScalingOperator, coef_srcdest)
    for i in eachindex(coef_srcdest)
        coef_srcdest[i] *= op.scalar
    end
    coef_srcdest
end

# Extra definition for out-of-place version to avoid making an intermediate copy
function apply!(op::ScalingOperator, coef_dest, coef_src)
    for i in eachindex(coef_src)
        coef_dest[i] = op.scalar * coef_src[i]
    end
    coef_dest
end

convert{ELT}(::Type{ScalingOperator{ELT}}, op::IdentityOperator{ELT}) =
    ScalingOperator(src(op), dest(op), one(ELT))

promote_rule{ELT}(::Type{ScalingOperator{ELT}}, ::Type{IdentityOperator{ELT}}) = ScalingOperator{ELT}

for op in (:+, :*, :-, :/)
    @eval $op(op1::ScalingOperator, op2::ScalingOperator) =
        (@assert size(op1) == size(op2); ScalingOperator(src(op1), dest(op1), $op(scalar(op1),scalar(op2))))
end

(*)(a::Number, op::IdentityOperator) = ScalingOperator(src(op), dest(op), a)
(*)(op::IdentityOperator, a::Number) = a*op



"A diagonal operator."
immutable DiagonalOperator{ELT} <: AbstractOperator{ELT}
    src         ::  FunctionSet
    dest        ::  FunctionSet
    diagonal    ::  Vector{ELT}
end

DiagonalOperator{T <: Real}(diagonal::AbstractVector{T}) = DiagonalOperator(Rn{T}(length(diagonal)), diagonal)
DiagonalOperator{T <: Complex}(diagonal::AbstractVector{T}) = DiagonalOperator(Cn{T}(length(diagonal)), diagonal)

DiagonalOperator{ELT}(src::FunctionSet, diagonal::AbstractVector{ELT}) = DiagonalOperator{ELT}(src, src, diagonal)

promote_eltype{ELT,S}(op::DiagonalOperator{ELT}, ::Type{S}) =
    DiagonalOperator{S}(op.src, op.dest, convert(Array{S,1}, op.diagonal))

diagonal(op::DiagonalOperator) = copy(op.diagonal)

is_inplace(::DiagonalOperator) = true
is_diagonal(::DiagonalOperator) = true

inv(op::DiagonalOperator) = DiagonalOperator(dest(op), src(op), op.diagonal.^(-1))

ctranspose(op::DiagonalOperator) = DiagonalOperator(dest(op), src(op), conj(diagonal(op)))

function matrix!(op::DiagonalOperator, a)
    a[:] = 0
    for i in 1:min(size(a,1),size(a,2))
        a[i,i] = op.diagonal[i]
    end
    a
end

function apply_inplace!(op::DiagonalOperator, coef_srcdest)
    for i in 1:length(coef_srcdest)
        coef_srcdest[i] *= op.diagonal[i]
    end
    coef_srcdest
end

# Extra definition for out-of-place version to avoid making an intermediate copy
function apply!(op::DiagonalOperator, coef_dest, coef_src)
    for i in 1:length(coef_dest)
        coef_dest[i] = op.diagonal[i] * coef_src[i]
    end
    coef_dest
end

matrix(op::DiagonalOperator) = diagm(diagonal(op))

promote_rule{ELT}(::Type{DiagonalOperator{ELT}}, ::Type{IdentityOperator{ELT}}) = DiagonalOperator{ELT}
promote_rule{ELT}(::Type{DiagonalOperator{ELT}}, ::Type{ScalingOperator{ELT}}) = DiagonalOperator{ELT}

convert{ELT}(::Type{DiagonalOperator{ELT}}, op::ScalingOperator{ELT}) =
    DiagonalOperator(src(op), dest(op), op.scalar*ones(ELT,length(src(op))))

convert{ELT}(::Type{DiagonalOperator{ELT}}, op::IdentityOperator{ELT}) =
    DiagonalOperator(src(op), dest(op), ones(ELT,length(src(op))))

(*)(op1::DiagonalOperator, op2::DiagonalOperator) = DiagonalOperator(src(op1), dest(op1), diagonal(op1) .* diagonal(op2))
(*)(op1::ScalingOperator, op2::DiagonalOperator) = DiagonalOperator(src(op1), dest(op1), scalar(op1) * diagonal(op2))
(*)(op2::DiagonalOperator, op1::ScalingOperator) = op1 * op2

function diagonal(op::AbstractOperator)
    if ~is_diagonal(op)
        diag(matrix(op))
    else
        diagonal =ones(eltype(op),size(src(op)))
        apply!(op,diagonal)
        diagonal = reshape(diagonal,length(src(op)))
    end
end

function inv_diagonal(op::AbstractOperator)
    @assert is_diagonal(op)
    d = diagonal(op)
    # Avoid getting Inf values, we prefer a pseudo-inverse in this case
    d[find(d.==0)] = Inf
    DiagonalOperator(dest(op), src(op), d.^(-1))
end


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


"""
A MatrixOperator is defined by a full matrix, or more generally by something
that can multiply coefficients.
"""
immutable MatrixOperator{ARRAY,ELT} <: AbstractOperator{ELT}
    src     ::  FunctionSet
    dest    ::  FunctionSet
    matrix  ::  ARRAY

    function MatrixOperator(src, dest, matrix)
        @assert size(matrix,1) == length(dest)
        @assert size(matrix,2) == length(src)

        new(src, dest, matrix)
    end
end


function MatrixOperator(src::FunctionSet, dest::FunctionSet, matrix)
    ELT = promote_type(eltype(matrix), op_eltype(src,dest))
    MatrixOperator{typeof(matrix),ELT}(src, dest, matrix)
end

MatrixOperator{T <: Number}(matrix::AbstractMatrix{T}) =
    MatrixOperator(Rn{T}(size(matrix,2)), Rn{T}(size(matrix,1)), matrix)

MatrixOperator{T <: Number}(matrix::AbstractMatrix{Complex{T}}) =
    MatrixOperator(Cn{T}(size(matrix,2)), Cn{T}(size(matrix,1)), matrix)

ctranspose(op::MatrixOperator) = MatrixOperator(dest(op), src(op), ctranspose(matrix(op)))

# General definition
function apply!(op::MatrixOperator, coef_dest, coef_src)
    coef_dest[:] = op.matrix * coef_src
    coef_dest
end

# Definition in terms of A_mul_B
apply!{T}(op::MatrixOperator, coef_dest::AbstractArray{T,1}, coef_src::AbstractArray{T,1}) =
    A_mul_B!(coef_dest, op.matrix, coef_src)

# # Be forgiving: whenever one of the coefficients is multi-dimensional, reshape to a linear array first.
apply!{T,N1,N2}(op::MatrixOperator, coef_dest::AbstractArray{T,N1}, coef_src::AbstractArray{T,N2}) =
    apply!(op, reshape(coef_dest, length(coef_dest)), reshape(coef_src, length(coef_src)))


matrix(op::MatrixOperator) = op.matrix

matrix!(op::MatrixOperator, a::Array) = (a[:] = op.matrix)


# A SolverOperator wraps around a solver that is used when the SolverOperator is applied. The solver
# should implement the \ operator.
# Examples include a QR or SVD factorization, or a dense matrix.
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

inv(op::MatrixOperator) = SolverOperator(dest(op), src(op), qr(matrix(op)))



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
        coef_srcdest[i] *= op.scale(ELT(natural_index(op.src,i)))^op.order
    end
    coef_srcdest
end

function apply_inplace!{TS1,TS2}(op::IdxnScalingOperator, dest::TensorProductSet{Tuple{TS1,TS2}}, src, coef_srcdest)
    ELT = eltype(op)
    for i in eachindex(coef_srcdest)
        indices = ind2sub(size(dest),i)
        coef_srcdest[i]*=op.scale(ELT(natural_index(TS1,indices[1])),ELT(natural_index(TS2,indices[2])))^op.order
    end
    coef_srcdest
end
inv(op::IdxnScalingOperator) = IdxnScalingOperator(op.src, order=op.order*-1, scale=op.scale)





"A linear combination of operators: val1 * op1 + val2 * op2."
immutable OperatorSum{OP1 <: AbstractOperator,OP2 <: AbstractOperator,ELT,N} <: AbstractOperator{ELT}
    op1         ::  OP1
    op2         ::  OP2
    val1        ::  ELT
    val2        ::  ELT
    scratch     ::  Array{ELT,N}

    function OperatorSum(op1, op2, val1, val2)
        # We don't enforce that source and destination of op1 and op2 are the same, but at least
        # their sizes must match.
        @assert size(src(op1)) == size(src(op2))
        @assert size(dest(op1)) == size(dest(op2))

        new(op1, op2, val1, val2, zeros(ELT,size(dest(op1))))
    end
end

function OperatorSum(op1::AbstractOperator, op2::AbstractOperator, val1::Number, val2::Number)
#    @assert eltype(op1) == eltype(op2)
    ELT = promote_type(eltype(op1), eltype(op2), typeof(val1), typeof(val2))
    OperatorSum{typeof(op1),typeof(op2),ELT,index_dim(dest(op1))}(op1, op2, convert(ELT, val1), convert(ELT, val2))
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
