# special_operators.jl


"The identity operator"
immutable IdentityOperator{SRC,DEST} <: AbstractOperator{SRC,DEST}
    src     ::  SRC
    dest    ::  DEST
end

IdentityOperator(src::FunctionSet) = IdentityOperator(src, src)

is_inplace{OP <: IdentityOperator}(::Type{OP}) = True

is_diagonal{OP <: IdentityOperator}(::Type{OP}) = True

inv(op::IdentityOperator) = IdentityOperator(dest(op), src(op))

ctranspose(op::IdentityOperator) = inv(op)

apply!(op::IdentityOperator, dest, src, coef_srcdest) = nothing

# The identity operator is, well, the identity
(*)(op2::IdentityOperator, op1::IdentityOperator) = IdentityOperator(src(op1), dest(op2))
(*)(op2::AbstractOperator, op1::IdentityOperator) = op2
(*)(op2::IdentityOperator, op1::AbstractOperator) = op1


"""
A ScalingOperator is the identity operator up to a scaling.
"""
immutable ScalingOperator{ELT,SRC} <: AbstractOperator{SRC,SRC}
    src     ::  SRC
    scalar  ::  ELT
end

function ScalingOperator{S <: Number,SRC}(src::SRC, scalar::S)
        # Make sure the scalar type is promotable to the SRC type
        @assert promote_type(eltype(SRC),S) == eltype(SRC)
        ELT = S
    ScalingOperator{ELT,SRC}(src, scalar)
end

dest(op::ScalingOperator) = src(op)

eltype{ELT,SRC}(::Type{ScalingOperator{ELT,SRC}}) = ELT

is_inplace{OP <: ScalingOperator}(::Type{OP}) = True

is_diagonal{OP <: ScalingOperator}(::Type{OP}) = True

scalar(op::ScalingOperator) = op.scalar

ctranspose(op::ScalingOperator) = op

ctranspose{T <: Real}(op::ScalingOperator{Complex{T}}) = ScalingOperator(src(op), conj(scalar(op)))

inv(op::ScalingOperator) = ScalingOperator(src(op), 1/scalar(op))

convert{ELT,SRC}(::Type{ScalingOperator{ELT,SRC}}, op::IdentityOperator{SRC}) = ScalingOperator(src(op), ELT(1))

promote_rule{ELT,SRC}(::Type{ScalingOperator{ELT,SRC}}, ::Type{IdentityOperator{SRC}}) = ScalingOperator{ELT,SRC}

for op in (:+, :*, :-, :/)
    @eval $op(op1::ScalingOperator, op2::ScalingOperator) =
        (@assert size(op1) == size(op2); ScalingOperator(src(op1), $op(scalar(op1),scalar(op2))))
end

(*)(a::Number, op::IdentityOperator) = ScalingOperator(src(op), a)
(*)(op::IdentityOperator, a::Number) = ScalingOperator(src(op), a)
# Universal scaling of operators
(*)(a::Number, op::AbstractOperator) = ScalingOperator(src(op), a)*op
(*)(op::AbstractOperator, a::Number) = op*ScalingOperator(src(op), a)


(+){SRC}(op2::IdentityOperator{SRC}, op1::IdentityOperator{SRC}) =
    convert(ScalingOperator{eltype(SRC),SRC}, op2) + convert(ScalingOperator{eltype(SRC),SRC}, op1)



function apply!(op::ScalingOperator, dest, src, coef_srcdest)
    for i in eachindex(coef_srcdest)
        coef_srcdest[i] *= op.scalar
    end
end

# Extra definition for out-of-place version to avoid making an intermediate copy
function apply!(op::ScalingOperator, dest, src, coef_dest, coef_src)
    for i in eachindex(coef_src)
        coef_dest[i] = op.scalar * coef_src[i]
    end
end



"A diagonal operator."
immutable DiagonalOperator{ELT,SRC,DEST} <: AbstractOperator{SRC,DEST}
    src     ::  SRC
    dest    ::  DEST
    diagonal::  Array{ELT,1}
end

diagonal(op::DiagonalOperator) = op.diagonal

DiagonalOperator{ELT,SRC,DEST,N}(src::SRC,dest::DEST,diagonal :: Array{ELT,N}) = DiagonalOperator(src,dest,diagonal[:])
is_inplace{OP <: DiagonalOperator}(::Type{OP}) = True

is_diagonal{OP <: DiagonalOperator}(::Type{OP}) = True

function inv(op::AbstractOperator, is_diagonal::True)
    # check for zero elements.
    if length(find(diagonal(op).==0))==0
        DiagonalOperator(dest(op), src(op), diagonal(op).^(-1))
    else
        d = diagonal(op)
        d[find(d.==0)] = Inf
        DiagonalOperator(dest(op), src(op), d.^(-1))
    end
end
# Any is_diagonal operator can be inverted into a diagonal operator


ctranspose(op::DiagonalOperator) = DiagonalOperator(dest(op), src(op), diagonal(op))

ctranspose{T <: Number}(op::DiagonalOperator{Complex{T}}) = DiagonalOperator(dest(op), src(op), conj(diagonal(op)))

function apply!(op::DiagonalOperator, dest, src, coef_srcdest)
    for i in eachindex(coef_srcdest)
        coef_srcdest[i] *= op.diagonal[i]
    end
end

# Extra definition for out-of-place version to avoid making an intermediate copy
function apply!(op::ScalingOperator, dest, src, coef_dest, coef_src)
    for i in eachindex(coef_src)
        coef_dest[i] = op.scalar * coef_src[i]
    end
end
# Calculus with DiagonalOperators is calculus on the diagonals
(+){ELT,SRC,DEST}(op1::DiagonalOperator{ELT,SRC,DEST}, op2::DiagonalOperator{ELT,SRC,DEST}) = DiagonalOperator(SRC,DEST,diagonal(op1)+diagonal(op2))
(-){ELT,SRC,DEST}(op1::DiagonalOperator{ELT,SRC,DEST}, op2::DiagonalOperator{ELT,SRC,DEST}) = DiagonalOperator(SRC,DEST,diagonal(op1)-diagonal(op2))
(*)(a::Number, op::DiagonalOperator) = DiagonalOperator(src(op),dest(op),a*diagonal(op))
(*)(op::AbstractOperator, a::Number) = DiagonalOperator(src(op),dest(op),a*diagonal(op))



"""
A CoefficientScalingOperator scales a single coefficient.
"""
immutable CoefficientScalingOperator{ELT,SRC} <: AbstractOperator{SRC,SRC}
    src     ::  SRC
    index   ::  Int
    scalar  ::  ELT
end

function CoefficientScalingOperator{S <: Number, SRC}(src::SRC, index::Int, scalar::S)
        # Make sure the scalar type is promotable to the SRC type
        @assert promote_type(eltype(SRC),S) == eltype(SRC)
        ELT = S
        CoefficientScalingOperator{ELT,SRC}(src, index, scalar)
end

eltype{ELT,SRC}(::Type{CoefficientScalingOperator{ELT,SRC}}) = ELT

dest(op::CoefficientScalingOperator) = src(op)

index(op::CoefficientScalingOperator) = op.index

scalar(op::CoefficientScalingOperator) = op.scalar

is_inplace{OP <: CoefficientScalingOperator}(::Type{OP}) = True

is_diagonal{OP <: CoefficientScalingOperator}(::Type{OP}) = True

ctranspose(op::CoefficientScalingOperator) = op

ctranspose{T <: Number}(op::CoefficientScalingOperator{Complex{T}}) =
    CoefficientScalingOperator(src(op), index(op), conj(scalar(op)))

inv(op::CoefficientScalingOperator) = CoefficientScalingOperator(src(op), index(op), 1/scalar(op))

apply!(op::CoefficientScalingOperator, dest, src, coef_srcdest) = coef_srcdest[op.index] *= op.scalar



"""
A WrappedOperator has a source and destination, as well as an embedded operator with its own
source and destination. The coefficients of the source of the WrappedOperator are passed on
unaltered as coefficients of the source of the embedded operator. The resulting coefficients
of the embedded destination are returned as coefficients of the wrapping destination.
"""
immutable WrappedOperator{OP,SRC,DEST} <: AbstractOperator{SRC,DEST}
    src     ::  SRC
    dest    ::  DEST

    op      ::  OP
end

operator(op::WrappedOperator) = op.op

is_inplace{OP,SRC,DEST}(::Type{WrappedOperator{OP,SRC,DEST}}) = is_inplace(OP)

is_diagonal{OP,SRC,DEST}(::Type{WrappedOperator{OP,SRC,DEST}}) = is_diagonal(OP)

apply!(op::WrappedOperator, dest, src, coef_srcdest) = apply!(op.op, coef_srcdest)

apply!(op::WrappedOperator, dest, src, coef_dest, coef_src) = apply!(op.op, coef_dest, coef_src)

inv(op::WrappedOperator) = WrappedOperator(dest(op), src(op), inv(op.op))

ctranspose(op::WrappedOperator) = WrappedOperator(dest(op), src(op), ctranspose(op.op))


"A MatrixOperator is defined by a full matrix."
immutable MatrixOperator{ARRAY,SRC,DEST} <: AbstractOperator{SRC,DEST}
    matrix  ::  ARRAY
    src     ::  SRC
    dest    ::  DEST

    function MatrixOperator(matrix::AbstractMatrix, src, dest)
        @assert size(matrix,1) == length(dest)
        @assert size(matrix,2) == length(src)

        new(matrix, src, dest)
    end
end


MatrixOperator{ARRAY <: AbstractMatrix,SRC,DEST}(matrix::ARRAY, src::SRC, dest::DEST) =
    MatrixOperator{ARRAY, SRC, DEST}(matrix, src, dest)

MatrixOperator{T <: Number}(matrix::AbstractMatrix{T}) = MatrixOperator(matrix, Rn{T}(size(matrix,2)), Rn{T}(size(matrix,1)))

MatrixOperator{T <: Number}(matrix::AbstractMatrix{Complex{T}}) =
    MatrixOperator(matrix, Cn{T}(size(matrix,2)), Cn{T}(size(matrix,1)))

eltype{ARRAY,SRC,DEST}(::Type{MatrixOperator{ARRAY,SRC,DEST}}) = eltype(ARRAY)

ctranspose(op::MatrixOperator) = MatrixOperator(ctranspose(matrix(op)), dest(op), src(op))

inv(op::MatrixOperator) = MatrixOperator(inv(matrix(op)), dest(op), src(op))

# General definition
apply!(op::MatrixOperator, dest, src, coef_dest, coef_src) = (coef_dest[:] = op.matrix * coef_src)

# Definition in terms of A_mul_B
apply!{T}(op::MatrixOperator, dest, src, coef_dest::AbstractArray{T,1}, coef_src::AbstractArray{T,1}) =
    A_mul_B!(coef_dest, op.matrix, coef_src)

# Be forgiving: whenever one of the coefficients is multi-dimensional, reshape to a linear array first.
apply!{T,N1,N2}(op::MatrixOperator, dest, src, coef_dest::AbstractArray{T,N1}, coef_src::AbstractArray{T,N2}) =
    apply!(op, reshape(coef_dest, length(coef_dest)), reshape(coef_src, length(coef_src)))


matrix(op::MatrixOperator) = op.matrix

matrix!(op::MatrixOperator, a::Array) = (a[:] = op.matrix)


# A SolverOperator wraps around a solver that is used when the SolverOperator is applied. The solver
# should implement the \ operator.
# Examples include a QR or SVD factorization, or a dense matrix.
immutable SolverOperator{Q,SRC,DEST} <: AbstractOperator{SRC,DEST}
    src     ::  SRC
    dest    ::  DEST
    solver  ::  Q
end

apply!(op::SolverOperator, dest, src, coef_dest, coef_src) = (coef_dest[:] = op.solver \ coef_src)


# An operator to flip the signs of the coefficients at uneven positions. Used in Chebyshev normalization.
immutable UnevenSignFlipOperator{SRC,DEST} <: AbstractOperator{SRC,DEST}
    src :: SRC
    dest :: DEST
end

UnevenSignFlipOperator{SRC}(src::SRC) = UnevenSignFlipOperator{SRC,SRC}(src,src)

is_inplace{OP <: UnevenSignFlipOperator}(::Type{OP}) = True

inv(op::UnevenSignFlipOperator) = op

function apply!(op::UnevenSignFlipOperator, dest, src, coef_srcdest)
    for i in eachindex(coef_srcdest)
        coef_srcdest[i] *= (-1)^(i+1) 
    end
    coef_srcdest
end

# An index scaling operator, used to generate weights for the polynomial scaling algorithm.
immutable IdxnScalingOperator{SRC,DEST} <: AbstractOperator{SRC,DEST}
    src   :: SRC
    dest  :: DEST
    order :: Int
    scale :: Function
end

IdxnScalingOperator{SRC}(src::SRC; scale = default_scaling_function) = IdxnScalingOperator(src,src,1,scale)

default_scaling_function(i) = 10.0^-4+(abs(i))+abs(i)^2+abs(i)^3
default_scaling_function(i,j) = 1+(abs(i)^2+abs(j)^2)
is_inplace{OP <: IdxnScalingOperator}(::Type{OP}) = True
is_diagonal{OP <: IdxnScalingOperator}(::Type{OP}) = True

ctranspose(op::IdxnScalingOperator) = IdxnScalingOperator(dest(op),src(op),op.order,op.scale)
function apply!(op::IdxnScalingOperator, dest, src, coef_srcdest)
    for i in eachindex(coef_srcdest)
        coef_srcdest[i]*=op.scale(convert(eltype(src),natural_index(op.src,i)))^op.order
    end
end

function apply!{TS,SN,LEN}(op::IdxnScalingOperator, dest::TensorProductSet{TS,SN,LEN,2}, src, coef_srcdest)
    for i in eachindex(coef_srcdest)
        indices = ind2sub(size(dest),i)
        coef_srcdest[i]*=op.scale(convert(eltype(src),natural_index(TS[1],indices[1])),convert(eltype(src),natural_index(TS[2],indices[2])))^op.order
    end
end





"A linear combination of operators: val1 * op1 + val2 * op2."
immutable OperatorSum{OP1,OP2,ELT,N,SRC,DEST} <: AbstractOperator{SRC,DEST}
    op1         ::  OP1
    op2         ::  OP2
    val1        ::  ELT
    val2        ::  ELT
    scratch     ::  Array{ELT,N}

    function OperatorSum(op1::AbstractOperator, op2::AbstractOperator, val1::ELT, val2::ELT)
        # We don't enforce that source and destination of op1 and op2 are the same, but at least
        # their sizes must match.
        @assert size(src(op1)) == size(src(op2))
        @assert size(dest(op1)) == size(dest(op2))
        
        new(op1, op2, val1, val2, zeros(ELT,size(dest(op1))))
    end
end

function OperatorSum{SRC,DEST,S1 <: Number, S2 <: Number}(op1::AbstractOperator{SRC,DEST}, op2::AbstractOperator, val1::S1, val2::S2)
#    @assert eltype(op1) == eltype(op2)
    ELT = promote_type(eltype(op1), eltype(op2))
    # make sure that the type of the values matches that of SRC and DEST
    @assert promote_type(S1, S2, eltype(op1), eltype(op2)) == ELT
    OperatorSum{typeof(op1), typeof(op2), ELT, index_dim(dest(op1)), SRC, DEST}(op1, op2, convert(ELT, val1), convert(ELT, val2))
end

src(op::OperatorSum) = src(op.op1)

dest(op::OperatorSum) = dest(op.op1)

eltype{OP1,OP2,ELT,N,SRC,DEST}(::Type{OperatorSum{OP1,OP2,ELT,N,SRC,DEST}}) = ELT

ctranspose(op::OperatorSum) = OperatorSum(ctranspose(op.op1), ctranspose(op.op2), conj(op.val1), conj(op.val2))


is_diagonal{OP1,OP2,ELT,N,SRC,DEST}(::Type{OperatorSum{OP1,OP2,ELT,N,SRC,DEST}}) = is_diagonal(OP1) & is_diagonal(OP2)


apply!(op::OperatorSum, dest, src, coef_srcdest) = apply!(op, op.op1, op.op2, coef_srcdest)

function apply!(op::OperatorSum, op1::AbstractOperator, op2::AbstractOperator, coef_srcdest)
    scratch = op.scratch

    apply!(op1, scratch, coef_srcdest)
    apply!(op2, coef_srcdest)

    for i in eachindex(coef_srcdest)
        coef_srcdest[i] = op.val1 * scratch[i] + op.val2 * coef_srcdest[i]
    end
end

apply!(op::OperatorSum, dest, src, coef_dest, coef_src) = apply!(op, op.op1, op.op2, coef_dest, coef_src)

function apply!(op::OperatorSum, op1::AbstractOperator, op2::AbstractOperator, coef_dest, coef_src)
    scratch = op.scratch

    apply!(op1, scratch, coef_src)
    apply!(op2, coef_dest, coef_src)

    for i in eachindex(coef_dest)
        coef_dest[i] = op.val1 * scratch[i] + op.val2 * coef_dest[i]
    end
end

# Avoid unnecessary work when one of the operators (or both) is a ScalingOperator.
function apply!(op::OperatorSum, op1::ScalingOperator, op2::ScalingOperator, coef_dest, coef_src)
    val = op.val1 * scalar(op1) + op.val2 * scalar(op2)
    for i in eachindex(coef_dest)
        coef_dest[i] = val * coef_src[i]
    end
end

function apply!(op::OperatorSum, op1::ScalingOperator, op2::AbstractOperator, coef_dest, coef_src)
    apply!(op2, coef_dest, coef_src)

    val1 = op.val1 * scalar(op1)
    for i in eachindex(coef_dest)
        coef_dest[i] = val1 * coef_src[i] + op.val2 * coef_dest[i]
    end
end

function apply!(op::OperatorSum, op1::AbstractOperator, op2::ScalingOperator, coef_dest, coef_src)
    apply!(op1, coef_dest, coef_src)

    val2 = op.val2 * scalar(op2)
    for i in eachindex(coef_dest)
        coef_dest[i] = op.val1 * coef_dest[i] + val2 * coef_src[i]
    end
end


(+)(op1::AbstractOperator, op2::AbstractOperator) = OperatorSum(op1, op2, one(eltype(op1)), one(eltype(op2)))
(-)(op1::AbstractOperator, op2::AbstractOperator) = OperatorSum(op1, op2, one(eltype(op1)), -one(eltype(op2)))



