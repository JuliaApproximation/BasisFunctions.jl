# basic_operators.jl

"""
The identity operator between two (possibly different) function sets.
"""
immutable IdentityOperator{ELT} <: AbstractOperator{ELT}
    src     ::  FunctionSet
    dest    ::  FunctionSet
end

IdentityOperator(src, dest = src) = IdentityOperator{op_eltype(src,dest)}(src,dest)

promote_eltype{ELT,S}(op::IdentityOperator{ELT}, ::Type{S}) =
    IdentityOperator{S}(promote_eltype(src(op), S), promote_eltype(dest(op), S))

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

diagonal(op::IdentityOperator) = ones(eltype(op), length(src(op)))

apply_inplace!(op::IdentityOperator, coef_srcdest) = coef_srcdest


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
    ScalingOperator{S}(promote_eltype(src(op),S), promote_eltype(dest(op),S), S(op.scalar))

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
        # Can we assume that coef_dest and coef_src can both be indexed by i?
        # TODO: do we need eachindex(coef_src, coef_dest)?
        coef_dest[i] = op.scalar * coef_src[i]
    end
    coef_dest
end

for op in (:+, :*, :-, :/)
    @eval $op(op1::ScalingOperator, op2::ScalingOperator) =
        (@assert size(op1) == size(op2); ScalingOperator(src(op1), dest(op1), $op(scalar(op1),scalar(op2))))
end

diagonal(op::ScalingOperator) = [scalar(op) for i in 1:length(src(op))]

function matrix!(op::ScalingOperator, a)
    a[:] = 0
    for i in 1:min(size(a,1),size(a,2))
        a[i,i] = scalar(op)
    end
    a
end

# default implementation for scalar multiplication is a scaling operator
*(scalar::Number, op::AbstractOperator) = ScalingOperator(dest(op),scalar) * op


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

matrix!(op::ZeroOperator, a) = (fill!(a, 0); a)

apply_inplace!(op::ZeroOperator, coef_srcdest) = (fill!(coef_srcdest, 0); coef_srcdest)

diagonal(op::ZeroOperator) = zeros(eltype(op), min(length(src(op)), length(dest(op))))





"""
A diagonal operator is represented by a diagonal matrix.

Several other operators can be converted into a diagonal matrix, and this
conversion happens automatically when such operators are combined into a composite
operator.
"""
immutable DiagonalOperator{ELT} <: AbstractOperator{ELT}
    src         ::  FunctionSet
    dest        ::  FunctionSet
    # We store the diagonal in a vector
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


##################################
# Promotions and simplifications
##################################

## PROMOTION RULES

promote_rule{S,T}(::Type{IdentityOperator{S}}, ::Type{IdentityOperator{T}}) = IdentityOperator{promote_type(S,T)}
promote_rule{S,T}(::Type{ScalingOperator{S}}, ::Type{IdentityOperator{T}}) = ScalingOperator{promote_type(S,T)}
promote_rule{S,T}(::Type{DiagonalOperator{S}}, ::Type{IdentityOperator{T}}) = DiagonalOperator{promote_type(S,T)}

promote_rule{S,T}(::Type{ScalingOperator{S}}, ::Type{ScalingOperator{T}}) = ScalingOperator{promote_type(S,T)}
promote_rule{S,T}(::Type{DiagonalOperator{S}}, ::Type{ScalingOperator{T}}) = DiagonalOperator{promote_type(S,T)}

promote_rule{S,T}(::Type{ScalingOperator{S}}, ::Type{ZeroOperator{T}}) = ScalingOperator{promote_type(S,T)}
promote_rule{S,T}(::Type{DiagonalOperator{S}}, ::Type{ZeroOperator{T}}) = DiagonalOperator{promote_type(S,T)}

## CONVERSIONS

convert{S,T}(::Type{IdentityOperator{S}}, op::IdentityOperator{T}) = promote_eltype(op, S)
convert{S,T}(::Type{ScalingOperator{S}}, op::IdentityOperator{T}) =
    ScalingOperator(src(op), dest(op), one(S))
convert{S,T}(::Type{DiagonalOperator{S}}, op::IdentityOperator{T}) =
    DiagonalOperator(src(op), dest(op), ones(S,length(src(op))))

convert{S,T}(::Type{ScalingOperator{S}}, op::ScalingOperator{T}) = promote_eltype(op, S)
convert{S,T}(::Type{DiagonalOperator{S}}, op::ScalingOperator{T}) =
    DiagonalOperator(src(op), dest(op), S(scalar(op))*ones(S,length(src(op))))

convert{S,T}(::Type{ZeroOperator{S}}, op::ZeroOperator{T}) = promote_eltype(op, S)
convert{S,T}(::Type{DiagonalOperator{S}}, op::ZeroOperator{T}) =
    DiagonalOperator(src(op), dest(op), zeros(S,length(src(op))))

convert{S,T}(::Type{DiagonalOperator{S}}, op::DiagonalOperator{T}) = promote_eltype(op, S)


## SIMPLIFICATIONS

# The simplify routine is used to simplify the action of composite operators.
# It is called with one or two arguments. The result should be an operator
# (or a tuple of operators in case of two arguments) that does the same thing,
# but may be faster to apply. The correct chaining of src and destination sets
# for consecutive operators is verified before simplification, and does not need
# to be respected afterwards. Only the sizes (and representations) of the coefficients
# must match.

# The identity operator is, well, the identity
simplify(op::IdentityOperator) = nothing
simplify(op1::IdentityOperator, op2::IdentityOperator) = (op1,)
simplify(op1::IdentityOperator, op2::AbstractOperator) = (op2,)
simplify(op1::AbstractOperator, op2::IdentityOperator) = (op1,)

(*)(a::Number, op::IdentityOperator) = ScalingOperator(src(op), dest(op), a)
(*)(op::IdentityOperator, a::Number) = a*op

# The zero operator annihilates all other operators
simplify(op1::ZeroOperator, op2::ZeroOperator) = (op1,)
simplify(op1::ZeroOperator, op2::AbstractOperator) = (op1,)
simplify(op1::AbstractOperator, op2::ZeroOperator) = (op2,)



(*)(op1::DiagonalOperator, op2::DiagonalOperator) = DiagonalOperator(src(op1), dest(op1), diagonal(op1) .* diagonal(op2))
(*)(op1::ScalingOperator, op2::DiagonalOperator) = DiagonalOperator(src(op1), dest(op1), scalar(op1) * diagonal(op2))
(*)(op2::DiagonalOperator, op1::ScalingOperator) = op1 * op2

(+)(op1::DiagonalOperator, op2::DiagonalOperator) = DiagonalOperator(src(op1), dest(op1), diagonal(op1) + diagonal(op2))
(+)(op1::ScalingOperator, op2::DiagonalOperator) = DiagonalOperator(src(op1), dest(op1), scalar(op1) +  diagonal(op2))
(+)(op2::DiagonalOperator, op1::ScalingOperator) = op1 + op2
