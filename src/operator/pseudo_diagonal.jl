# pseudo_diagonal.jl

struct PseudoDiagonalOperator{T} <: DerivedOperator{T}
    superoperator   :: DiagonalOperator{T}
    tolerance       :: T
end

PseudoDiagonalOperator(diagonal::AbstractVector) = PseudoDiagonalOperator(DiagonalOperator(diagonal))

PseudoDiagonalOperator(src::Span, diagonal::AbstractVector) = PseudoDiagonalOperator(DiagonalOperator(src, diagonal))

PseudoDiagonalOperator(src::Span, dest::Span, diagonal::AbstractVector) = PseudoDiagonalOperator(DiagonalOperator(src, dest, diagonal))

PseudoDiagonalOperator(op::DiagonalOperator{T}, tolerance = default_tolerance(T)) where {T} =
    PseudoDiagonalOperator{T}(op, tolerance)

similar_operator(op::PseudoDiagonalOperator, ::Type{S}, src, dest) where {S} =
    PseudoDiagonalOperator(similar_operator(op.superoperator, S, src, dest))

default_tolerance(::Type{T}) where {T <: Number} = sqrt(eps(real(T)))

tolerance(op::PseudoDiagonalOperator) = op.tolerance

maxtolerance(op1::PseudoDiagonalOperator, op2::PseudoDiagonalOperator) = max(tolerance(op1), tolerance(op2))

function inv(op::PseudoDiagonalOperator)
    T = eltype(op)
    diag = diagonal(op)
    for i in 1:length(diag)
        abs(diag[i]) < tolerance(op) ? diag[i] = zero(T) : diag[i] = T(one(T)/diag[i])
    end
    PseudoDiagonalOperator(DiagonalOperator(dest(op), src(op), diag), tolerance(op))
end

function ctranspose(op::PseudoDiagonalOperator)
    PseudoDiagonalOperator(ctranspose(superoperator(op)), tolerance(op))
end


(*)(op1::PseudoDiagonalOperator, op2::PseudoDiagonalOperator) = PseudoDiagonalOperator(src(op1), diagonal(op1) .* diagonal(op2), maxtolerance(op1, op2))
(*)(op1::ScalingOperator, op2::PseudoDiagonalOperator) = PseudoDiagonalOperator(src(op1), scalar(op1) * diagonal(op2), tolerance(op2))
(*)(op2::PseudoDiagonalOperator, op1::ScalingOperator) = op1 * op2

(*)(op1::DiagonalOperator, op2::PseudoDiagonalOperator) = PseudoDiagonalOperator(src(op1), diagonal(op1) .* diagonal(op2), maxtolerance(op1, op2))
(*)(op1::PseudoDiagonalOperator, op2::DiagonalOperator) = op1 * op2

(+)(op1::PseudoDiagonalOperator, op2::PseudoDiagonalOperator) = PseudoDiagonalOperator(src(op1), dest(op1), diagonal(op1) + diagonal(op2), maxtolerance(op1, op2))
(+)(op1::ScalingOperator, op2::PseudoDiagonalOperator) = PseudoDiagonalOperator(src(op1), dest(op1), scalar(op1) +  diagonal(op2), tolerance(op2))
(+)(op2::PseudoDiagonalOperator, op1::ScalingOperator) = op1 + op2

(+)(op1::DiagonalOperator, op2::PseudoDiagonalOperator) = PseudoDiagonalOperator(src(op1), dest(op1), diagonal(op1) + diagonal(op2), tolerance(op2))
(+)(op2::PseudoDiagonalOperator, op1::DiagonalOperator) = op1 + op2

promote_rule(::Type{PseudoDiagonalOperator{S}}, ::Type{IdentityOperator{T}}) where {S,T} = PseudoDiagonalOperator{promote_type(S,T)}
promote_rule(::Type{PseudoDiagonalOperator{S}}, ::Type{ScalingOperator{T}}) where {S,T} = PseudoDiagonalOperator{promote_type(S,T)}
promote_rule(::Type{PseudoDiagonalOperator{S}}, ::Type{ZeroOperator{T}}) where {S,T} = PseudoDiagonalOperator{promote_type(S,T)}
promote_rule(::Type{PseudoDiagonalOperator{S}}, ::Type{DiagonalOperator{T}}) where {S,T} = PseudoDiagonalOperator{promote_type(S,T)}


## CONVERSIONS

convert(::Type{PseudoDiagonalOperator{S}}, op::IdentityOperator{T}) where {S,T} =
    PseudoDiagonalOperator(src(op), dest(op), ones(S,length(src(op))), default_tolerance(op))

convert(::Type{PseudoDiagonalOperator{S}}, op::ScalingOperator{T}) where {S,T} =
    PseudoDiagonalOperator(src(op), dest(op), S(scalar(op))*ones(S,length(src(op))), default_tolerance(op))

convert(::Type{PseudoDiagonalOperator{S}}, op::DiagonalOperator{T}) where {S,T} =
    PseudoDiagonalOperator(src(op), dest(op), map(S, diagonal(op)), default_tolerance(op))

convert(::Type{PseudoDiagonalOperator{S}}, op::PseudoDiagonalOperator{T}) where {S,T} = promote_eltype(op, S)
