# pseudo_diagonal.jl
immutable PseudoDiagonalOperator{T} <: DerivedOperator{T}
  superoperator   :: DiagonalOperator{T}
  tolerance       :: Real
end

PseudoDiagonalOperator{T <: Real}(diagonal::AbstractVector{T}) = PseudoDiagonalOperator(Rn{T}(length(diagonal)), diagonal)
PseudoDiagonalOperator{T <: Complex}(diagonal::AbstractVector{T}) = PseudoDiagonalOperator(Cn{T}(length(diagonal)), diagonal)

PseudoDiagonalOperator{ELT}(src::FunctionSet, diagonal::AbstractVector{ELT}, tolerance = default_tolerance(ELT)) =
    PseudoDiagonalOperator(src, src, diagonal, tolerance)

PseudoDiagonalOperator{ELT}(src::FunctionSet, dest::FunctionSet, diagonal::AbstractVector{ELT}, tolerance = default_tolerance(ELT)) =
    PseudoDiagonalOperator{ELT}(DiagonalOperator{ELT}(src, dest, diagonal), tolerance)

op_promote_eltype{ELT,S}(op::PseudoDiagonalOperator{ELT}, ::Type{S}) =
    PseudoDiagonalOperator{S}(promote_eltype(superoperator(op), S), tolerance(S))

default_tolerance{T}(op::AbstractOperator{T}) = default_tolerance(T)

default_tolerance{T}(::Type{T}) = sqrt(eps(real(T)))

tolerance(op::PseudoDiagonalOperator) = op.tolerance

tolerance(op1::PseudoDiagonalOperator, op2::PseudoDiagonalOperator) = max(tolerance(op1), tolerance(op2))

function inv{T}(op::PseudoDiagonalOperator{T})
  diag = diagonal(op)
  for i in 1:length(diag)
    abs(diag[i]) < tolerance(op) ? diag[i] = T(0) : diag[i] = T(T(1)/diag[i])
  end
  PseudoDiagonalOperator(DiagonalOperator(dest(op), src(op), diag), tolerance(op))
end

function ctranspose{T}(op::PseudoDiagonalOperator{T})
  PseudoDiagonalOperator(ctranspose(superoperator(op)), tolerance(op))
end


(*)(op1::PseudoDiagonalOperator, op2::PseudoDiagonalOperator) = PseudoDiagonalOperator(src(op1), diagonal(op1) .* diagonal(op2), tolerance(op1, op2))
(*)(op1::ScalingOperator, op2::PseudoDiagonalOperator) = PseudoDiagonalOperator(src(op1), scalar(op1) * diagonal(op2), tolerance(op2))
(*)(op2::PseudoDiagonalOperator, op1::ScalingOperator) = op1 * op2

(*)(op1::DiagonalOperator, op2::PseudoDiagonalOperator) = PseudoDiagonalOperator(src(op1), diagonal(op1) .* diagonal(op2), tolerance(op1op2))
(*)(op1::PseudoDiagonalOperator, op2::DiagonalOperator) = op1 * op2

(+)(op1::PseudoDiagonalOperator, op2::PseudoDiagonalOperator) = PseudoDiagonalOperator(src(op1), dest(op1), diagonal(op1) + diagonal(op2), tolerance(op1, op2))
(+)(op1::ScalingOperator, op2::PseudoDiagonalOperator) = PseudoDiagonalOperator(src(op1), dest(op1), scalar(op1) +  diagonal(op2), tolerance(op2))
(+)(op2::PseudoDiagonalOperator, op1::ScalingOperator) = op1 + op2

(+)(op1::DiagonalOperator, op2::PseudoDiagonalOperator) = PseudoDiagonalOperator(src(op1), dest(op1), diagonal(op1) + diagonal(op2), tolerance(op2))
(+)(op2::PseudoDiagonalOperator, op1::DiagonalOperator) = op1 + op2

promote_rule{S,T}(::Type{PseudoDiagonalOperator{S}}, ::Type{IdentityOperator{T}}) = PseudoDiagonalOperator{promote_type(S,T)}

promote_rule{S,T}(::Type{PseudoDiagonalOperator{S}}, ::Type{ScalingOperator{T}}) = PseudoDiagonalOperator{promote_type(S,T)}

promote_rule{S,T}(::Type{PseudoDiagonalOperator{S}}, ::Type{ZeroOperator{T}}) = PseudoDiagonalOperator{promote_type(S,T)}

promote_rule{S,T}(::Type{PseudoDiagonalOperator{S}}, ::Type{DiagonalOperator{T}}) = PseudoDiagonalOperator{promote_type(S,T)}
## CONVERSIONS

convert{S,T}(::Type{PseudoDiagonalOperator{S}}, op::IdentityOperator{T}) =
    PseudoDiagonalOperator(src(op), dest(op), ones(S,length(src(op))), default_tolerance(op))

convert{S,T}(::Type{PseudoDiagonalOperator{S}}, op::ScalingOperator{T}) =
    PseudoDiagonalOperator(src(op), dest(op), S(scalar(op))*ones(S,length(src(op))), default_tolerance(op))

convert{S,T}(::Type{PseudoDiagonalOperator{S}}, op::DiagonalOperator{T}) =
    PseudoDiagonalOperator(src(op), dest(op), map(S, diagonal(op)), default_tolerance(op))

convert{S,T}(::Type{PseudoDiagonalOperator{S}}, op::PseudoDiagonalOperator{T}) = promote_eltype(op, S)
