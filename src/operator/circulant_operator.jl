"""
A circulant operator is represented by a circulant matrix.

Several other operators can be converted into a circulant matrix, and this
conversion happens automatically when such operators are combined into a composite
operator.
"""
CirculantOperator{T <: Real}(firstcolumn::AbstractVector{T}; options...) = CirculantOperator(Rn{T}(length(firstcolumn)), firstcolumn; options...)
CirculantOperator{T <: Complex}(firstcolumn::AbstractVector{T}; options...) = CirculantOperator(Cn{T}(length(firstcolumn)), firstcolumn; options...)

CirculantOperator{ELT}(src::FunctionSet, firstcolumn::AbstractVector{ELT}; options...) = CirculantOperator(src, src, firstcolumn; options...)

function CirculantOperator{ELT}(src::FunctionSet, dest::FunctionSet, firstcolumn::AbstractVector{ELT}; options...)
    Csrc = ComplexifyOperator(src)
    Cdest = ComplexifyOperator(dest)

    complex_src = BasisFunctions.dest(Csrc)
    complex_dest = BasisFunctions.dest(Cdest)

    F = forward_fourier_operator(complex_src, complex_src, eltype(complex_src); options...)
    iF = backward_fourier_operator(complex_dest, complex_dest, eltype(complex_dest); options...)
    D = PseudoDiagonalOperator(complex_src, complex_dest,fftw_operator(complex_src,complex_dest,1:1,FFTW.MEASURE)*firstcolumn)

    R = inv(Cdest)
    R*iF*D*F*Csrc
end


immutable SelectOperator{T} <: AbstractOperator{T}
  src     :: FunctionSet
  dest    :: FunctionSet
  offset  :: Int
  m       :: Int
end

SelectOperator{N,T}(src::FunctionSet{N,T}, dest::FunctionSet{N,T}, offset::Int=1, m::Int=2) =
    SelectOperator{T}(src, dest, offset, m)

SelectOperator(src::FunctionSet, offset::Int=1, m::Int=2) = SelectOperator(src, resize(src, length(src)>>1), offset, m)

function BasisFunctions.apply!(op::SelectOperator, coef_dest, coef_src)
  for i in 1:length(coef_dest)
      coef_dest[i] = coef_src[op.offset+op.m*(i-1)]
  end
end

Base.inv(op::SelectOperator) = ExpandOperator(dest(op), src(op), op.offset, op.m)
Base.ctranspose(op::SelectOperator) = inv(op)

immutable ExpandOperator{T} <: AbstractOperator{T}
  src     :: FunctionSet
  dest    :: FunctionSet
  offset  :: Int
  m       :: Int
end

ExpandOperator{N,T}(src::FunctionSet{N,T}, dest::FunctionSet{N,T}, offset::Int=1, m::Int=2) =
    ExpandOperator{T}(src, dest, offset, m)

ExpandOperator(src::FunctionSet, offset::Int=1, m::Int=2) = ExpandOperator(src, resize(src, length(src)<<1), offset, m)

function BasisFunctions.apply!(op::ExpandOperator, coef_dest, coef_src)
  coef_dest[:] = 0
  for i in 1:length(coef_src)
      coef_dest[op.offset+op.m*(i-1)] = coef_src[i]
  end
end

Base.inv(op::ExpandOperator) = SelectOperator(dest(op), src(op), op.offset, op.m)
Base.ctranspose(op::ExpandOperator) = inv(op)

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
