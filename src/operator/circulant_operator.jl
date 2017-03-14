"""
A circulant operator is represented by a circulant matrix.

Several other operators can be converted into a circulant matrix, and this
conversion happens automatically when such operators are combined into a composite
operator.
"""
immutable CirculantOperator{T} <: DerivedOperator{T}
  superoperator   :: AbstractOperator

  eigenvaluematrix  :: PseudoDiagonalOperator
end

function CirculantOperator{N,ELT <: Real}(src::FunctionSet{N,ELT}, dest::FunctionSet{N,ELT}, firstcolumn::AbstractVector{ELT}; options...)
    D = PseudoDiagonalOperator(promote_eltype(src,complex(eltype(src))), promote_eltype(src,complex(eltype(dest))), fft(firstcolumn))
    CirculantOperator(src, dest, D; options...)
end

function CirculantOperator{N,ELT <: Complex}(complex_src::FunctionSet{N,ELT}, complex_dest::FunctionSet{N,ELT}, firstcolumn::AbstractVector{ELT}; options...)
    D = PseudoDiagonalOperator(complex_src, complex_dest,fftw_operator(complex_src,complex_dest,1:1,FFTW.MEASURE)*firstcolumn)
    CirculantOperator(complex_src, complex_dest, D; options...)
end

function CirculantOperator{N,T<:Real}(src::FunctionSet{N,T}, dest::FunctionSet{N,T}, D::PseudoDiagonalOperator; options...)
  Csrc = ComplexifyOperator(src)
  Cdest = ComplexifyOperator(dest)

  complex_src = BasisFunctions.dest(Csrc)
  complex_dest = BasisFunctions.dest(Cdest)

  F = forward_fourier_operator(complex_src, complex_src, eltype(complex_src); options...)
  iF = backward_fourier_operator(complex_dest, complex_dest, eltype(complex_dest); options...)

  R = inv(Cdest)
  CirculantOperator{T}(R*iF*D*F*Csrc, D)
end

function CirculantOperator{N,T<:Complex}(complex_src::FunctionSet{N,T}, complex_dest::FunctionSet{N,T}, D::PseudoDiagonalOperator; options...)
    F = forward_fourier_operator(complex_src, complex_src, eltype(complex_src); options...)
    iF = backward_fourier_operator(complex_dest, complex_dest, eltype(complex_dest); options...)
    CirculantOperator{T}(iF*D*F, D)
end

CirculantOperator{ELT}(src::FunctionSet, firstcolumn::AbstractVector{ELT}; options...) = CirculantOperator(src, src, firstcolumn; options...)

CirculantOperator{T <: Real}(firstcolumn::AbstractVector{T}; options...) = CirculantOperator(Rn{T}(length(firstcolumn)), firstcolumn; options...)
CirculantOperator{T <: Complex}(firstcolumn::AbstractVector{T}; options...) = CirculantOperator(Cn{T}(length(firstcolumn)), firstcolumn; options...)

eigenvalues(C::CirculantOperator) = diagonal(C.eigenvaluematrix)

op_promote_eltype{ELT,S}(op::CirculantOperator{ELT}, ::Type{S}) =
    CirculantOperator(set_promote_eltype(src(op), S), set_promote_eltype(dest(op), S), op_promote_eltype(op.eigenvaluematrix, complex(S)))

for op in (:inv, :ctranspose)
  @eval $op{T}(C::CirculantOperator{T}) = CirculantOperator{T}($op(superoperator(C)), $op(C.eigenvaluematrix))
end

for op in (:+, :-, :*)
  @eval $op{T}(c1::CirculantOperator{T}, c2::CirculantOperator{T}) = CirculantOperator(src(c2), dest(c1), PseudoDiagonalOperator($(op).(eigenvalues(c1),eigenvalues(c2))))
end

*(scalar::Real, c::CirculantOperator) = CirculantOperator(src(c), dest(c), PseudoDiagonalOperator(scalar*eigenvalues(c)))
*(c::CirculantOperator, scalar::Real) = scalar*c
