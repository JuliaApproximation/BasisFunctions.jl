# circulant_operator.jl

"""
A circulant operator is represented by a circulant matrix.

Several other operators can be converted into a circulant matrix, and this
conversion happens automatically when such operators are combined into a composite
operator.
"""
struct CirculantOperator{T} <: DerivedOperator{T}
    superoperator       :: AbstractOperator
    eigenvaluematrix    :: PseudoDiagonalOperator
end

function CirculantOperator{T <: Real}(src::Span{T}, dest::Span{T}, firstcolumn::AbstractVector{T}; options...)
    D = PseudoDiagonalOperator(src, dest, fft(firstcolumn))
    CirculantOperator(src, dest, D; options...)
end

function CirculantOperator{T <: Complex}(complex_src::Span{T}, complex_dest::Span{T}, firstcolumn::AbstractVector; options...)
    D = PseudoDiagonalOperator(complex_src, complex_dest,fftw_operator(complex_src,complex_dest,1:1,FFTW.MEASURE)*firstcolumn)
    CirculantOperator(complex_src, complex_dest, D; options...)
end

function CirculantOperator{T<:Real}(src::Span{T}, dest::Span{T}, D::PseudoDiagonalOperator; options...)
    # Csrc = ComplexifyOperator(src)
    # Cdest = ComplexifyOperator(dest)
    complex_src = complex(src)
    complex_dest = complex(dest)

    # complex_src = BasisFunctions.dest(Csrc)
    # complex_dest = BasisFunctions.dest(Cdest)

    F = forward_fourier_operator(complex_src, complex_src, coeftype(complex_src); options...)
    iF = backward_fourier_operator(complex_dest, complex_dest, coeftype(complex_dest); options...)

    # R = inv(Cdest)
    CirculantOperator{T}(iF*D*F, D)
end

function CirculantOperator{T<:Complex}(complex_src::Span{T}, complex_dest::Span{T}, D::PseudoDiagonalOperator; options...)
    F = forward_fourier_operator(complex_src, complex_src, eltype(complex_src); options...)
    iF = backward_fourier_operator(complex_dest, complex_dest, eltype(complex_dest); options...)
    CirculantOperator{T}(iF*D*F, D)
end

CirculantOperator{T}(src::Span, firstcolumn::AbstractVector{T}; options...) = CirculantOperator(src, src, firstcolumn; options...)

CirculantOperator(firstcolumn::AbstractVector; options...) = CirculantOperator(Span(DiscreteSet(length(firstcolumn)), eltype(firstcolumn)), firstcolumn; options...)

function CirculantOperator{T}(op::AbstractOperator{T})
    e = zeros(T, size(op,1))
    e[1] = 1
    C = CirculantOperator(src(op), dest(op), op*e)
    e = map(T,rand(size(op,1)))
    @assert C*e≈op*e
    C
end

eigenvalues(C::CirculantOperator) = diagonal(C.eigenvaluematrix)

op_promote_eltype{ELT,S}(op::CirculantOperator{ELT}, ::Type{S}) =
    CirculantOperator{S}(src(op), dest(op), op_promote_eltype(op.eigenvaluematrix, complex(S)))

Base.sqrt(c::CirculantOperator{T}) where {T} = CirculantOperator(src(c), dest(c), PseudoDiagonalOperator(sqrt.(eigenvalues(c))))

for op in (:inv, :ctranspose)
    @eval $op{T}(C::CirculantOperator{T}) = CirculantOperator{T}($op(superoperator(C)), $op(C.eigenvaluematrix))
end

for op in (:+, :-, :*)
    @eval $op{T}(c1::CirculantOperator{T}, c2::CirculantOperator{T}) = CirculantOperator(src(c2), dest(c1), PseudoDiagonalOperator($(op).(eigenvalues(c1),eigenvalues(c2))))
end

*(scalar::Real, c::CirculantOperator) = CirculantOperator(src(c), dest(c), PseudoDiagonalOperator(scalar*eigenvalues(c)))
*(c::CirculantOperator, scalar::Real) = scalar*c
