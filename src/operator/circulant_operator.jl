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

CirculantOperator(firstcolumn::AbstractVector; options...) =
    CirculantOperator(Span(DiscreteSet(length(firstcolumn)), eltype(firstcolumn)), firstcolumn; options...)

CirculantOperator(src::Span, firstcolumn::AbstractVector; options...) = CirculantOperator(src, src, firstcolumn; options...)

# TODO: Make CirculantOperator between real type sets work again... currently everything is transformed into
# complex becase Realification is not a linear operation
function CirculantOperator(op_src::Span, op_dest::Span, firstcolumn::AbstractVector; options...)
    D = PseudoDiagonalOperator(op_src, op_dest, fft(firstcolumn))
    # Using src(D) and dest(D) ensures that they have complex types, because the fft
    # in the live above will result in complex numbers
    # Note that this makes all CirculantOperators complex! TODO: fix
    CirculantOperator(src(D), dest(D), D; options...)
end

CirculantOperator(src::Span, dest::Span, D::PseudoDiagonalOperator; options...) =
    CirculantOperator(eltype(D), src, dest, D; options...)

function CirculantOperator(::Type{T}, src::Span, dest::Span, opD::PseudoDiagonalOperator; options...) where {T}
    S, D, A = op_eltypes(src, dest, T)
    c_src = promote_coeftype(src, S)
    c_dest = promote_coeftype(dest, D)
    c_D = similar_operator(opD, A, c_src, c_dest)
    F = forward_fourier_operator(c_src, c_src, A; options...)
    iF = backward_fourier_operator(c_dest, c_dest, A; options...)
    CirculantOperator{A}(iF*c_D*F, c_D)
end

function CirculantOperator(op::AbstractOperator{T}) where {T}
    e = zeros(T, size(op,1))
    e[1] = one(T)
    C = CirculantOperator(src(op), dest(op), op*e)
    e = map(T,rand(size(op,1)))
    @assert C*e≈op*e
    C
end

eigenvalues(C::CirculantOperator) = diagonal(C.eigenvaluematrix)

similar_operator(op::CirculantOperator, ::Type{S}, src, dest) where {S} =
    CirculantOperator(S, src, dest, op.eigenvaluematrix)

Base.sqrt(c::CirculantOperator{T}) where {T} = CirculantOperator(src(c), dest(c), PseudoDiagonalOperator(sqrt.(eigenvalues(c))))

for op in (:inv, :ctranspose)
    @eval $op{T}(C::CirculantOperator{T}) = CirculantOperator{T}($op(superoperator(C)), $op(C.eigenvaluematrix))
end

for op in (:+, :-, :*)
    @eval $op{T}(c1::CirculantOperator{T}, c2::CirculantOperator{T}) = CirculantOperator(src(c2), dest(c1), PseudoDiagonalOperator($(op).(eigenvalues(c1),eigenvalues(c2))))
end

*(scalar::Real, c::CirculantOperator) = CirculantOperator(src(c), dest(c), PseudoDiagonalOperator(scalar*eigenvalues(c)))
*(c::CirculantOperator, scalar::Real) = scalar*c
