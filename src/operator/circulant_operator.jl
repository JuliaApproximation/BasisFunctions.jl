
"""
A circulant operator is represented by a circulant matrix.

Several other operators can be converted into a circulant matrix, and this
conversion happens automatically when such operators are combined into a composite
operator.
"""
struct CirculantOperator{T} <: DerivedOperator{T}
    src                 :: Dictionary
    dest                :: Dictionary
    superoperator       :: DictionaryOperator
    eigenvaluematrix    :: DiagonalOperator
    scratch

    CirculantOperator{T}(op_src::Dictionary, op_dest::Dictionary, op::DictionaryOperator, opD::DiagonalOperator) where {T} =
      new(op_src, op_dest, op, opD, zeros(dest(op)))
end

src(c::CirculantOperator) = c.src
dest(c::CirculantOperator) = c.dest

function CirculantOperator(firstcolumn::AbstractVector; options...)
    T = eltype(firstcolumn)
    CirculantOperator(DiscreteVectorDictionary{T}(length(firstcolumn)), firstcolumn; options...)
end

CirculantOperator(src::Dictionary, firstcolumn::AbstractVector; options...) = CirculantOperator(src, src, firstcolumn; options...)

function CirculantOperator(op_src::Dictionary, op_dest::Dictionary, firstcolumn::AbstractVector; options...)
    Dsrc = DiscreteVectorDictionary{complex(eltype(firstcolumn))}(length(firstcolumn))
    D = DiagonalOperator(Dsrc, Dsrc, fft(firstcolumn))
    CirculantOperator(op_src, op_dest, D; options...)
end

CirculantOperator(src::Dictionary, dest::Dictionary, D::DiagonalOperator; options...) = CirculantOperator(eltype(D), src, dest, D; options...)

CirculantOperator(::Type{S}, op_src::Dictionary, op_dest::Dictionary, opD::DiagonalOperator;
            T=promote_type(eltype(opD),op_eltype(op_src,op_dest)), options...) where {S} =
    CirculantOperator{promote_type(S,T)}(S, op_src, op_dest, opD; options...)

function CirculantOperator{T}(::Type{S}, op_src::Dictionary, op_dest::Dictionary, opD::DiagonalOperator;
            real_circulant_tol=sqrt(eps(real(S))), verbose=false, options...) where {S,T}
    cpx_src = DiscreteVectorDictionary{eltype(opD)}(length(src(opD)))

    F = forward_fourier_operator(cpx_src, cpx_src, T; verbose=verbose, options...)
    iF = inv(F)
    #realify a circulant operator if src and dest are real (one should imply the other).
    if isreal(op_src) && isreal(op_dest)
        imag_norm = norm(imag(fft(diagonal(opD))))
        imag_norm > real_circulant_tol && warn("realified circulant operator, lost an accuracy of $(imag_norm)")
        r_S, r_D, r_A = op_eltypes(op_src, op_dest, real(S))
        r_src = promote_coefficienttype(op_src, r_S)
        r_dest = promote_coefficienttype(op_dest, r_D)

        return CirculantOperator{r_A}(r_src, r_dest, iF*opD*F, opD)

    end
    CirculantOperator{T}(op_src, op_dest, iF*opD*F, opD)
end

function CirculantOperator(op::DictionaryOperator{T}) where {T}
    e = zeros(T, size(op,1))
    e[1] = one(T)
    C = CirculantOperator(src(op), dest(op), op*e)
    e = rand(T, size(op,1))
    @assert C*e≈op*e
    C
end

eigenvalues(C::CirculantOperator) = diagonal(C.eigenvaluematrix)

similar_operator(op::CirculantOperator, src, dest) =
    CirculantOperator(src, dest, op.eigenvaluematrix)

Base.sqrt(c::CirculantOperator{T}) where {T} = CirculantOperator(src(c), dest(c), DiagonalOperator(sqrt.(eigenvalues(c))))

for op in (:inv, :adjoint)
    @eval $op(C::CirculantOperator{T}) where {T} = CirculantOperator{T}(dest(C), src(C), $op(superoperator(C)), $op(C.eigenvaluematrix))
end

# What tolerance should be used for the pinv here?
pinv(C::CirculantOperator{T}, tolerance = eps(numtype(C))) where {T} = CirculantOperator(src(c), dest(c), DiagonalOperator(pinv(C.eigenvaluematrix), tolerance))

for op in (:+, :-, :*)
    @eval $op(c1::CirculantOperator{T}, c2::CirculantOperator{T}) where {T} = CirculantOperator(src(c2), dest(c1), DiagonalOperator($(op).(eigenvalues(c1),eigenvalues(c2))))
end

*(scalar::Real, c::CirculantOperator) = CirculantOperator(src(c), dest(c), DiagonalOperator(scalar*eigenvalues(c)))
*(c::CirculantOperator, scalar::Real) = scalar*c

function sparse_matrix(op::CirculantOperator; sparse_tol = 1e-14, options...)
    coef_src  = zeros(src(op))
    coef_dest = zeros(dest(op))
    coef_dest_1 = zeros(dest(op))
    R = spzeros(eltype(op),size(op,1),0)
    coef_src[1] = 1
    apply!(op, coef_dest, coef_src)
    coef_dest[abs.(coef_dest).<sparse_tol] .= 0
    R = hcat(R,sparse(coef_dest))
    for i in 2:length(dest(op))
        circshift!(coef_dest_1, coef_dest, 1)
        R = hcat(R,sparse(coef_dest_1))
        copyto!(coef_dest, coef_dest_1)
    end
    R
end

apply!(c::CirculantOperator, coef_dest, coef_src) = apply!(c, coef_dest, coef_src, Val{isreal(c)})
apply!(c::CirculantOperator, coef_dest, coef_src, ::Type{Val{false}}) = apply!(superoperator(c), coef_dest, coef_src)
function apply!(c::CirculantOperator, coef_dest, coef_src, ::Type{Val{true}})
  apply!(superoperator(c), c.scratch, coef_src)
  coef_dest[:] = real(c.scratch)
end


apply_inplace!(c::CirculantOperator, coef_dest) = apply_inplace!(c, coef_dest, Val{isreal(c)})
apply_inplace!(c::CirculantOperator, coef_dest, ::Type{Val{false}}) = apply_inplace!(superoperator(c), coef_dest)
function apply_inplace!(c::CirculantOperator, coef_dest, ::Type{Val{true}})
  c.scratch[:] = coef_dest
  apply_inplace!(superoperator(c), c.scratch)
  coef_dest[:] = real(c.scratch)
end

# TODO find to do this without copying code (CirculantOperator is no DerivedOperator?)
function matrix!(op::CirculantOperator, a)
    coef_src  = zeros(src(op))
    coef_dest = zeros(dest(op))
    matrix_fill!(op, a, coef_src, coef_dest)
end
# TODO find to do this without copying code (CirculantOperator is no DerivedOperator?)
function unsafe_getindex(op::CirculantOperator, i::Int, j::Int)
  coef_src = zeros(src(op))
	coef_dest = zeros(dest(op))
	coef_src[j] = one(eltype(op))
	apply!(op, coef_dest, coef_src)
	coef_dest[i]
end
