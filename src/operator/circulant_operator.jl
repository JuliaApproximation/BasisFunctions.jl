# circulant_operator.jl

"""
A circulant operator is represented by a circulant matrix.

Several other operators can be converted into a circulant matrix, and this
conversion happens automatically when such operators are combined into a composite
operator.
"""
struct CirculantOperator{T} <: DerivedOperator{T}
    src                 :: Span
    dest                :: Span
    superoperator       :: AbstractOperator
    eigenvaluematrix    :: DiagonalOperator
    scratch

    CirculantOperator{T}(op_src::Span, op_dest::Span, op::AbstractOperator, opD::DiagonalOperator) where {T} =
      new(op_src, op_dest, op, opD, zeros(dest(op)))
end

src(c::CirculantOperator) = c.src
dest(c::CirculantOperator) = c.dest

function CirculantOperator(firstcolumn::AbstractVector; options...)
    T = eltype(firstcolumn)
    CirculantOperator(Span(DiscreteVectorSet{T}(length(firstcolumn))), firstcolumn; options...)
end

CirculantOperator(src::Span, firstcolumn::AbstractVector; options...) = CirculantOperator(src, src, firstcolumn; options...)

# TODO: Make CirculantOperator between real type sets work again... currently everything is transformed into
# complex becase Realification is not a linear operation
function CirculantOperator(op_src::Span, op_dest::Span, firstcolumn::AbstractVector; options...)
    D = DiagonalOperator(op_src, op_dest, fft(firstcolumn))
    # Using src(D) and dest(D) ensures that they have complex types, because the fft
    # in the line above will result in complex numbers
    # Note that this makes all CirculantOperators complex! TODO: fix
    CirculantOperator(op_src, op_dest, D; options...)
end

CirculantOperator(src::Span, dest::Span, D::DiagonalOperator; options...) =
    CirculantOperator(eltype(D), src, dest, D; options...)

function CirculantOperator(::Type{T}, op_src::Span, op_dest::Span, opD::DiagonalOperator;
  realify_circulant_operator=true, real_circulant_tol=sqrt(eps(real(T))), verbose=false, options...) where {T}
    cpx_src = src(opD)
    cpx_dest = dest(opD)

    S, D, A = op_eltypes(cpx_src, cpx_dest, T)
    c_src = promote_coeftype(cpx_src, S)
    c_dest = promote_coeftype(cpx_dest, D)
    c_D = similar_operator(opD, A, c_src, c_dest)
    F = forward_fourier_operator(c_src, c_src, A; verbose=verbose, options...)
    iF = backward_fourier_operator(c_dest, c_dest, A; verbose=verbose, options...)
    iF = wrap_operator(c_dest,c_dest,inv(F))
    #realify a circulant operator if asked, both spans are real and if it contains almost real elements
    if realify_circulant_operator && isreal(op_src) && isreal(op_dest)
        imag_norm = Base.norm(imag(fft(diagonal(opD))))
        if  imag_norm < real_circulant_tol
            verbose && warn("realified circulant operator, lost an accuracy of $(imag_norm)")
            r_S, r_D, r_A = op_eltypes(op_src, op_dest, real(T))
            r_src = promote_coeftype(op_src, r_S)
            r_dest = promote_coeftype(op_dest, r_D)

            return CirculantOperator{r_A}(r_src, r_dest, iF*c_D*F, c_D)
        else
            verbose && warn("was not able to realify circulant operator")
        end
    end
    CirculantOperator{A}(src(F), dest(iF), iF*c_D*F, c_D)
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

Base.sqrt(c::CirculantOperator{T}) where {T} = CirculantOperator(src(c), dest(c), DiagonalOperator(sqrt.(eigenvalues(c))))

for op in (:inv, :ctranspose)
    @eval $op{T}(C::CirculantOperator{T}) = CirculantOperator{T}(dest(C), src(C), $op(superoperator(C)), $op(C.eigenvaluematrix))
end

# What tolerance should be used for the pinv here?
pinv(C::CirculantOperator{T}, tolerance = eps(numtype(C))) where {T} = CirculantOperator(src(c), dest(c), DiagonalOperator(pinv(C.eigenvaluematrix), tolerance))

for op in (:+, :-, :*)
    @eval $op{T}(c1::CirculantOperator{T}, c2::CirculantOperator{T}) = CirculantOperator(src(c2), dest(c1), DiagonalOperator($(op).(eigenvalues(c1),eigenvalues(c2))))
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
    coef_dest[abs.(coef_dest).<sparse_tol] = 0
    R = hcat(R,sparse(coef_dest))
    for i in 2:length(dest(op))
        circshift!(coef_dest_1, coef_dest, 1)
        R = hcat(R,sparse(coef_dest_1))
        copy!(coef_dest, coef_dest_1)
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
