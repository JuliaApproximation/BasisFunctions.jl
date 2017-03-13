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


# immutable SelectOperator{T} <: AbstractOperator{T}
#   src     :: FunctionSet
#   dest    :: FunctionSet
#   offset  :: Int
#   m       :: Int
# end

# SelectOperator{N,T}(src::FunctionSet{N,T}, dest::FunctionSet{N,T}, offset::Int=1, m::Int=2) =
#   IndexRestrictionOperator(src, dest, offset:m:length(src))

# function SelectOperator{N,T}(src::FunctionSet{N,T}, dest::FunctionSet{N,T}, offset::Int=1, m::Int=2)
#   @assert length(src) >= length(dest)
#   SelectOperator{T}(src, dest, offset, m)
# end

# SelectOperator(src::FunctionSet, offset::Int=1, m::Int=2) = SelectOperator(src, resize(src, length(src)>>1), offset, m)
#
# SelectOperator{T}(s1::Int, s2::Int, offset::Int=1, m::Int=2, ::Type{T}= Float64) = SelectOperator(Rn{T}(s1),Rn{T}(s2), offset, m)
#
# function BasisFunctions.apply!(op::SelectOperator, coef_dest, coef_src)
#   for i in 1:length(coef_dest)
#       coef_dest[i] = coef_src[op.offset+op.m*(i-1)]
#   end
# end

# Base.inv(op::SelectOperator) = ExpandOperator(dest(op), src(op), op.offset, op.m)
# Base.ctranspose(op::SelectOperator) = inv(op)

# immutable ExpandOperator{T} <: AbstractOperator{T}
#   src     :: FunctionSet
#   dest    :: FunctionSet
#   offset  :: Int
#   m       :: Int
# end

# ExpandOperator{N,T}(src::FunctionSet{N,T}, dest::FunctionSet{N,T}, offset::Int=1, m::Int=2) =
#     IndexExtensionOperator(src, dest, offset:m:length(dest))

# ExpandOperator{N,T}(src::FunctionSet{N,T}, dest::FunctionSet{N,T}, offset::Int=1, m::Int=2) =
#     ExpandOperator{T}(src, dest, offset, m)
#
# ExpandOperator(src::FunctionSet, offset::Int=1, m::Int=2) = ExpandOperator(src, resize(src, length(src)<<1), offset, m)
#
# ExpandOperator{T}(s1::Int, s2::Int, offset::Int=1, m::Int=2, ::Type{T}= Float64) = ExpandOperator(Rn{T}(s1),Rn{T}(s2), offset, m)
#
# function BasisFunctions.apply!(op::ExpandOperator, coef_dest, coef_src)
#   coef_dest[:] = 0
#   for i in 1:length(coef_src)
#       coef_dest[op.offset+op.m*(i-1)] = coef_src[i]
#   end
# end
#
# Base.inv(op::ExpandOperator) = SelectOperator(dest(op), src(op), op.offset, op.m)
# Base.ctranspose(op::ExpandOperator) = inv(op)
#
