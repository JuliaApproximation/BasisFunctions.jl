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
