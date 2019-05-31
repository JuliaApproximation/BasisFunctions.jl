
using Test, FFTW, BasisFunctions, LinearAlgebra, SparseArrays

# Verify types of FFT and DCT plans by FFTW
# If anything changes here, the aliases in fouriertransforms.jl have to change as well
d1 = FFTW.plan_fft!(zeros(Complex{Float64}, 10), 1:1)
@test typeof(d1) == FFTW.cFFTWPlan{Complex{Float64},-1,true,1}
d2 = FFTW.plan_fft!(zeros(Complex{Float64}, 10, 10), 1:2)
@test typeof(d2) == FFTW.cFFTWPlan{Complex{Float64},-1,true,2}
d3 = FFTW.plan_bfft!(zeros(Complex{Float64}, 10), 1:1)
@test typeof(d3) == FFTW.cFFTWPlan{Complex{Float64},1,true,1}
d4 = FFTW.plan_bfft!(zeros(Complex{Float64}, 10, 10), 1:2)
@test typeof(d4) == FFTW.cFFTWPlan{Complex{Float64},1,true,2}

d5 = FFTW.plan_dct!(zeros(10), 1:1)
@test typeof(d5) == FFTW.DCTPlan{Float64,5,true}
d6 = FFTW.plan_idct!(zeros(10), 1:1)
@test typeof(d6) == FFTW.DCTPlan{Float64,4,true}

D = Diagonal(1:400)
R = BasisFunctions.RestrictionIndexMatrix{Float64}((400,),2:3)
E = BasisFunctions.ExtensionIndexMatrix{Float64}((400,),2:3)
@test R' == E
@test E' == R
@test R*D == Matrix(R)*Matrix(D)
@test D*E == Matrix(D)*Matrix(E)
@test R*D isa AbstractSparseArray
@test D*E isa AbstractSparseArray

V = BasisFunctions.VerticalBandedMatrix(400,200, collect(1:3))
H = BasisFunctions.HorizontalBandedMatrix(200,400, collect(1:3))
@test V' == H
@test H' == V
@test R*V == Matrix(R)*Matrix(V)
@test H*E == Matrix(H)*Matrix(E)
@test R*V isa AbstractSparseArray
@test H*E isa AbstractSparseArray

B = rand(400,2)
@test R*B == Matrix(R)*Matrix(B)
B = rand(2,400)
@test B*E == Matrix(B)*Matrix(E)

include("test_normalization.jl")
include("test_space.jl")
