
using Test, FFTW, BasisFunctions, LinearAlgebra, SparseArrays

# Verify types of FFT and DCT plans by FFTW
# If anything changes here, the aliases in fouriertransforms.jl have to change as well
d1 = FFTW.plan_fft!(zeros(Complex{Float64}, 10), 1:1)
@test d1 isa FFTW.cFFTWPlan{Complex{Float64},-1,true,1,UnitRange{Int64}}
@test d1 isa BasisFunctions.FFTPLAN
d2 = FFTW.plan_fft!(zeros(Complex{Float64}, 10, 10), 1:2)
@test d2 isa FFTW.cFFTWPlan{Complex{Float64},-1,true,2,UnitRange{Int64}}
@test d2 isa BasisFunctions.FFTPLAN
d3 = FFTW.plan_bfft!(zeros(Complex{Float64}, 10), 1:1)
@test d3 isa FFTW.cFFTWPlan{Complex{Float64},1,true,1,UnitRange{Int64}}
@test d3 isa BasisFunctions.IFFTPLAN
d4 = FFTW.plan_bfft!(zeros(Complex{Float64}, 10, 10), 1:2)
@test d4 isa FFTW.cFFTWPlan{Complex{Float64},1,true,2,UnitRange{Int64}}
@test d4 isa BasisFunctions.IFFTPLAN

d5 = FFTW.plan_dct!(zeros(10), 1:1)
@test d5 isa FFTW.DCTPlan{Float64,5,true}
@test d5 isa BasisFunctions.DCTII_PLAN
d6 = FFTW.plan_idct!(zeros(10), 1:1)
@test d6 isa FFTW.DCTPlan{Float64,4,true}
@test d6 isa BasisFunctions.INV_DCTII_PLAN

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

include("test_common.jl")
include("test_normalization.jl")
include("test_space.jl")
include("test_discrete_measure.jl")
include("test_measure.jl")
