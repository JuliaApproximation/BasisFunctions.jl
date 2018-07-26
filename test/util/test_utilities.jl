
if VERSION < v"0.7-"
    using Base.Test, BasisFunctions, BasisFunctions.Test
else
    using Test, FFTW
end

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

include("test_multiarray.jl")
