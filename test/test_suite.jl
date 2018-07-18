# test_suite.jl
module test_suite

using BasisFunctions, Domains, StaticArrays

if VERSION < v"0.7-"
    using Base.Test
    my_rand(T, a...) = map(T, rand(a...))
else
    using Test, Random, FFTW, LinearAlgebra
    my_rand = rand
end

srand(1234)
BF = BasisFunctions

const show_timings = false

######### #
# Testing
##########

include("util_functions.jl")
include("test_generic_grids.jl")
include("test_generic_dicts.jl")
include("test_mapped_dicts.jl")
include("test_tensors.jl")
include("test_derived_dict.jl")
include("test_operators.jl")
include("test_generic_operators.jl")
include("test_ops.jl")
include("test_fourier.jl")
include("test_discrete_sets.jl")
include("test_DCTI.jl")
include("test_gram.jl")


delimit("Utilities")

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

SETS = [FourierBasis, ChebyshevBasis, ChebyshevU, LegendrePolynomials,
        LaguerrePolynomials, HermitePolynomials, CosineSeries, SineSeries]
# SETS = [FourierBasis, ChebyshevBasis, ChebyshevU, LegendrePolynomials,
#         LaguerrePolynomials, HermitePolynomials, CosineSeries, SineSeries]
T = Float64
# for T in [Float64,BigFloat,]
println()
delimit("T is $T", )

delimit("Operators")
test_operators(T)
test_generic_operators(T)

delimit("Generic dictionary interfaces")
@testset "$(rpad("$(name(instantiate(SET,n))) with $n dof",80," "))" for SET in SETS,
        n = 9
        basis = instantiate(SET, n, T)

        @test length(basis) == n
        @test domaintype(basis) == T

        test_generic_dict_interface(basis)
end
# also try a Fourier series with an even length
test_generic_dict_interface(FourierBasis{T}(8))

test_derived_dicts(T)

delimit("Tensor specific tests")
@testset "$(rpad("test iteration",80))" begin
    test_tensor_sets(T) end

delimit("Tensor product set interfaces")
# TODO: all sets in the test below should use type T!
@testset "$(rpad("$(name(basis))",80," "))" for basis in
            ( FourierBasis(11) ⊗ FourierBasis(21), # Two odd-length Fourier series
              FourierBasis(10) ⊗ ChebyshevBasis(12), # combination of Fourier and Chebyshev
              FourierBasis(11) ⊗ FourierBasis(10), # Odd and even-length Fourier series
              ChebyshevBasis(11) ⊗ ChebyshevBasis(20), # Two Chebyshev sets
              FourierBasis(11, 2, 3) ⊗ FourierBasis(11, 4, 5), # Two mapped Fourier series
              ChebyshevBasis(9, 2, 3) ⊗ ChebyshevBasis(7, 4, 5)) # Two mapped Chebyshev series
    test_generic_dict_interface(basis)
end

delimit("Discrete sets")
@testset "$(rpad("discrete sets",80))" begin
    test_discrete_sets(T)
end

delimit("Derived dictionaries")
test_derived_dicts(T)


delimit("Test Grids")
@testset "$(rpad("Grids",80))" begin
    test_grids(T) end

delimit("Gram")
@testset "$(rpad("Gram functionality",80))" begin
    discrete_gram_test(T)
    general_gram_test(T)
end

delimit("Check evaluations, interpolations, extensions, setexpansions")

@testset "$(rpad("Fourier expansions",80))" begin
    test_fourier_series(T) end

@testset "$(rpad("Orthogonal polynomials",80))" begin
    test_ops(T) end

# end # for T in...

delimit("Test DCTI")
@testset "$(rpad("evaluation",80))"  begin test_full_transform_extremagrid() end
@testset "$(rpad("inverse",80))" begin test_inverse_transform_extremagrid() end

delimit("Generic OPS")
include("test_generic_OPS.jl")

delimit("Mapped dictionaries")
test_mapped_dicts()

println()
println(" All tests passed!")

end # module
