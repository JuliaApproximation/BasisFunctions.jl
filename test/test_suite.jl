# test_suite.jl
module test_suite

srand(1234)
using StaticArrays
using Base.Test
using Domains
using BasisFunctions

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
include("test_bsplinetranslatedbasis.jl")
include("test_DCTI.jl")
include("test_gram.jl")

delimit("Wavelets")
include("test_wavelets.jl")
delimit("Compact approximation")
include("test_compact_approximation.jl")

# Verify types of FFT and DCT plans by FFTW
# If anything changes here, the aliases in fouriertransforms.jl have to change as well
d1 = plan_fft!(zeros(Complex{Float64}, 10), 1:1)
@test typeof(d1) == Base.DFT.FFTW.cFFTWPlan{Complex{Float64},-1,true,1}
d2 = plan_fft!(zeros(Complex{Float64}, 10, 10), 1:2)
@test typeof(d2) == Base.DFT.FFTW.cFFTWPlan{Complex{Float64},-1,true,2}
d3 = plan_bfft!(zeros(Complex{Float64}, 10), 1:1)
@test typeof(d3) == Base.DFT.FFTW.cFFTWPlan{Complex{Float64},1,true,1}
d4 = plan_bfft!(zeros(Complex{Float64}, 10, 10), 1:2)
@test typeof(d4) == Base.DFT.FFTW.cFFTWPlan{Complex{Float64},1,true,2}

d5 = plan_dct!(zeros(10), 1:1)
@test typeof(d5) == Base.DFT.FFTW.DCTPlan{Float64,5,true}
d6 = plan_idct!(zeros(10), 1:1)
@test typeof(d6) == Base.DFT.FFTW.DCTPlan{Float64,4,true}

SETS = [FourierBasis, ChebyshevBasis, ChebyshevU, LegendrePolynomials,
        LaguerrePolynomials, HermitePolynomials, CosineSeries, SineSeries,
        BSplineTranslatesBasis, SymBSplineTranslatesBasis]
# SETS = [FourierBasis, ChebyshevBasis, ChebyshevU, LegendrePolynomials,
#         LaguerrePolynomials, HermitePolynomials, CosineSeries, SineSeries]

for T in [Float64,BigFloat,]
    println()
    delimit("T is $T", )

    @testset "$(rpad("$(name(basis))",80," "))" for basis in
                ( FourierBasis(11) ⊗ FourierBasis(21), # Two odd-length Fourier series
                  FourierBasis(10) ⊗ ChebyshevBasis(12), # combination of Fourier and Chebyshev
                  FourierBasis(11) ⊗ FourierBasis(10), # Odd and even-length Fourier series
                  ChebyshevBasis(11) ⊗ ChebyshevBasis(20), # Two Chebyshev sets
                  FourierBasis(11, 2, 3) ⊗ FourierBasis(11, 4, 5), # Two mapped Fourier series
                  ChebyshevBasis(9, 2, 3) ⊗ ChebyshevBasis(7, 4, 5)) # Two mapped Chebyshev series
        test_generic_dict_interface(basis, Span(basis))
    end

    delimit("Tensor specific tests")
    @testset "$(rpad("test iteration",80))" begin
        test_tensor_sets(T) end

    test_derived_dicts(T)
    delimit("Operators")
    test_operators(T)
    test_generic_operators(T)

    delimit("Generic interfaces")

    @testset "$(rpad("$(name(instantiate(SET,n))) with $n dof",80," "))" for SET in SETS,
            n = 9
            basis = instantiate(SET, n, T)

            @test length(basis) == n
            @test domaintype(basis) == T

            test_generic_dict_interface(basis, Span(basis))
    end
    # also try a Fourier series with an even length
    test_generic_dict_interface(FourierBasis{T}(8))

    delimit("Discrete sets")
    @testset "$(rpad("discrete sets",80))" begin
        test_discrete_sets(T)
    end

    delimit("Derived dictionaries")

    test_derived_dicts(T)

    delimit("Tensor product set interfaces")

    # TODO: all sets in the test below should use type T!

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

    @testset "$(rpad("Periodic translate expansions",80))" begin
        test_generic_periodicbsplinebasis(T) end

    @testset "$(rpad("Translates of B spline expansions",80))" begin
        test_translatedbsplines(T)
        test_translatedsymmetricbsplines(T)
        # test_orthonormalsplinebasis(T)
        # test_discrete_orthonormalsplinebasis(T)
        test_dualsplinebasis(T)
        test_discrete_dualsplinebasis(T)
        test_bspline_platform(T)
        test_sparsity_speed(T)
    end

end # for T in...

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
