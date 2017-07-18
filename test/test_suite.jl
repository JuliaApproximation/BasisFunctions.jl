# test_suite.jl
module test_suite

using Base.Test
srand(1234)
using StaticArrays
using Domains
using BasisFunctions

BF = BasisFunctions

const show_timings = false
######### #
# Testing
##########

include("util_functions.jl")
include("test_generic_grids.jl")
include("test_generic_sets.jl")
include("test_derived_set.jl")
include("test_bsplines.jl")
include("test_generic_operators.jl")
include("test_ops.jl")
include("test_fourier.jl")
include("test_chebyshev.jl")
include("test_bsplinetranslatedbasis.jl")
include("test_DCTI.jl")
include("test_gram.jl")


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

# SETS = [FourierBasis, ChebyshevBasis, ChebyshevII, LegendreBasis,
#         LaguerreBasis, HermiteBasis, PeriodicSplineBasis, CosineSeries, SineSeries,
#         BSplineTranslatesBasis, SymBSplineTranslatesBasis, OrthonormalSplineBasis,
#         DiscreteOrthonormalSplineBasis]
SETS = [FourierBasis, ChebyshevBasis, ChebyshevII, LegendreBasis,
        LaguerreBasis, HermiteBasis, CosineSeries, SineSeries]

for T in [Float64,BigFloat]
    println()
    delimit("T is $T", )
    delimit("Operators")

    test_generic_operators(T)

    @testset "$(rpad("test diagonal operators",80))" begin
        test_diagonal_operators(T) end

    @testset "$(rpad("test multidiagonal operators",80))" begin
        test_multidiagonal_operators(T) end

    @testset "$(rpad("test invertible operators",80))" begin
        test_invertible_operators(T) end

    @testset "$(rpad("test noninvertible operators",80))" begin
        test_noninvertible_operators(T) end

    @testset "$(rpad("test tensor operators",80))" begin
        test_tensor_operators(T)
    end

    @testset "$(rpad("test circulant operator",80))" begin
        test_circulant_operator(T)
    end

    delimit("Generic interfaces")

    @testset "$(rpad("$(name(instantiate(SET,n))) with $n dof",80," "))" for SET in SETS, n in (8,11)
        # Choose an odd and even number of degrees of freedom
            basis = instantiate(SET, n, T)

            @test length(basis) == n
            @test domaintype(basis) == T

            test_generic_set_interface(basis, span(basis))
    end

    delimit("Tensor product set interfaces")

    # TODO: all sets in the test below should use type T!
    @testset "$(rpad("$(name(basis))",80," "))" for basis in (FourierBasis(10) ⊗ ChebyshevBasis(12),
                  FourierBasis(11) ⊗ FourierBasis(21), # Two odd-length Fourier series
                  FourierBasis(11) ⊗ FourierBasis(10), # Odd and even-length Fourier series
                  ChebyshevBasis(11) ⊗ ChebyshevBasis(20),
                  FourierBasis(11, 2, 3) ⊗ FourierBasis(11, 4, 5), # Two mapped Fourier series
                  ChebyshevBasis(9, 2, 3) ⊗ ChebyshevBasis(7, 4, 5))
        test_generic_set_interface(basis, span(basis))
    end

    delimit("Derived sets")
        test_derived_sets(T)

    delimit("Tensor specific tests")
    @testset "$(rpad("test iteration",80))" begin
        test_tensor_sets(T) end

    delimit("Test Grids")
    @testset "$(rpad("Grids",80))" begin
        test_grids(T) end
    delimit("Test B splines")
    @testset "$(rpad("Elementary properties",80))" begin
      elementarypropsofsplinetest(T)
    end
    @testset "$(rpad("periodic B splines",80))"  begin
      periodicbsplinetest(T)
    end
    @testset "$(rpad("symmetric B splines",80))"  begin
      symmetricbsplinestest(T)
    end
    @testset "$(rpad("integration of B splines",80))"  begin
      test_spline_integration()
    end

    delimit("Gram")
    @testset "$(rpad("Gram functionality",80))" begin
      discrete_gram_test(T)
    end

    delimit("Check evaluations, interpolations, extensions, setexpansions")

    @testset "$(rpad("Fourier expansions",80))" begin
        test_fourier_series(T) end

    @testset "$(rpad("Chebyshev expansions",80))" begin
        test_chebyshev(T) end

    @testset "$(rpad("Orthogonal polynomial evaluation",80))" begin
        test_ops(T) end

    @testset "$(rpad("Periodic translate expansions",80))"begin
        test_generic_periodicbsplinebasis(T) end

    @testset "$(rpad("Translates of B spline expansions",80))"begin
        test_translatedbsplines(T)
        test_translatedsymmetricbsplines(T)
        test_orthonormalsplinebasis(T)
        test_discrete_orthonormalsplinebasis(T)
        test_dualsplinebasis(T)
        test_discrete_dualsplinebasis(T)
      end



end # for T in...

delimit("Test DCTI")
@testset "$(rpad("evaluation",80))"  begin test_full_transform_extremagrid() end
@testset "$(rpad("inverse",80))" begin test_inverse_transform_extremagrid() end

println()
println(" All tests passed!")

end # module
