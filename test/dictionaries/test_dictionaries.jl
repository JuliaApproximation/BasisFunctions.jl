
using BasisFunctions, BasisFunctions.Test, Domains, StaticArrays
import BasisFunctions.Test: supports_approximation, supports_interpolation, suitable_function, suitable_interpolation_grid
BF = BasisFunctions

if VERSION < v"0.7-"
    using Base.Test
    # types = [Float64, BigFloat,]
else
    using Test
    # types = [Float64,]
end
types = [Float64, BigFloat,]

include("test_dictionaries_util.jl")
include("test_dictionaries_derived.jl")
include("test_dictionaries_discrete.jl")
include("test_dictionaries_tensor.jl")
include("test_dictionaries_mapped.jl")


oned_dictionaries = [FourierBasis, ChebyshevBasis, ChebyshevU, LegendrePolynomials,
        LaguerrePolynomials, HermitePolynomials, CosineSeries, SineSeries,]

for T in types
    delimit("1D dictionaries ($(T))")
    for DICT in oned_dictionaries
        @testset "$(rpad(string(DICT),80))" begin
            n = 9
            basis = instantiate(DICT, n, T)
            @test length(basis) == n
            @test domaintype(basis) == T
            test_generic_dict_interface(basis)
        end
    end
    if VERSION < v"0.7-"
        # also try a Fourier series with an even length
        test_generic_dict_interface(FourierBasis{T}(8))
    end

    delimit("derived dictionaries ($(T))")
    test_derived_dicts(T)

    delimit("Tensor specific tests ($(T))")
    @testset "$(rpad("test iteration",80))" begin
        test_tensor_sets(T)
    end

    delimit("Tensor product set interfaces ($(T))")
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

    delimit("Discrete sets ($(T))")
    @testset "$(rpad("discrete sets",80))" begin
        test_discrete_sets(T)
    end
end

delimit("Mapped dictionaries")
test_mapped_dicts()
