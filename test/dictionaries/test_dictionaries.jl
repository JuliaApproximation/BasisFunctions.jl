
using BasisFunctions, BasisFunctions.Test, DomainSets, StaticArrays, Test
import BasisFunctions.Test: supports_approximation, supports_interpolation, suitable_function, suitable_interpolation_grid
BF = BasisFunctions

# types = [Float64,]

domaintypes = (Float64, BigFloat)

include("test_dictionaries_util.jl")
include("test_dictionaries_derived.jl")
include("test_dictionaries_discrete.jl")
include("test_dictionaries_tensor.jl")
include("test_dictionaries_mapped.jl")

@testset "$(rpad("test custom dictionary",80))" begin
    mydict = BasisFunctions.MyDictionary()
    test_generic_dict_interface(mydict)
end


test_dictionaries = [Fourier,
    ChebyshevT,
    ChebyshevU,
    Legendre,
    Laguerre,
    Hermite,
    Jacobi,
    CosineSeries,
    SineSeries,
    Monomials,
    Rationals]

for T in domaintypes
    delimit("1D dictionaries ($(T))")
    for DICT in test_dictionaries
        @testset "$(rpad(string(DICT),80))" begin
            n = 9
            basis = DICT{T}(n)
            @test length(basis) == n
            @test domaintype(basis) == T
            test_generic_dict_interface(basis)
        end
    end
    # also try a Fourier series with an even length
    @testset begin test_generic_dict_interface(Fourier{T}(8)) end

    delimit("derived dictionaries ($(T))")
    test_derived_dicts(T)

    delimit("Tensor specific tests ($(T))")
    @testset "$(rpad("test iteration",80))" begin
        test_tensor_sets(T)
    end

    delimit("Tensor product set interfaces ($(T))")
    # TODO: all sets in the test below should use type T!
    @testset "$(rpad("$(name(basis))",80," "))" for basis in
                ( Fourier(11) ⊗ Fourier(21), # Two odd-length Fourier series
                  Fourier(10) ⊗ ChebyshevT(12), # combination of Fourier and Chebyshev
                  Fourier(11) ⊗ Fourier(10), # Odd and even-length Fourier series
                  ChebyshevT(11) ⊗ ChebyshevT(20), # Two Chebyshev sets
                  Fourier(11, 2, 3) ⊗ Fourier(11, 4, 5), # Two mapped Fourier series
                  ChebyshevT(9, 2, 3) ⊗ ChebyshevT(7, 4, 5)) # Two mapped Chebyshev series
        test_generic_dict_interface(basis)
    end

    delimit("Discrete sets ($(T))")
    @testset "$(rpad("discrete sets",80))" begin
        test_discrete_sets(T)
    end
end

delimit("Mapped dictionaries")
test_mapped_dicts()
