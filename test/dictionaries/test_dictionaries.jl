
using BasisFunctions, BasisFunctions.Test, DomainSets, StaticArrays,
    DoubleFloats, GaussQuadrature, SpecialFunctions

using Test

import BasisFunctions.Test:
    supports_approximation,
    supports_interpolation,
    suitable_function,
    suitable_interpolation_grid,
    dimension_tuple

BF = BasisFunctions

domaintypes = (Float64, LargeFloat)

GaussQuadrature.maxiterations[Double64] = 50
SpecialFunctions.logabsgamma(x::Double64) = Double64.(logabsgamma(BigFloat(x)))
Base.rationalize(x::SVector{N,Double64}) where {N} = SVector{N,Rational{Int}}([rationalize(x_i) for x_i in x])

include("test_dictionaries_util.jl")
include("test_dictionaries_derived.jl")
include("test_dictionaries_discrete.jl")
include("test_dictionaries_tensor.jl")
include("test_dictionaries_mapped.jl")

@testset "test custom dictionary" begin
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
    PeriodicSincFunctions,
    TrigSeries,
    Monomials,
    RationalFunctions]

for T in domaintypes
    delimit("1D dictionaries ($(T))")
    for DICT in test_dictionaries
        n = 9
        if DICT isa PeriodicSincFunctions
            n = 18
        end
        basis = DICT{T}(n)
        @testset "$(string(basis))" begin
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
    @testset "test iteration" begin
        test_tensor_sets(T)
    end

    delimit("Tensor product set interfaces ($(T))")
    @testset "$(repr(basis))" for basis in
                ( Fourier{T}(5) ⊗ Fourier{T}(7), # Two odd-length Fourier series
                  Fourier{T}(6) ⊗ ChebyshevT{T}(4), # combination of Fourier and Chebyshev
                  Fourier{T}(5) ⊗ Fourier{T}(6), # Odd and even-length Fourier series
                  ChebyshevT{T}(5) ⊗ ChebyshevT{T}(6), # Two Chebyshev sets
                  (Fourier{T}(5) → 2..3) ⊗ (Fourier{T}(7) → 4..5), # Two mapped Fourier series
                  (ChebyshevT{T}(5) → 2..3) ⊗ (ChebyshevT{T}(6) → 4..5), # Two mapped Chebyshev series
                  Fourier{T}(2)^2 ∘ AffineMap(rand(T,2,2),rand(T,2))) # A 2d-mapped dict
        test_generic_dict_interface(basis)
    end

    delimit("Discrete sets ($(T))")
    @testset "discrete sets" begin
        test_discrete_sets(T)
    end
end

delimit("Mapped dictionaries")
test_mapped_dicts()
