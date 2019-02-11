
chebyshev_types = (Float32, Float64, BigFloat)

function test_chebyshev_expansions(T)
    n = 10
    b = ChebyshevT{T}(n)
    if T == Float32 || T == Float64
        g = BasisFunctions.secondgrid(b)
        BasisFunctions.Test.test_generic_dict_transform(b, g)
        BasisFunctions.Test.test_generic_dict_transform(complex(b), g)
    end

    @test support(b) == ChebyshevInterval{T}()
end

function test_chebyshev_orthogonality()
    B = ChebyshevT(10)
    test_orthogonality_orthonormality(B, true, false, BasisFunctions.ChebyshevTMeasure{Float64}())
    test_orthogonality_orthonormality(B, true, false, OPSNodesMeasure(B))
    test_orthogonality_orthonormality(B, true, false, BasisFunctions.DiscreteMeasure(interpolation_grid(B)))
end


for T in chebyshev_types
    @testset "$(rpad("Chebyshev expansions ($T)",80))" begin
        test_chebyshev_expansions(T)
    end
end

@testset "$(rpad("ChebyshevT orthogonality",80))" begin
    test_chebyshev_orthogonality()
end
