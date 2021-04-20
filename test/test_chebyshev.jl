
chebyshev_types = (Float32, Float64, LargeFloat)

function test_chebyshev_expansions(T)
    n = 10
    b = ChebyshevT{T}(n)
    if T == Float32 || T == Float64
        g = BasisFunctions.secondgrid(b)
        BasisFunctions.Test.test_generic_dict_transform(b, g)
        BasisFunctions.Test.test_generic_dict_transform(complex(b), g)
        @test convert(ChebyshevT{BigFloat}, ChebyshevT{T}(10)) isa ChebyshevT{BigFloat}
    end

    @test support(b) == ChebyshevInterval{T}()
end

function test_chebyshev_orthogonality()
    B = ChebyshevT(10)
    test_orthogonality_orthonormality(B, true, false, ChebyshevTWeight{Float64}())
    test_orthogonality_orthonormality(B, true, false, gauss_rule(B))
    test_orthogonality_orthonormality(B, true, false, discretemeasure(interpolation_grid(B)))
    test_orthogonality_orthonormality(B, true, false, discretemeasure(interpolation_grid(resize(B,length(B)+1))))
end


for T in chebyshev_types
    @testset "$(rpad("Chebyshev expansions ($T)",80))" begin
        test_chebyshev_expansions(T)
    end
end

@testset "$(rpad("ChebyshevT orthogonality",80))" begin
    test_chebyshev_orthogonality()
end
