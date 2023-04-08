
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

    # restrict to BlasFloat for now, see GenericLinearAlgebra issue #98
    # (https://github.com/JuliaLinearAlgebra/GenericLinearAlgebra.jl/issues/98)
    if T isa LinearAlgebra.BlasFloat
        b_orth = orthogonalize(b, LegendreWeight{T}())
        @test innerproduct(b_orth[1], b_orth[1], LegendreWeight{T}()) ≈ 1
        @test innerproduct(b_orth[1], b_orth[2], LegendreWeight{T}())+1 ≈ 1
    end
end

function test_chebyshev_orthogonality()
    B = ChebyshevT(10)
    test_orthogonality_orthonormality(B, true, false, ChebyshevTWeight{Float64}())
    test_orthogonality_orthonormality(B, true, false, gauss_rule(B))
    test_orthogonality_orthonormality(B, true, false, discretemeasure(interpolation_grid(B)))
    test_orthogonality_orthonormality(B, true, false, discretemeasure(interpolation_grid(resize(B,length(B)+1))))
end


for T in chebyshev_types
    @testset "Chebyshev expansions ($T)" begin
        test_chebyshev_expansions(T)
    end
end

@testset "ChebyshevT orthogonality" begin
    test_chebyshev_orthogonality()
end
