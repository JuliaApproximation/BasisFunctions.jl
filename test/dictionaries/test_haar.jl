using Test
using BasisFunctions, GridArrays

@testset "Basic Haar tests" begin
    s = zeros(10);s[1]=1
    H = Haar(10)
    @test BasisFunctions.dict_eval(H,0)≈s
    g = interpolation_grid(H)
    @test g≈MidpointEquispacedGrid(10,0,1)
    @test isorthogonal(H,discretemeasure(g))
    @test isorthogonal(H,FourierMeasure())
    @test isorthogonal(H,discretemeasure(MidpointEquispacedGrid(10,0,1)))

    @test support(H) ≈ 0.0..1.0
    @test support(H,1) ≈ 0.0..0.1

    @test 0 ∈ support(H)
    @test 1 ∉ support(H)

    @test 0 ∈ support(H,1)
    @test .1 ∉ support(H,1)

    @test gram(H,discretemeasure(g)) ≈ IdentityOperator(H)
    @test evaluation(H,g)≈IdentityOperator(H)

    @test gram(H,FourierMeasure())≈.1IdentityOperator(H)
end
