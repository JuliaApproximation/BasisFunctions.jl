using Test
using BasisFunctions, GridArrays

@testset "Basic PiecewiseConstants tests" begin
    s = zeros(10);s[1]=1
    H = PiecewiseConstants(10)
    @test BasisFunctions.dict_eval(H,0)≈s
    g = interpolation_grid(H)
    @test g≈MidpointEquispacedGrid(10,0,1)
    @test isorthogonal(H,discretemeasure(g))
    @test isorthogonal(H,FourierWeight())
    @test isorthogonal(H,discretemeasure(MidpointEquispacedGrid(10,0,1)))

    @test support(H) == Interval{:closed,:open}(0,1)
    @test support(H,1) == Interval{:closed,:open}(0,0.1)

    @test 0 ∈ support(H)
    @test 1 ∉ support(H)

    @test 0 ∈ support(H,1)
    @test .1 ∉ support(H,1)

    @test gram(H,discretemeasure(g)) ≈ IdentityOperator(H)
    @test evaluation(H,g)≈IdentityOperator(H)

    @test gram(H,FourierWeight())≈.1IdentityOperator(H)
end
