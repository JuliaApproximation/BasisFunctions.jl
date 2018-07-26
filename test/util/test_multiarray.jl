
if VERSION < v"0.7-"
    using Base.Test, BasisFunctions
else
    using Test
end

delimit("MultiArray")
@testset begin
    A = Vector{Vector{Int}}
    a = fill!(MultiArray{A}([0,4,7]),1)
    b = fill!(MultiArray{A}([0,4,7]),2)
    @test size(a) == (7,)
    @test length(a,1) == 4
    @test length(a,2) == 3

    @test a .+ a ≈ b
    @test broadcast(*, a, a) ≈ a
    @test broadcast(+, 1, a) ≈ b
    @test broadcast(+, a, 1) ≈ b
    @test a + a ≈ b
    @test a[:] == ones(Int,7)
end
