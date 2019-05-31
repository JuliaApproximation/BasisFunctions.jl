using BasisFunctions, LinearAlgebra, DomainSets, Grids, Test, StaticArrays
@testset begin

    F = Fourier(3)
    S = SynthesisOperator(F)
    G = GridSampling(PeriodicEquispacedGrid(3,-1,1))
    @test Matrix(G*S)≈Matrix(G*F)
    @test_throws MethodError SynthesisOperator(F,nothing)''

end
