using BasisFunctions, LinearAlgebra, DomainSets, Grids, Test, StaticArrays
@testset begin

    F = Fourier(3)
    S = SynthesisOperator(F)
    G = GridSampling(PeriodicEquispacedGrid(3,-1,1))
    @test Matrix(G*S)≈Matrix(G*F)
    @test_throws MethodError SynthesisOperator(F,nothing)''

    g = interpolation_grid(Fourier(3,-1,1))
    g1 = gramoperator(Fourier(3,-1,1),discretemeasure(mapped_grid(FourierGrid(6),mapping(g))))
    g2 = gramoperator(Fourier(3),discretemeasure(FourierGrid(6)))
    g3 = BasisFunctions.default_gramoperator(Fourier(3),discretemeasure(FourierGrid(6)))

    @test g1≈g2≈g3


    g1 = gramoperator(Fourier(3,-1,1))
    g2 = gramoperator(Fourier(3))
    g3 = BasisFunctions.default_gramoperator(Fourier(3))

    @test g1≈g2≈g3
    
end
