using BasisFunctions, LinearAlgebra, DomainSets, GridArrays, Test, StaticArrays
@testset begin

    F = Fourier(3) ⇒ -1..1
    S = SynthesisOperator(F)
    G = GridSampling(PeriodicEquispacedGrid(3,-1,1))
    @test Matrix(G*S)≈Matrix(G*F)
    @test_throws MethodError SynthesisOperator(F,nothing)''

    g = interpolation_grid(Fourier(3) ⇒ -1..1)
    g1 = gram(Fourier(3) ⇒ -1..1,discretemeasure(mapped_grid(FourierGrid(6),mapping(g))))
    g2 = gram(Fourier(3),discretemeasure(FourierGrid(6)))
    g3 = BasisFunctions.default_gram(Fourier(3),discretemeasure(FourierGrid(6)))

    @test g1≈g2≈g3


    g1 = gram(Fourier(3) ⇒ -1..1)
    g2 = gram(Fourier(3))
    g3 = BasisFunctions.default_gram(Fourier(3))

    @test g1≈g2≈g3

    @test weights(discretemeasure(subgrid(PeriodicEquispacedGrid(10,0,1)^2,(0.0..0.5)^2))) isa OuterProductArray

end
