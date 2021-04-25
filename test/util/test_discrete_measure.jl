
using Test
using DomainIntegrals, DomainSets, BasisFunctions, GridArrays, FillArrays

using BasisFunctions.Test: generic_test_discrete_measure

@testset "Discrete measure" begin
    g = ScatteredGrid(rand(2))
    μ = discretemeasure(g)
    generic_test_discrete_measure(μ)
    @test weights(μ) isa FillArrays.Ones
    @test !isnormalized(μ)


    g2 = g×g
    μ = discretemeasure(g2)
    generic_test_discrete_measure(μ)
    @test Matrix(weights(μ))≈ones(2,2)
    @test !isnormalized(μ)


    μ = DiracWeight(rand())
    generic_test_discrete_measure(μ)
    @test weights(μ) isa FillArrays.Ones
    @test isnormalized(μ)

    μ = discretemeasure(PeriodicEquispacedGrid(3,0,1))
    @test isuniform(μ)
    generic_test_discrete_measure(μ)
    μ = discretemeasure(MidpointEquispacedGrid(3,0,1))
    @test isuniform(μ)
    generic_test_discrete_measure(μ)
    μ = discretemeasure(EquispacedGrid(3,0,1))
    @test isuniform(μ)
    generic_test_discrete_measure(μ)
    μ = BasisFunctions.NormalizedDiracComb(EquispacedGrid(3,0,1))
    @test isnormalized(μ)
    generic_test_discrete_measure(μ)
    μ = discretemeasure(MidpointEquispacedGrid(3,0,1),rand(3))
    generic_test_discrete_measure(μ)

    g = FourierGrid(10)
    mg = rescale(FourierGrid(10),-1, 1)
    m = forward_map(mg)
    μ = discretemeasure(mg)
    generic_test_discrete_measure(μ)
    @test forward_map(μ) == m
    @test supermeasure(μ) == discretemeasure(g)
    @test BasisFunctions.apply_map(discretemeasure(g),m) ≈ μ
    @test BasisFunctions.apply_map(μ,inv(m))≈discretemeasure(g)


    g = PeriodicEquispacedGrid(3,-1,1)×PeriodicEquispacedGrid(4,-1,1)
    μ = discretemeasure(g)
    generic_test_discrete_measure(μ)
    @test μ isa BasisFunctions.DiscreteProductWeight
    @test components(μ) == (component(μ,1),component(μ,2))
end
