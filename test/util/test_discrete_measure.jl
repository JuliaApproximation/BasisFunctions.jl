
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


    μ = DiracMeasure(rand())
    generic_test_discrete_measure(μ)
    @test weights(μ) isa FillArrays.Ones
    @test isnormalized(μ)

    μ = discretemeasure(PeriodicEquispacedGrid(3,0,1))
    @test  μ isa DiracCombMeasure && μ isa UniformDiracCombMeasure
    generic_test_discrete_measure(μ)
    μ = discretemeasure(MidpointEquispacedGrid(3,0,1))
    @test μ isa DiracCombMeasure
    generic_test_discrete_measure(μ)
    μ = discretemeasure(EquispacedGrid(3,0,1))
    @test μ isa DiracCombMeasure
    generic_test_discrete_measure(μ)
    μ = DiracCombProbabilityMeasure(EquispacedGrid(3,0,1))
    @test μ isa DiracCombProbabilityMeasure
    @test isnormalized(μ)
    generic_test_discrete_measure(μ)
    μ = WeightedDiracCombMeasure(MidpointEquispacedGrid(3,0,1),rand(3))
    generic_test_discrete_measure(μ)

    g = FourierGrid(10)
    mg = FourierGrid(10,-1,1)
    m = mapping(mg)
    μ = discretemeasure(mg)
    @test μ isa BasisFunctions.DiscreteMappedMeasure
    generic_test_discrete_measure(μ)
    @test mapping(μ) == m
    @test supermeasure(μ) == discretemeasure(g)
    @test BasisFunctions.apply_map(discretemeasure(g),m) ≈ μ
    @test BasisFunctions.apply_map(μ,inv(m))≈discretemeasure(g)


    g = PeriodicEquispacedGrid(3,-1,1)×PeriodicEquispacedGrid(4,-1,1)
    μ = discretemeasure(g)
    generic_test_discrete_measure(μ)
    @test μ isa BasisFunctions.DiscreteProductMeasure
    @test elements(μ) == (element(μ,1),element(μ,2))
end
