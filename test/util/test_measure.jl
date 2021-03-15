using Test

using DomainSets, DomainIntegrals, GridArrays, StaticArrays

using BasisFunctions
using BasisFunctions.Test: generic_test_measure

@testset "Measure" begin
    μ = BasisFunctions.GenericWeight(UnitInterval(),x->1.)
    generic_test_measure(μ)
    # @test_throws ErrorException !(isnormalized(μ))

    μ = LebesgueDomain(UnitInterval())
    generic_test_measure(μ)

    μ = LegendreWeight()
    generic_test_measure(μ)
    @test !isnormalized(μ)

    μ = FourierWeight()
    generic_test_measure(μ)
    @test isnormalized(μ)

    @test lebesguemeasure(UnitInterval()) isa FourierWeight
    @test lebesguemeasure(ChebyshevInterval()) isa LegendreWeight
    @test lebesguemeasure(0.3..0.4) isa LebesgueDomain

    μ = ChebyshevTWeight()
    generic_test_measure(μ)
    @test !isnormalized(μ)

    μ = ChebyshevUWeight()
    generic_test_measure(μ)
    @test !isnormalized(μ)

    μ = JacobiWeight(rand(),rand())
    generic_test_measure(μ)
    @test !isnormalized(μ)

    m = mapping(FourierGrid(10,-1,1))
    μ = mappedmeasure(m,FourierWeight())
    generic_test_measure(μ)

    m = FourierWeight()
    μ = productmeasure(m,m)
    @test elements(μ) == (element(μ,1),element(μ,2))
    io = IOBuffer()
    show(io, μ)
    @test length(take!(io))>0
    support(μ)
    x = SVector(rand(),rand())
    @test weight(μ,x)≈weightfunction(μ)(x)
end
