using BasisFunctions, Test, DomainSets, GridArrays

@testset "Measure" begin
function generic_test_measure(measure)
    io = IOBuffer()
    show(io, measure)
    @test length(take!(io))>0
    support(measure)
    x = rand()
    @test weight(measure,x) ≈ weightfunction(measure)(x)
end

μ = BasisFunctions.GenericWeightMeasure(UnitInterval(),x->1.)
generic_test_measure(μ)
@test_throws ErrorException !(isprobabilitymeasure(μ))

μ = BasisFunctions.GenericLebesgueMeasure(UnitInterval())
generic_test_measure(μ)

μ = BasisFunctions.LegendreMeasure()
generic_test_measure(μ)
@test !isprobabilitymeasure(μ)

μ = BasisFunctions.FourierMeasure()
generic_test_measure(μ)
@test isprobabilitymeasure(μ)

@test lebesguemeasure(UnitInterval()) isa FourierMeasure
@test lebesguemeasure(ChebyshevInterval()) isa LegendreMeasure
@test lebesguemeasure(0.3..0.4) isa BasisFunctions.GenericLebesgueMeasure

μ = ChebyshevTMeasure()
generic_test_measure(μ)
@test !isprobabilitymeasure(μ)

μ = ChebyshevUMeasure()
generic_test_measure(μ)
@test !isprobabilitymeasure(μ)

μ = JacobiMeasure(rand(),rand())
generic_test_measure(μ)
@test !isprobabilitymeasure(μ)

m = JacobiMeasure(rand(),rand())
μ = restrict(m, UnitInterval())
generic_test_measure(μ)
@test supermeasure(μ) == m
μ = submeasure(m, UnitInterval())
generic_test_measure(μ)
@test supermeasure(μ) == m

m = mapping(FourierGrid(10,-1,1))
μ = mappedmeasure(m,FourierMeasure())
generic_test_measure(μ)

m = FourierMeasure()
μ = productmeasure(m,m)
@test elements(μ) == (element(μ,1),element(μ,2))
io = IOBuffer()
show(io, μ)
@test length(take!(io))>0
support(μ)
x = (rand(),rand())
@test weight(μ,x)≈weightfunction(μ)(x)

end
