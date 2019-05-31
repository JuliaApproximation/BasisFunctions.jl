using BasisFunctions, Test
@testset "Normalization" begin
F = Fourier(10)
f = exp
for g in (PeriodicEquispacedGrid(10,support(F)), interpolation_grid(F), MidpointEquispacedGrid(10,support(F)))
    s = sampling_normalization(GridBasis(g), FourierMeasure())
    @test abs(sum(s*s*f)-applymeasure(FourierMeasure(),f)) < .1
    s = sampling_normalization(GridBasis(g), BasisFunctions.GenericLebesgueMeasure(support(F)))
    @test abs(sum(s*s*f)-applymeasure(BasisFunctions.GenericLebesgueMeasure(support(F)),f)) < .1
end

C = ChebyshevT(9)
f = exp
g = ChebyshevNodes(9)
s = sampling_normalization(GridBasis(g), BasisFunctions.GenericLebesgueMeasure(support(C)))
@test abs(sum(s*s*f)-applymeasure(BasisFunctions.GenericLebesgueMeasure(support(C)),f)) < .1
s = sampling_normalization(GridBasis(g), LegendreMeasure{Float64}())
@test abs(sum(s*s*f)-applymeasure(BasisFunctions.GenericLebesgueMeasure(support(C)),f)) < .1


g = ChebyshevExtremae(9)
s = sampling_normalization(GridBasis(g), BasisFunctions.GenericLebesgueMeasure(support(C)))
@test abs(sum(s*s*f)-applymeasure(BasisFunctions.GenericLebesgueMeasure(support(C)),f)) < .1
s = sampling_normalization(GridBasis(g), LegendreMeasure{Float64}())
@test abs(sum(s*s*f)-applymeasure(BasisFunctions.GenericLebesgueMeasure(support(C)),f)) < .1

C = ChebyshevT(1000)
f = exp
g = MidpointEquispacedGrid(1000,-1,1)
s = sampling_normalization(GridBasis(g), BasisFunctions.ChebyshevTMeasure{Float64}())
@test abs(sum(s*s*f)-applymeasure(BasisFunctions.ChebyshevTMeasure{Float64}(),f)) < .1

x, w = rectangular_rule(10)
μ = gauss_rule(Fourier(10))
@test grid(μ) ≈ x
@test weights(μ) ≈ w
end
