function test_translatedbsplines(T)
  n = 5
  b = BSplineTranslatesBasis(n,3, T)

  @test length(b)==5
  @test degree(b)==3
  @test BasisFunctions.name(b) == "Set of translates of a function"
  @test is_basis(b)
  @test is_biorthogonal(b)
  @test !is_orthogonal(b)
  @test !is_orthonormal(b)
  @test !has_unitary_transform(b)
  @test left(b) == 0
  @test right(b)==1
  @test left(b,1)≈ 0
  @test left(b,2)≈1//5
  @test left(b,3)≈2//5
  @test left(b,4)≈3//5
  @test left(b,5)≈4//5
  @test right(b,1)≈4//5
  @test right(b,2)≈5//5
  @test right(b,3)≈6//5
  @test right(b,4)≈7//5
  @test right(b,5)≈8//5
  @test instantiate(BSplineTranslatesBasis, 4, Float16)==BSplineTranslatesBasis(4,3,Float16)
  @test set_promote_eltype(b, Float16)==BSplineTranslatesBasis(n,degree(b),Float16)
  @test set_promote_eltype(b, complex(Float64))==BSplineTranslatesBasis(n,degree(b),Complex128)
  @test resize(b, 20)==BSplineTranslatesBasis(20,degree(b),T)
  @test BasisFunctions.grid(b)==PeriodicEquispacedGrid(n,0,1)
  @test BasisFunctions.period(b)==T(1)
  @test BasisFunctions.stepsize(b)==T(1//5)
  t = .001
  @test in_support(b,1,.5)
  @test !in_support(b,1,.8+t)
  @test !in_support(b,1,1.-t)
  @test in_support(b,3,.2-t)
  @test in_support(b,3,.4+t)
  @test !in_support(b,3,.2+t)
  @test !in_support(b,3,.4-t)

  grid(BSplineTranslatesBasis(n,2, T)) == MidpointEquispacedGrid(n,0,1)
  @test degree(BSplineTranslatesBasis(5,2, T)) == 2
  @test BasisFunctions.compatible_grid(b, grid(b))
  @test !BasisFunctions.compatible_grid(b, MidpointEquispacedGrid(n,0,1))
  @test !BasisFunctions.compatible_grid(b, PeriodicEquispacedGrid(n+1,0,1))
  @test !BasisFunctions.compatible_grid(b, PeriodicEquispacedGrid(n,0.1,1))
  @test !BasisFunctions.compatible_grid(b, PeriodicEquispacedGrid(n,0,1.1))

  n = 3
  b=BSplineTranslatesBasis(n,1,T)
  @test degree(b) == 1
  # gramcolu
  @test BasisFunctions.grammatrix(b) ≈ [2//3 1//6 1//6; 1//6 2//3 1//6;1//6 1//6 2//3]//n
  @test BasisFunctions.dualgrammatrix(b) ≈ [5/3 -1/3 -1/3; -1/3 5/3 -1/3; -1/3 -1/3 5/3]*n
  # @test matrix(BasisFunctions.Gram(b)) ≈ BasisFunctions.grammatrix(b)
  # @test sum(abs(BasisFunctions.dualgrammatrix(b)-matrix(BasisFunctions.DualGram(b)))) < 1e-5
  # @test matrix(BasisFunctions.Gram(b))*matrix(BasisFunctions.DualGram(b)) ≈ eye(n)
  # @test matrix(BasisFunctions.Gram(b))*matrix(BasisFunctions.DualGram(b)) ≈ matrix(BasisFunctions.MixedGram(b))





  n = 8
  b=BSplineTranslatesBasis(n,0,T)
  @test degree(b) == 0
  t = linspace(0,1)
  fp = map(x->BasisFunctions.eval_element(b,1,x),t)
  fd = 1/n*map(x->BasisFunctions.eval_dualelement(b,1,x),t)
  @test norm(fp-fd) < sqrt(eps(T))


  @test BasisFunctions.compatible_grid(b, grid(b))
  @test !BasisFunctions.compatible_grid(b, PeriodicEquispacedGrid(n,0,1))
  @test !BasisFunctions.compatible_grid(b, MidpointEquispacedGrid(n+1,0,1))
  @test !BasisFunctions.compatible_grid(b, MidpointEquispacedGrid(n,0.1,1))
  @test !BasisFunctions.compatible_grid(b, MidpointEquispacedGrid(n,0,1.1))

end


# using Base.Test
# using BasisFunctions
# @testset begin test_translatedbsplines(BigFloat) end


# using Plots
# using BasisFunctions
# n = 7
# k = 0
# b = BSplineTranslatesBasis(n,k)
# t = linspace(-0,2,200)
# f = map(x->eval_element(b,1,x), t)
# using Plots
# plot!(t,f)
# @which eval_element(b,1,.25)
# BasisFunctions.fun(b)
# f = map(BasisFunctions.fun(b), t)
# plot!(t,f)
# f = map(x->BasisFunctions.eval_expansion(b,ones(n),x),t)
# # f = map(x->BasisFunctions.eval_expansion(b,[1,0,0,0,0,0,0,0,0,0],x),t)
# plot!(t,f,ylims=[-n,2n])
