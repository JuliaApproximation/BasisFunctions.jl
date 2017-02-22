function test_periodicbsplines(T)
  b = PeriodicBSplineBasis(5,3, T)

  @test degree(b) == 3
  @test length(b)==5
  @test BasisFunctions.name(b) == "Periodic squeezed B-splines of degree 3"
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
  @test instantiate(PeriodicBSplineBasis, 4, Float16)==PeriodicBSplineBasis(4,3,Float16)
  @test set_promote_eltype(b, Float16)==PeriodicBSplineBasis(5,3,Float16)
  @test set_promote_eltype(b, complex(Float64))==PeriodicBSplineBasis(5,3,Complex128)
  @test resize(b, 20)==PeriodicBSplineBasis(20,3,T)
  @test grid(b)==PeriodicEquispacedGrid(5,0,1)
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

  n = 3
  b=PeriodicBSplineBasis(n,1,T)
  # gramcolu
  @test BasisFunctions.grammatrix(b) ≈ [2//3 1//6 1//6; 1//6 2//3 1//6;1//6 1//6 2//3]*BasisFunctions.splinescaling(n,Float64)^2/n
  @test BasisFunctions.dualgrammatrix(b) ≈ [5/3 -1/3 -1/3; -1/3 5/3 -1/3; -1/3 -1/3 5/3]*n/BasisFunctions.splinescaling(n,Float64)^2
  @test matrix(BasisFunctions.Gram(b)) ≈ BasisFunctions.grammatrix(b)
  @test sum(abs(BasisFunctions.dualgrammatrix(b)-matrix(BasisFunctions.DualGram(b)))) < 1e-5
  @test matrix(BasisFunctions.Gram(b))*matrix(BasisFunctions.DualGram(b)) ≈ eye(n)
  @test matrix(BasisFunctions.Gram(b))*matrix(BasisFunctions.DualGram(b)) ≈ matrix(BasisFunctions.MixedGram(b))

  n = 8
  b=PeriodicBSplineBasis(n,0,T)
  t = linspace(0,1)
  fp = map(x->BasisFunctions.eval_element(b,1,x),t)
  fd = map(x->BasisFunctions.eval_dualelement(b,1,x),t)
  @test norm(fp-fd) < sqrt(eps(T))

end

# using Plots
# n = 20
# b=PeriodicBSplineBasis(n,0)
# t = linspace(0,1,1000)
# f = map(x->BasisFunctions.eval_element(b,1,x),t)
#   plot(t,f)
#
#
# f = map(x->BasisFunctions.eval_dualelement(b,1,x),t)
# # f = map(x->BasisFunctions.eval_expansion(b,[1,0,0,0,0],x),t)
#   plot!(t,f,ylims=[-n,2n],linestyle=:dot)
#
# @which eval_expansion(b,ones(n),t)
#
#
# f = map(x->BasisFunctions.eval_expansion(b,ones(n),x),t)
# # f = map(x->BasisFunctions.eval_expansion(b,[1,0,0,0,0,0,0,0,0,0],x),t)
#   plot!(t,f,ylims=[-n,2n])
