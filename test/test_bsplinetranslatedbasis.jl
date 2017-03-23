function test_translatedbsplines(T)
  tol = sqrt(eps(real(T)))
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
  @test abs(sum(BasisFunctions.grammatrix(b) - [2//3 1//6 1//6; 1//6 2//3 1//6;1//6 1//6 2//3]//n)) < tol
  @test abs(sum(BasisFunctions.dualgrammatrix(b) - [5/3 -1/3 -1/3; -1/3 5/3 -1/3; -1/3 -1/3 5/3]*n)) < tol
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
  @test norm(fp-fd) < tol


  @test BasisFunctions.compatible_grid(b, grid(b))
  @test !BasisFunctions.compatible_grid(b, PeriodicEquispacedGrid(n,0,1))
  @test !BasisFunctions.compatible_grid(b, MidpointEquispacedGrid(n+1,0,1))
  @test !BasisFunctions.compatible_grid(b, MidpointEquispacedGrid(n,0.1,1))
  @test !BasisFunctions.compatible_grid(b, MidpointEquispacedGrid(n,0,1.1))

  # Test extension_operator and invertability of restriction_operator w.r.t. extension_operator.
  n = 8
  for degree in 0:3
    b = BSplineTranslatesBasis(n, degree, T)
    basis_ext = extend(b)
    r = restriction_operator(basis_ext, b)
    e = extension_operator(b, basis_ext)
    @test abs(sum(eye(n)-matrix(r*e))) < tol

    grid_ext = grid(basis_ext)
    L = evaluation_operator(b, grid_ext)
    e = random_expansion(b)
    z = L*e
    L2 = evaluation_operator(basis_ext, grid_ext) * extension_operator(b, basis_ext)
    z2 = L2*e
    @test maximum(abs.(z-z2)) < tol
  end



  for K in 0:3
    for s2 in 5:6
      s1 = s2<<1
      b1 = BSplineTranslatesBasis(s1,K)
      b2 = BSplineTranslatesBasis(s2,K)

      e1 = random_expansion(b1)
      e2 = random_expansion(b2)

      @test coefficients(BasisFunctions.default_evaluation_operator(b2, gridspace(b2))*e2) ≈ coefficients(grid_evaluation_operator(b2, gridspace(b2), grid(b2))*e2)
      @test coefficients(BasisFunctions.default_evaluation_operator(b2, gridspace(b1))*e2) ≈ coefficients(grid_evaluation_operator(b2, gridspace(b1), grid(b1))*e2)
      @test coefficients(BasisFunctions.default_evaluation_operator(b1, gridspace(b1))*e1) ≈ coefficients(grid_evaluation_operator(b1, gridspace(b1), grid(b1))*e1)
      @test coefficients(BasisFunctions.default_evaluation_operator(b1, gridspace(b2))*e1) ≈ coefficients(grid_evaluation_operator(b1, gridspace(b2), grid(b2))*e1)
    end
  end

  @test_throws InexactError restriction_operator(BSplineTranslatesBasis(4,0),BSplineTranslatesBasis(2,1))
  @test_throws InexactError extension_operator(BSplineTranslatesBasis(2,0),BSplineTranslatesBasis(4,1))
  @test_throws AssertionError restriction_operator(BSplineTranslatesBasis(4,0), BSplineTranslatesBasis(3,0))
  @test_throws AssertionError extension_operator(BSplineTranslatesBasis(4,0), BSplineTranslatesBasis(6,0))
end

# exit()
# using Base.Test
# using BasisFunctions
# @testset begin test_translatedbsplines(Float64) end

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
#
