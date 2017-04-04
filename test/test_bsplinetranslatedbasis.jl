function test_generic_periodicbsplinebasis(T)

  for B in (BSplineTranslatesBasis, SymBSplineTranslatesBasis,)
    tol = sqrt(eps(real(T)))
    n = 5
    b = B(n,3, T)
    @test left(b) == 0
    @test right(b)==1

    @test length(b)==5
    @test degree(b)==3
    @test is_basis(b)
    @test is_biorthogonal(b)
    @test !is_orthogonal(b)
    @test !is_orthonormal(b)
    @test !has_unitary_transform(b)

    @test BasisFunctions.left_of_compact_function(b) <= 0 <= BasisFunctions.right_of_compact_function(b)
    @test 0 < BasisFunctions.right_of_compact_function(b) - BasisFunctions.left_of_compact_function(b) < BasisFunctions.period(b)

    @test instantiate(B, 4, Float16)==B(4,3,Float16)
    @test set_promote_eltype(b, Float16)==B(n,degree(b),Float16)
    @test set_promote_eltype(b, complex(Float64))==B(n,degree(b),Complex128)
    @test resize(b, 20)==B(20,degree(b),T)
    @test BasisFunctions.grid(b)==PeriodicEquispacedGrid(n,0,1)
    @test BasisFunctions.period(b)==T(1)
    @test BasisFunctions.stepsize(b)==T(1//5)

    n = 3
    b=B(n,1,T)
    @test abs(sum(BasisFunctions.grammatrix(b) - [2//3 1//6 1//6; 1//6 2//3 1//6;1//6 1//6 2//3]//n)) < tol
    @test abs(sum(BasisFunctions.dualgrammatrix(b) - [5/3 -1/3 -1/3; -1/3 5/3 -1/3; -1/3 -1/3 5/3]*n)) < tol

    n = 8
    b=B(n,0,T)
    t = linspace(T(0),T(1))
    fp = map(x->BasisFunctions.eval_element(b,1,x),t)
    fd = 1/n*map(x->BasisFunctions.eval_dualelement(b,1,x),t)
    @test fp≈fd
  end
end

function test_translatedbsplines(T)
  tol = sqrt(eps(real(T)))
  n = 5
  b = BSplineTranslatesBasis(n,3, T)
  @test BasisFunctions.name(b) == "Set of translates of a function (B spline of degree 3)"

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

  t = .001
  @test in_support(b,1,.5)
  @test !in_support(b,1,.8+t)
  @test !in_support(b,1,1.-t)
  @test in_support(b,3,.2-t)
  @test in_support(b,3,.4+t)
  @test !in_support(b,3,.2+t)
  @test !in_support(b,3,.4-t)

  @test BasisFunctions.compatible_grid(b, grid(b))
  @test !BasisFunctions.compatible_grid(b, MidpointEquispacedGrid(n,0,1))
  @test !BasisFunctions.compatible_grid(b, PeriodicEquispacedGrid(n+1,0,1))
  @test !BasisFunctions.compatible_grid(b, PeriodicEquispacedGrid(n,0.1,1))
  @test !BasisFunctions.compatible_grid(b, PeriodicEquispacedGrid(n,0,1.1))

  grid(BSplineTranslatesBasis(n,2, T)) == MidpointEquispacedGrid(n,0,1)
  @test degree(BSplineTranslatesBasis(5,2, T)) == 2
  b = BSplineTranslatesBasis(n,2,T)
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

      mr = matrix(restriction_operator(b1, b2))
      me = matrix(extension_operator(b2, b1))
      pinvme = pinv(me)
      r = rand(size(pinvme,2))
      @test pinvme*r ≈ mr*r
    end
  end

  @test_throws AssertionError restriction_operator(BSplineTranslatesBasis(4,0), BSplineTranslatesBasis(3,0))
  @test_throws AssertionError extension_operator(BSplineTranslatesBasis(4,0), BSplineTranslatesBasis(6,0))
end


function test_translatedsymmetricbsplines(T)
  tol = sqrt(eps(real(T)))
  n = 5
  b = SymBSplineTranslatesBasis(n,3, T)
  @test BasisFunctions.name(b) == "Set of translates of a function (symmetric B spline of degree 3)"


  @test left(b,1)≈ -4//10
  @test left(b,2)≈ -2//10
  @test left(b,3)≈ 0//10
  @test left(b,4)≈ 2//10
  @test left(b,5)≈ 4//10
  @test right(b,1)≈4//10
  @test right(b,2)≈6//10
  @test right(b,3)≈8//10
  @test right(b,4)≈10//10
  @test right(b,5)≈12//10

  t = .001
  @test in_support(b,1,.0)
  @test !in_support(b,1,.4+t)
  @test !in_support(b,1,.6-t)
  @test in_support(b,3,.8-t)
  @test in_support(b,3,.0+t)
  @test !in_support(b,3,.8+t)
  @test !in_support(b,3,.0-t)

  grid(SymBSplineTranslatesBasis(n,2, T)) == EquispacedGrid(n,0,1)
  @test BasisFunctions.compatible_grid(b, grid(b))
  @test BasisFunctions.compatible_grid(b, MidpointEquispacedGrid(n,0,1))
  @test !BasisFunctions.compatible_grid(b, PeriodicEquispacedGrid(n+1,0,1))
  @test !BasisFunctions.compatible_grid(b, PeriodicEquispacedGrid(n,0.1,1))
  @test !BasisFunctions.compatible_grid(b, PeriodicEquispacedGrid(n,0,1.1))

  @test BasisFunctions.compatible_grid(b, grid(b))
  @test BasisFunctions.compatible_grid(b, PeriodicEquispacedGrid(n,0,1))
  @test !BasisFunctions.compatible_grid(b, MidpointEquispacedGrid(n+1,0,1))
  @test !BasisFunctions.compatible_grid(b, MidpointEquispacedGrid(n,0.1,1))
  @test !BasisFunctions.compatible_grid(b, MidpointEquispacedGrid(n,0,1.1))

  # Test extension_operator and invertability of restriction_operator w.r.t. extension_operator.
  n = 8
  for degree in 1:2:3
    b = SymBSplineTranslatesBasis(n, degree, T)
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



  for K in 1:2:3
    for s2 in 5:6
      s1 = s2<<1
      b1 = SymBSplineTranslatesBasis(s1,K)
      b2 = SymBSplineTranslatesBasis(s2,K)

      e1 = random_expansion(b1)
      e2 = random_expansion(b2)

      @test coefficients(BasisFunctions.default_evaluation_operator(b2, gridspace(b2))*e2) ≈ coefficients(grid_evaluation_operator(b2, gridspace(b2), grid(b2))*e2)
      @test coefficients(BasisFunctions.default_evaluation_operator(b2, gridspace(b1))*e2) ≈ coefficients(grid_evaluation_operator(b2, gridspace(b1), grid(b1))*e2)
      @test coefficients(BasisFunctions.default_evaluation_operator(b1, gridspace(b1))*e1) ≈ coefficients(grid_evaluation_operator(b1, gridspace(b1), grid(b1))*e1)
      @test coefficients(BasisFunctions.default_evaluation_operator(b1, gridspace(b2))*e1) ≈ coefficients(grid_evaluation_operator(b1, gridspace(b2), grid(b2))*e1)

      mr = matrix(restriction_operator(b1, b2))
      me = matrix(extension_operator(b2, b1))
      pinvme = pinv(me)
      r = rand(size(pinvme,2))
      @test pinvme*r ≈ mr*r
    end
  end

  @test_throws AssertionError restriction_operator(SymBSplineTranslatesBasis(4,1), SymBSplineTranslatesBasis(3,1))
  @test_throws AssertionError extension_operator(SymBSplineTranslatesBasis(4,1), SymBSplineTranslatesBasis(6,1))
  @test_throws MethodError restriction_operator(SymBSplineTranslatesBasis(4,0), SymBSplineTranslatesBasis(2,0))
  @test_throws MethodError extension_operator(SymBSplineTranslatesBasis(3,0), SymBSplineTranslatesBasis(6,0))
end

function test_orthonormalsplinebasis(T)
  b = OrthonormalSplineBasis(5,2,Float64)
  b = OrthonormalSplineBasis(5,2,T)
  @test name(b) == "Set of translates of a function (B spline of degree 2) (orthonormalized)"
  @test instantiate(OrthonormalSplineBasis,5)==OrthonormalSplineBasis(5,3)

  G = sqrt(DualGram(b.superset))
  e = zeros(eltype(G),size(G,1))
  e[1] = 1
  @test b.coefficients ≈ G*e

  d = BasisFunctions.primalgramcolumn(b; abstol=1e-3)
  @test d ≈ zeros(T,d)
  @test typeof(Gram(b)) <: IdentityOperator
end

# exit()
# using Base.Test
# using BasisFunctions
# @testset begin test_orthonormalsplinebasis(BigFloat) end
# @testset begin test_translatedbsplines(Float64) end
# @testset begin test_translatedsymmetricbsplines(Float64) end
# @testset begin test_generic_periodicbsplinebasis(Float64) end

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
