function elementarypropsofsplinetest(T)
  T = real(T)
  tol = sqrt(eps(T))
  S = 20
  for N in 1:10
    f = x->BasisFunctions.Cardinal_b_splines.evaluate_Bspline(N-1, x, Float64)
    # Integral should be 1
    if !(T <: BigFloat)
      I,e = quadgk(f, 0, N, reltol = tol)
      @test I≈T(1)
    end
    # Infinite summation of shifted versions is 1
    xx = linspace(T(N-1), T(N), S)[1:end-1]
    g = zeros(T,xx)
    for k in 0:N-1
      g += map(x->f(x-k), xx)
    end
    @test g ≈ ones(T,g)  # (norm(g-1) < tol)
    # Two scale relation
    x = linspace(T(-1), T(N+1), S)
    g = zeros(T,x)
    for k in 0:N
      g += T(binomial(N,k))*map(x->f(2x-k), x)
    end
    g *= T(2)^(-N+1)
    G = map(f, x)
    @test g ≈ G
  end
end

function periodicbsplinetest(T)
  for N in 0:4
    period = T(N+1)
    for x in linspace(T(0),period,10)[1:end-1]
      @test (BasisFunctions.Cardinal_b_splines.evaluate_periodic_Bspline(N, x, period, T) ≈ BasisFunctions.Cardinal_b_splines.evaluate_Bspline(N, x, T))
      for k in -2:2
        @test (BasisFunctions.Cardinal_b_splines.evaluate_periodic_Bspline(N, x, period, T) ≈ BasisFunctions.Cardinal_b_splines.evaluate_periodic_Bspline(N, x+k*period, period, T))
      end
    end
    period = T(N+1)/T(3)
    for x in linspace(0,period,10)[1:end-1]
      for k in -2:2
        @test (BasisFunctions.Cardinal_b_splines.evaluate_periodic_Bspline(N, x, period, T) ≈ BasisFunctions.Cardinal_b_splines.evaluate_periodic_Bspline(N, x+k*period, period, T))
      end
    end
  end
end
function symmetricbsplinestest(T)
  K = 20
  for N in 0:4
    xs = linspace(T(0)+eps(T), T(1), K)
    for x in xs
      @test BasisFunctions.Cardinal_b_splines.evaluate_symmetric_periodic_Bspline(N, x, T(10(N+1)), T) ≈ BasisFunctions.Cardinal_b_splines.evaluate_symmetric_periodic_Bspline(N, -x, T(10(N+1)), T)
    end
  end
end
# using BasisFunctions
# using Base.Test
# P = 80
# T = Float64
# @testset "$(rpad("Elementary properties",P))" begin
#   elementarypropsofsplinetest(T)
# end
# @testset "$(rpad("periodic B splines",P))"  begin
#   periodicbsplinetest(T)
# end
# @testset "$(rpad("symmetric B splines",P))"  begin
#   symmetricbsplinestest(T)
# end
