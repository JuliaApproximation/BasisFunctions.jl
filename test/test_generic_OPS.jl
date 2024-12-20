
using BasisFunctions, FastGaussQuadrature, DomainSets, DoubleFloats
⊕ = BasisFunctions.:⊕

using Test


function test_half_range_chebyshev()
    n = 60
    T = 2.
    b1 = BasisFunctions.HalfRangeChebyshevIkind(n,T)
    b2 = BasisFunctions.HalfRangeChebyshevIIkind(n,T)

    α = -.5
    nodes, weights = gaussjacobi(100, α, 0)
    modified_weights = weights.*(nodes .- BasisFunctions.m_forward(T,T)).^α

    αstieltjes,βstieltjes = BasisFunctions.stieltjes(n,nodes,modified_weights)
    setprecision(350)
    αchebyshev,βchebyshev = BasisFunctions.modified_chebyshev(μ.(0:2n))

    @test maximum(abs.(αstieltjes-αchebyshev)) < 1e-12
    @test maximum(abs.(βstieltjes-βchebyshev)) < 1e-12
end

# Exact moments of HalfRangeChebyshevIkind for T = 2
μ(k::BigInt) = sum(binomial(k,i)*(-1)^(k-i)*2^i*my_I(i) for i in 0:k)
#util
my_I(k, T=BigFloat) = iseven(k) ? T(π)/2*T(_coef(k)) : T(_coef(k))
_coef(k::BigInt) = (k==1 || k==0) ? 1 : (k-1)//k*_coef(k-2)
μ(k::Int) = μ(BigInt(k))

function test_generic_ops_from_quadrature()
    N = 10
    # Legendre
    c1 = BasisFunctions.OrthonormalOPSfromQuadrature(N,N->gaussjacobi(N, 0., 0.),Interval(-1,1))
    c2 = Legendre(N)

    compare_OPS(N, c1, c2, -1, 1, 2, 1)

    c1 = BasisFunctions.OrthonormalOPSfromQuadrature(N,N->gaussjacobi(N, 0., 0.),Interval(-1,1))
    c2 = Jacobi(N, 0, 0)

    compare_OPS(N, c1, c2, -1, 1, 2, 1)

    # chebyshevI
    c1 = BasisFunctions.OrthonormalOPSfromQuadrature(N,N->gaussjacobi(N, -.5, -.5),Interval(-1,1))
    c2 = ChebyshevT(N)

    compare_OPS(N, c1, c2, -1, 1, pi, 1)

    c1 = BasisFunctions.OrthonormalOPSfromQuadrature(N,N->gaussjacobi(N, -.5, -.5),Interval(-1,1))
    c2 = Jacobi(N, -.5, -.5)

    compare_OPS(N, c1, c2, -1, 1, pi, 1)

    # chebyshevII
    c1 = BasisFunctions.OrthonormalOPSfromQuadrature(N,N->gaussjacobi(N, .5, .5),Interval(-1,1))
    c2 = ChebyshevU(N)

    compare_OPS(N, c1, c2, -1, 1, pi/2, 1)

    c1 = BasisFunctions.OrthonormalOPSfromQuadrature(N,N->gaussjacobi(N, .5, .5),Interval(-1,1))
    c2 = Jacobi(N, .5, .5)

    compare_OPS(N, c1, c2, -1, 1, pi/2, 1)

    # ChebyshevIII
    c1 = BasisFunctions.OrthonormalOPSfromQuadrature(N,N->gaussjacobi(N, -.5, .5),Interval(-1,1))
    c2 = Jacobi(N, -.5, .5)

    compare_OPS(N, c1, c2, -1, 1, pi, 1)

    # ChebyshevIIII
    c1 = BasisFunctions.OrthonormalOPSfromQuadrature(N,N->gaussjacobi(N, .5, -.5),Interval(-1,1))
    c2 = Jacobi(N, .5, -.5)

    compare_OPS(N, c1, c2, -1, 1, pi, 1)
end

function compare_OPS(N, c1::GenericOPS, c2, left_point, right_point, firstmoment, constant)
    t = range(left_point, stop=right_point, length=20)
    s = zeros(N)
    for k in 1:N
        y1 = c1[k].(t)
        y2 = c2[k].(t)

        y = y1./y2
        s[k] = y[1]
        y /= y[1]
        y .-= 1
        @test maximum(y) < 1e-10
    end

    @test dict_support(c1,1) == Interval(left_point,right_point)
    @test length(c1) == N

    @test first_moment(c1) ≈ firstmoment
    @test BasisFunctions.p0(c1) ≈ 1/sqrt(firstmoment)
    A = BasisFunctions.rec_An.(Ref(c2),0:N-1)
    B = BasisFunctions.rec_Bn.(Ref(c2),0:N-1)
    C = BasisFunctions.rec_Cn.(Ref(c2),0:N-1)
    D = BasisFunctions.rec_An.(Ref(c1),0:N-1)
    E = BasisFunctions.rec_Bn.(Ref(c1),0:N-1)
    F = BasisFunctions.rec_Cn.(Ref(c1),0:N-1)

    @test 1 .+ A[1:N-1] ≈ 1 .+ D[1:N-1].*s[1:N-1]./s[2:N]
    @test 1 .+ B[1:N-1] ≈ 1 .+ E[1:N-1].*s[1:N-1]./s[2:N]
    @test 1 .+ C[2:N-1] ≈ 1 .+ F[2:N-1].*s[1:N-2]./s[3:N]

    a, b = monic_recurrence_coefficients(c1)
    c, d = monic_recurrence_coefficients(c2)

    @test 1 .+ a ≈ 1 .+ c
    @test 1 .+ b ≈ 1 .+ d

    for k in 1:N
        f = [monic_recurrence_eval(a, b, k ,x) for x in t]
        g = [monic_recurrence_eval(c, d, k ,x) for x in t]
        @test 1 .+ f ≈ 1 .+ g
    end
end

function test_gaussjacobi()
    w0 = LargeFloat(128//225)
    w1 = (sqrt(LargeFloat(70))*13+322)/900
    w2 = (-sqrt(LargeFloat(70))*13+322)/900
    weights_test = [w2,w1,w0,w1,w2]
    n0 = LargeFloat(0)
    n1 = LargeFloat(1)/3*sqrt(5-2sqrt(LargeFloat(10)/7))
    n2 = LargeFloat(1)/3*sqrt(5+2sqrt(LargeFloat(10)/7))
    nodes_test = [-n2,-n1,n0,n1,n2]

    gjnodes, gjweights = gaussjacobi(5,0,0)
    @test Float64.(norm(gjweights-weights_test))+1≈1
    @test Float64.(norm(gjnodes-nodes_test))+1≈1

    gjnodes, gjweights = gaussjacobi(5,LargeFloat(0),LargeFloat(0))
    @test norm(gjweights-weights_test)+1≈1
    @test norm(gjnodes-nodes_test)+1≈1
end

function test_roots_of_legendre_halfrangechebyshev()
    N = 10
    B = BasisFunctions.HalfRangeChebyshevIkind(N,2.)
    @test 1+maximum(abs.(B[N].(real(ops_roots(resize(B,N-1))))))≈1
    B = Legendre(N)
    @test 1+maximum(abs.(B[N].(real(ops_roots(resize(B,N-1))))))≈1
    B = Legendre{LargeFloat}(N)
    @test 1+maximum(abs.(B[N].(real(ops_roots(resize(B,N-1))))))≈1
    # B = HalfRangeChebyshevIkind(N,LargeFloat(2))
    # @test 1+maximum(abs.(B[N].(real(ops_roots(resize(B,N-1))))))≈1
end

@testset "Generic OPS" begin
    test_generic_ops_from_quadrature()
end

@testset "Half Range Chebyshev" begin
    test_half_range_chebyshev()
end

@testset "Gauss jacobi quadrature" begin
    test_gaussjacobi()
    test_roots_of_legendre_halfrangechebyshev()
end
