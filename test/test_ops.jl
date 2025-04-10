
using BasisFunctions, BasisFunctions.Test, DomainSets, DoubleFloats

using Test

types = (Float64, LargeFloat)

#####
# Orthogonal polynomials
#####

function test_chebyshevT(T)
    println("- ChebyshevT polynomials ($T)")
    bc = ChebyshevT{T}(20)
    test_ops_generic(bc)
    x1 = T(4//10)
    @test bc[4](x1) ≈ cos(3*acos(x1))

    @test support(bc) == ChebyshevInterval{T}()

    n1 = 160
    b1 = ChebyshevT{T}(n1)

    n1 = 160
    b1 = ChebyshevT{T}(n1)
    f = exp
    e = approximate(b1, exp)
    x0 = T(1//2)
    @test abs(e(T(x0))-f(x0)) < sqrt(eps(T))

    @test hastransform(b1)
    @test hastransform(b1, interpolation_grid(b1))
    @test hastransform(b1, ChebyshevExtremae(n1))
    @test hastransform(b1, ChebyshevNodes(n1))
    @test ~hastransform(b1, PeriodicEquispacedGrid(n1, -1, 1))
    @test ~hastransform(b1, ChebyshevExtremae(n1+5))
    @test ~hastransform(b1, ChebyshevNodes(n1+5))
    @test BasisFunctions.jacobi_α(b1) ≈ -.5 ≈ BasisFunctions.jacobi_β(b1)
end

function test_chebyshevU(T)
    println("- ChebyshevU polynomials ($T)")
    n1 = 20
    b1 = ChebyshevU{T}(n1)
    test_ops_generic(b1)
    x1 = T(4//10)
    @test b1[4](x1) ≈ sin(4*acos(x1))/sin(acos(x1))

    @test support(b1) == ChebyshevInterval{T}()
    @test BasisFunctions.jacobi_α(b1) ≈ .5 ≈ BasisFunctions.jacobi_β(b1)

    test_orthogonality_orthonormality(b1, true, false, BasisFunctions.ChebyshevUWeight{T}())
    test_orthogonality_orthonormality(b1, true, false, gauss_rule(b1))
    test_orthogonality_orthonormality(b1, true, false, gauss_rule(resize(b1,length(b1)+1)))
end

function test_legendre(T)
    println("- Legendre polynomials ($T)")
    bl = Legendre{T}(10)
    test_ops_generic(bl)
    x1 = T(4//10)
    @test abs(bl[6](x1) - 0.27064) < 1e-5


    test_orthogonality_orthonormality(bl, true, false, BasisFunctions.LegendreWeight{T}())
    test_orthogonality_orthonormality(bl, true, false, gauss_rule(bl))
    test_orthogonality_orthonormality(bl, true, false, gauss_rule(resize(bl,length(bl)+1)))
end

function test_laguerre(T)
    println("- Laguerre polynomials ($T)")
    bl = Laguerre(10, T(1//3))
    test_ops_generic(bl)
    x1 = T(4//10)
    @test abs(bl[6](x1) + 0.08912346) < 1e-5

    test_orthogonality_orthonormality(bl, true, false, BasisFunctions.LaguerreWeight(T(1//3)))
    test_orthogonality_orthonormality(Laguerre(10,T(0)), true, true, BasisFunctions.LaguerreWeight(T(0)))
    test_orthogonality_orthonormality(bl, true, false, gauss_rule(bl))
    test_orthogonality_orthonormality(bl, true, false, gauss_rule(resize(bl,length(bl)+1)))
end

function test_hermite(T)
    println("- Hermite polynomials ($T)")
    bh = Hermite{T}(6)
    test_ops_generic(bh, tol = T == Float64 ? 1e-5 : T(1e-17)) # larger tolerance because the polynomials grow large
    x1 = T(4//10)
    @test abs(bh[6](x1) - 38.08768) < 1e-5

    test_orthogonality_orthonormality(bh, true, false, BasisFunctions.HermiteWeight{T}())
    test_orthogonality_orthonormality(bh, true, false, gauss_rule(bh))
    test_orthogonality_orthonormality(bh, true, false, gauss_rule(resize(bh,length(bh)+1)))
end

function test_jacobi(T)
    println("- Jacobi polynomials ($T)")
    bj = Jacobi(10, T(2//3), T(3//4))
    test_ops_generic(bj)
    x1 = T(4//10)
    @test abs(bj[6](x1) - 0.335157) < 1e-5
    @test BasisFunctions.jacobi_α(bj) ≈ T(2//3)
    @test BasisFunctions.jacobi_β(bj) ≈ T(3//4)

    test_orthogonality_orthonormality(bj, true, false, BasisFunctions.JacobiWeight(T(2//3), T(3//4)))
    test_orthogonality_orthonormality(bj, true, false, gauss_rule(bj))
    test_orthogonality_orthonormality(bj, true, false, gauss_rule(resize(bj,length(bj)+1)))
end

function test_ultraspherical(T)
    println("- Ultraspherical polynomials ($T)")
    λ = T(2//3)
    α = λ - T(1/2)
    bj = Ultraspherical(10, λ)
    jac = Jacobi(10, α, α)
    test_ops_generic(bj)
    x1 = T(4//10)
    factor = BasisFunctions.pochhammer(2λ, 5) / BasisFunctions.pochhammer(λ+T(1)/2, 5)
    @test abs(bj[6](x1) - factor*jac[6](x1)) < 1e-5
    @test BasisFunctions.jacobi_α(bj) ≈ α
    @test BasisFunctions.jacobi_β(bj) ≈ α

    test_orthogonality_orthonormality(bj, true, false, BasisFunctions.UltrasphericalWeight(λ))
    test_orthogonality_orthonormality(bj, true, false, gauss_rule(bj))
    test_orthogonality_orthonormality(bj, true, false, gauss_rule(resize(bj,length(bj)+1)))
end


function test_ops_generic(ops; T = codomaintype(ops), tol = test_tolerance(T))
    x = fixed_point_in_domain(ops)
    z1 = BasisFunctions.unsafe_eval_element(ops, length(ops), x)
    z2 = recurrence_eval(ops, length(ops), x)
    @test abs(z1-z2) < tol
    a,b = monic_recurrence_coefficients(ops)
    z3 = monic_recurrence_eval(a, b, length(ops), x)
    γ = leading_order_coefficient(ops, length(ops))
    @test abs(z1 - γ*z3) < tol

    d1 = BasisFunctions.unsafe_eval_element_derivative(ops, length(ops), x, 1)
    d2 = recurrence_eval_derivative(ops, length(ops), x)
    @test abs(d1-d2) < tol

    r = ops_roots(ops)
    @test maximum(abs.(BasisFunctions.unsafe_eval_element.(Ref(ops), length(ops)+1, r))) < 100tol

    m = gauss_rule(ops)
    x = points(m)
    w = BasisFunctions.weights(m)
    @test abs(sum(w) - first_moment(ops)) < tol

    n = length(ops)
    if n > 1
        # larger tolerances due to numerical integration tests involved
        @test abs(integral(ops[n], measure(ops))) < sqrt(tol)
        @test abs(integral(x->ops[n](x), measure(ops))) < sqrt(tol)
        @test abs(innerproduct(ops[n], ops[n-1])) < sqrt(tol)
    end
    @test abs(innerproduct(ops[1], ops[1]) - BasisFunctions.default_dict_innerproduct(ops, 1, ops, 1, measure(ops))) < sqrt(tol)
    @test abs(innerproduct(ops[n], ops[n]) - BasisFunctions.default_dict_innerproduct(ops, n, ops, n, measure(ops))) < sqrt(tol)
end

# for T in types
for T in (Float64, LargeFloat)
    @testset "Orthogonal polynomials ($T)" begin
        println("Classical orthogonal polynomials ($T):")
        test_chebyshevT(T)
        test_chebyshevU(T)
        test_legendre(T)
        test_laguerre(T)
        test_hermite(T)
        test_jacobi(T)
        test_ultraspherical(T)
    end
    println()
end

using BasisFunctions: isequaldict
@testset "Conversions of expansions" begin
    @test isequaldict(Legendre(5), Jacobi(5, 0, 0))
    @test isequaldict(Jacobi(5, 0, 0), Legendre(5))
    @test isequaldict(Legendre(5), Ultraspherical(5, 1/2))
    @test isequaldict(Ultraspherical(5, 1/2), Legendre(5))
    @test isequaldict(Jacobi(5, 0, 0), Ultraspherical(5, 1/2))
    @test isequaldict(Ultraspherical(5, 1/2), Jacobi(5, 0, 0))
    @test !isequaldict(Jacobi(6, 0, 0), Ultraspherical(5, 1/2))
    @test !isequaldict(Jacobi(5, 0.1, 0), Ultraspherical(5, 1/2))
    @test !isequaldict(Jacobi(5, 0, 0), Ultraspherical(5, 1))
    @test isequaldict(ChebyshevU(5), Ultraspherical(5, 1))
    @test isequaldict(Ultraspherical(5, 1), ChebyshevU(5))
    for T in types
        N = 5
        b1 = Jacobi{T}(N, 1.4, 0.3)
        test_generic_conversion(b1, Legendre{T}(N))
        test_generic_conversion(b1, ChebyshevT{T}(N))
        test_generic_conversion(b1, Monomials{T}(N))
        test_generic_conversion(b1, Jacobi{T}(N, 0.2, 0.4))
        test_generic_conversion(b1, Laguerre{T}(N))
        b2 = Jacobi{T}(N+1, 1.4, 0.3)
        @test conversion(b1, b2) isa BasisFunctions.IndexExtension
        @test_throws ArgumentError conversion(b2, b1)
    end
    @test conversion(Legendre(5), Jacobi(5, 0, 0)) isa IdentityOperator

    test_generic_conversion(Ultraspherical(5, 0.2), Ultraspherical(5, 1.2))
    @test matrix(conversion(Ultraspherical(5, 0.2), Ultraspherical(5, 1.2))) isa BandedMatrices.BandedMatrix

    test_generic_conversion(ChebyshevT(5), Ultraspherical(5, 1))
    @test matrix(conversion(ChebyshevT(5), Ultraspherical(5, 1))) isa BandedMatrices.BandedMatrix

    test_generic_conversion(ChebyshevT(3), Jacobi(3, -1/2, -1/2))
    @test matrix(conversion(ChebyshevT(3), Jacobi(3, -1/2, -1/2))) isa Diagonal
    @test !(matrix(conversion(ChebyshevT(3), Jacobi(3, -1/2, 1/2))) isa Diagonal)
    @test matrix(conversion(Jacobi(3, -1/2, -1/2), ChebyshevT(3))) isa Diagonal
    @test !(matrix(conversion(Jacobi(3, -1/2, 1/2), ChebyshevT(3))) isa Diagonal)

    test_generic_conversion(ChebyshevU(3), Jacobi(3, 1/2, 1/2))
    @test matrix(conversion(ChebyshevU(3), Jacobi(3, 1/2, 1/2))) isa Diagonal
    @test !(matrix(conversion(ChebyshevU(3), Jacobi(3, -1/2, 1/2))) isa Diagonal)
    @test matrix(conversion(Jacobi(3, 1/2, 1/2), ChebyshevU(3))) isa Diagonal
    @test !(matrix(conversion(Jacobi(3, -1/2, 1/2), ChebyshevU(3))) isa Diagonal)

    test_generic_conversion(Ultraspherical(3, 2.0), Jacobi(3, 3/2, 3/2))
    @test matrix(conversion(Ultraspherical(3, 2.0), Jacobi(3, 3/2, 3/2))) isa Diagonal
    @test !(matrix(conversion(Ultraspherical(3, 2.0), Jacobi(3, -1/2, 1/2))) isa Diagonal)
    @test matrix(conversion(Jacobi(3, 3/2, 3/2), Ultraspherical(3, 2.0))) isa Diagonal
    @test !(matrix(conversion(Jacobi(3, -1/2, 1/2), Ultraspherical(3, 2.0))) isa Diagonal)
end

@testset "Orthogonality of orthogonal polynomials" begin
    OPSs = [ChebyshevT, ChebyshevU, Legendre, Hermite, Jacobi, Laguerre]
    for ops in OPSs, n in (5,6), T in (Float64,LargeFloat)
        B = ops{T}(n)
        test_orthogonality_orthonormality(B, gauss_rule(B))
        test_orthogonality_orthonormality(B, gauss_rule(resize(B,2n)))
        test_orthogonality_orthonormality(B, gauss_rule(resize(B,n-1)))
    end
end

@testset "OPS differentiation matrices" begin
end
