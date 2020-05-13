
using BasisFunctions, BasisFunctions.Test, DomainSets

using Test

types = (Float64, BigFloat)

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

    test_orthogonality_orthonormality(b1, true, false, BasisFunctions.ChebyshevUMeasure{T}())
    test_orthogonality_orthonormality(b1, true, false, gauss_rule(b1))
    test_orthogonality_orthonormality(b1, true, false, gauss_rule(resize(b1,length(b1)+1)))
end

function test_legendre(T)
    println("- Legendre polynomials ($T)")
    bl = Legendre{T}(10)
    test_ops_generic(bl)
    x1 = T(4//10)
    @test abs(bl[6](x1) - 0.27064) < 1e-5


    test_orthogonality_orthonormality(bl, true, false, BasisFunctions.LegendreMeasure{T}())
    test_orthogonality_orthonormality(bl, true, false, gauss_rule(bl))
    test_orthogonality_orthonormality(bl, true, false, gauss_rule(resize(bl,length(bl)+1)))
end

function test_laguerre(T)
    println("- Laguerre polynomials ($T)")
    bl = Laguerre(10, T(1//3))
    test_ops_generic(bl)
    x1 = T(4//10)
    @test abs(bl[6](x1) + 0.08912346) < 1e-5

    test_orthogonality_orthonormality(bl, true, false, BasisFunctions.LaguerreMeasure(T(1//3)))
    test_orthogonality_orthonormality(Laguerre(10,T(0)), true, true, BasisFunctions.LaguerreMeasure(T(0)))
    test_orthogonality_orthonormality(bl, true, false, gauss_rule(bl))
    test_orthogonality_orthonormality(bl, true, false, gauss_rule(resize(bl,length(bl)+1)))
end

function test_hermite(T)
    println("- Hermite polynomials ($T)")
    bh = Hermite{T}(10)
    test_ops_generic(bh)
    x1 = T(4//10)
    @test abs(bh[6](x1) - 38.08768) < 1e-5

    test_orthogonality_orthonormality(bh, true, false, BasisFunctions.HermiteMeasure{T}())
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

    test_orthogonality_orthonormality(bj, true, false, BasisFunctions.JacobiMeasure(T(2//3), T(3//4)))
    test_orthogonality_orthonormality(bj, true, false, gauss_rule(bj))
    test_orthogonality_orthonormality(bj, true, false, gauss_rule(resize(bj,length(bj)+1)))
end


function test_ops_generic(ops)
    T = codomaintype(ops)
    tol = test_tolerance(T)

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

    if prectype(ops) == Float64
        # We only do these tests for Float64 because eig currently does not support BigFloat
        r = roots(ops)
        @test maximum(abs.(BasisFunctions.unsafe_eval_element.(Ref(ops), length(ops)+1, r))) < 100tol

        m = gauss_rule(ops)
        x = points(m)
        w = BasisFunctions.weights(m)
        @test abs(sum(w) - first_moment(ops)) < tol
    end
end

for T in types
    @testset "$(rpad("Orthogonal polynomials ($T)",80))" begin
        println("Classical orthogonal polynomials ($T):")
        test_chebyshevT(T)
        test_chebyshevU(T)
        test_legendre(T)
        test_laguerre(T)
        test_hermite(T)
        test_jacobi(T)
    end
    println()
end

@testset "$(rpad("Orthogonality of orthogonal polynomials",80))" begin
    OPSs = [ChebyshevT, ChebyshevU, Legendre, Hermite, Jacobi, Laguerre]
    for ops in OPSs, n in (5,6), T in (Float64,BigFloat)
        B = ops{T}(n)
        test_orthogonality_orthonormality(B, gauss_rule(B))
        test_orthogonality_orthonormality(B, gauss_rule(resize(B,2n)))
        test_orthogonality_orthonormality(B, gauss_rule(resize(B,n-1)))
    end
end
