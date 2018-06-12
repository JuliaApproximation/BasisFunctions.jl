# test_ops.jl

#####
# Orthogonal polynomials
#####

function test_ops(T)
    println("Classical orthogonal polynomials:")
    test_chebyshev(T)
    test_legendre(T)
    test_laguerre(T)
    test_hermite(T)
    test_jacobi(T)
end

function test_chebyshev(T)
    println("- Chebyshev polynomials")
    bc = ChebyshevBasis{T}(12)
    test_ops_generic(bc)
    x1 = T(4//10)
    @test bc[4](x1) ≈ cos(3*acos(x1))

    @test support(bc) == ChebyshevInterval{T}()

    n1 = 160
    b1 = ChebyshevBasis{T}(n1)
    A = approximation_operator(Span(b1))
    f = exp
    e = approximate(Span(b1), exp)
    x0 = T(1//2)
    @test abs(e(T(x0))-f(x0)) < sqrt(eps(T))

    @test has_transform(b1)
    @test has_transform(b1, grid(b1))
    @test has_transform(b1, ChebyshevExtremaGrid(n1))
    @test has_transform(b1, ChebyshevNodeGrid(n1))
    @test ~has_transform(b1, PeriodicEquispacedGrid(n1, -1, 1))
    @test ~has_transform(b1, ChebyshevExtremaGrid(n1+5))
    @test ~has_transform(b1, ChebyshevNodeGrid(n1+5))
end

function test_legendre(T)
    println("- Legendre polynomials")
    bl = LegendrePolynomials{T}(10)
    test_ops_generic(bl)
    x1 = T(4//10)
    @test abs(bl[6](x1) - 0.27064) < 1e-5
end

function test_laguerre(T)
    println("- Laguerre polynomials")
    bl = LaguerrePolynomials(10, T(1//3))
    test_ops_generic(bl)
    x1 = T(4//10)
    @test abs(bl[6](x1) + 0.08912346) < 1e-5
end

function test_hermite(T)
    println("- Hermite polynomials")
    bh = HermitePolynomials{T}(10)
    test_ops_generic(bh)
    x1 = T(4//10)
    @test abs(bh[6](x1) - 38.08768) < 1e-5
end

function test_jacobi(T)
    println("- Jacobi polynomials")
    bj = JacobiPolynomials(10, T(2//3), T(3//4))
    test_ops_generic(bj)
    x1 = T(4//10)
    @test abs(bj[6](x1) - 0.335157) < 1e-5
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

    d1 = BasisFunctions.unsafe_eval_element_derivative(ops, length(ops), x)
    d2 = recurrence_eval_derivative(ops, length(ops), x)
    @test abs(d1-d2) < tol

    if codomaintype(ops) == Float64
        # We only do these tests for Float64 because eig currently does not support BigFloat
        r = roots(ops)
        @test maximum(abs.(BasisFunctions.unsafe_eval_element.(ops, length(ops)+1, r))) < 100tol

        x,w = gauss_rule(ops)
        @test abs(sum(w) - first_moment(ops)) < tol
    end
end
