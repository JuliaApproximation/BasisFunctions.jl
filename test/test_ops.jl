# test_ops.jl

#####
# Orthogonal polynomials
#####

function test_ops(T)

    bc = ChebyshevBasis(12, T)

    x1 = T(4//10)
    @test bc[4](x1) ≈ cos(3*acos(x1))



    bl = LegendrePolynomials{T}(15)

    x1 = T(4//10)
    @test abs(bl[6](x1) - 0.27064) < 1e-5



    bj = JacobiPolynomials(15, T(2//3), T(3//4))

    x1 = T(4//10)
    @test abs(bj[6](x1) - 0.335157) < 1e-5



    bl = LaguerrePolynomials(15, T(1//3))

    x1 = T(4//10)
    @test abs(bl[6](x1) + 0.08912346) < 1e-5


    bh = HermitePolynomials{T}(15)

    x1 = T(4//10)
    @test abs(bh[6](x1) - 38.08768) < 1e-5

end

function test_ops_generic(ops)
    T = rangetype(ops)
    tol = test_tolerance(T)

    x = fixed_point_in_domain(ops)
    z1 = eval_element(ops, length(ops), x)
    z2 = BasisFunctions.recurrence_eval(ops, x)
    @test abs(z1-z2) < tol

    d1 = eval_element_derivative(ops, length(ops), x)
    d2 = BasisFunctions.recurrence_eval_derivative(ops, x)
    @test abs(d1-d2) < tol

    r = roots(ops)
    @test max(abs.(eval_element.(ops, r))) < tol
end
