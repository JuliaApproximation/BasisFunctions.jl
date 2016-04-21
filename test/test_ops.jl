# test_ops.jl

#####
# Orthogonal polynomials
#####

function test_ops(T)
    delimit("Chebyshev polynomials")

    bc = ChebyshevBasis(12, T)

    x1 = T(4//10)
    @test bc(4, x1) ≈ cos(3*acos(x1))


    delimit("Legendre polynomials")

    bl = LegendreBasis{T}(15)

    x1 = T(4//10)
    @test abs(bl(6, x1) - 0.27064) < 1e-5


    delimit("Jacobi polynomials")

    bj = JacobiBasis(15, T(2//3), T(3//4))

    x1 = T(4//10)
    @test abs(bj(6, x1) - 0.335157) < 1e-5


    delimit("Laguerre polynomials")

    bl = LaguerreBasis(15, T(1//3))

    x1 = T(4//10)
    @test abs(bl(6, x1) + 0.08912346) < 1e-5

    delimit("Hermite polynomials")

    bh = HermiteBasis{T}(15)

    x1 = T(4//10)
    @test abs(bh(6, x1) - 38.08768) < 1e-5

end
