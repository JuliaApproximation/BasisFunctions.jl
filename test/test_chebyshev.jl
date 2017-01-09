# test_chebyshev.jl


# Chebyshev polynomials
function test_chebyshev(T)
    n1 = 160
    b1 = ChebyshevBasis(n1, T)
    A = approximation_operator(b1)
    f = exp
    e = approximate(b1, exp)
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
