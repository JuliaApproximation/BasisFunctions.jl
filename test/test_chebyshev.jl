# test_chebyshev.jl


# Chebyshev polynomials
function test_chebyshev(T)
    b1 = ChebyshevBasis(160, T)
    A = approximation_operator(b1)
    f = exp
    e = approximate(b1, exp)
    x0 = T(1//2)
    @test abs(e(T(x0))-f(x0)) < sqrt(eps(T))
end
