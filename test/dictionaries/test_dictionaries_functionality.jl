
function test_generic_conversion(b1, b2)
    f = random_expansion(b1)
    x = fixed_point_in_domain(b1)
    C = conversion(b1, b2)
    tolerance = sqrt(eps(prectype(b1)))
    @test src(C) == b1
    @test dest(C) == b2
    g = C*f
    @test g(x) ≈ f(x)
    if length(b1) == length(b2)
        Cinv = conversion(b2, b1)
        f2 = Cinv * (C*f)
        @test norm(coefficients(f2-f2)) < tolerance
    end
end


function test_dictionary_conversions()
    # OPS are tested separately in test_ops

    @test_throws ErrorException conversion(Fourier(5), Fourier(4))
    test_generic_conversion(Fourier(5), Fourier(6))
    test_generic_conversion(Fourier(5), Fourier(7))
    test_generic_conversion(Fourier(6), Fourier(7))
    test_generic_conversion(Fourier(6), Fourier(8))

    test_generic_conversion(Fourier(4) → (-1..1), Fourier(5) → (-1..1))

    test_generic_conversion(ChebyshevT(3), complex(ChebyshevT(4)))
    test_generic_conversion(complex(ChebyshevT(3)), ChebyshevT(4))
    test_generic_conversion(complex(ChebyshevT(3)), complex(ChebyshevT(4)))
    test_generic_conversion(complex(ChebyshevT(3)) → -2..2, ChebyshevT(4) → -2..2)
end
