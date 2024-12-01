

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
