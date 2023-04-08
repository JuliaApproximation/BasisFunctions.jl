
function test_dictionary_conversions()
    b_cheb = ChebyshevT(10)
    b_leg = Legendre(10)
    b_mono = Monomials(10)

    e_cheb = random_expansion(b_cheb)
    e_leg = random_expansion(b_leg)
    e_mono = random_expansion(b_mono)

    C_cheb2leg = conversion(b_cheb, b_leg)
    e1 = C_cheb2leg * e_cheb
    @test dictionary(e1) == b_leg
    @test e_cheb(0.4) ≈ e1(0.4)

    C_leg2cheb = conversion(b_leg, b_cheb)
    e2 = C_leg2cheb * e_leg
    @test dictionary(e2) == b_cheb
    @test e_leg(0.4) ≈ e2(0.4)

    C_leg2mono = conversion(b_leg, b_mono)
    e3 = C_leg2mono * e_leg
    @test dictionary(e3) == b_mono
    @test e3(0.4) ≈ e_leg(0.4)

    C_mono2leg = conversion(b_mono, b_leg)
    e4 = C_mono2leg * e_mono
    @test dictionary(e4) == b_leg
    @test e4(0.4) ≈ e_mono(0.4)
end
