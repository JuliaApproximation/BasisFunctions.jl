function test_derived_sets(T)
    b1 = FourierBasis(11, T)
    b2 = ChebyshevBasis(12, T)

    @testset "$(rpad("Generic derived set",80))" begin
    test_generic_set_interface(BasisFunctions.ConcreteDerivedSet(b2)) end

    @testset "$(rpad("Linear mapped sets",80))" begin
    test_generic_set_interface(rescale(b1, -T(1), T(2)))
    test_generic_set_interface(rescale(b2, -T(2), T(3)))
    end

    @testset "$(rpad("A simple subset",80))" begin
    test_generic_set_interface(b1[2:6]) end

    @testset "$(rpad("Operated sets",80))" begin
    test_generic_set_interface(OperatedSet(differentiation_operator(b1))) end

    @testset "$(rpad("Weighted sets",80))" begin
        # Try a functor
        test_generic_set_interface(BF.Cos() * b1)
        # as well as a regular function
        test_generic_set_interface(cos * b1)
        # and a 2D example
        # (not just yet, fix 2D for BigFloat first)
#        test_generic_set_interface( ((x,y) -> cos(x+y)) * tensorproduct(b1, 2) )
    end


    @testset "$(rpad("Multiple sets",80))" begin
    # Test sets with internal representation as vector and as tuple separately
    test_generic_set_interface(multiset(b1,rescale(b2, 0, 1)))
    test_generic_set_interface(MultiSet((b1,rescale(b2, 0, 1)))) end

    @testset "$(rpad("A multiple and weighted set combination",80))" begin
    s = rescale(b1, 1/2, 1)
    test_generic_set_interface(multiset(s,Log()*s)) end

    @testset "$(rpad("A complicated subset",80))" begin
    s = rescale(b1, 1/2, 1)
    test_generic_set_interface(s[1:5]) end

    @testset "$(rpad("Piecewise sets",80))" begin
    part = PiecewiseInterval(T(0), T(10), 10)
    pw = PiecewiseSet(b2, part)
    test_generic_set_interface(pw) end

    # @testset "$(rpad("A tensor product of MultiSet's",80))" begin
    # b = multiset(b1,b2)
    # c = b ⊗ b
    # test_generic_set_interface(c) end
end
