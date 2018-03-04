# test_derived_set.jl

function test_derived_sets(T)
    b1 = FourierBasis{T}(11)
    b2 = ChebyshevBasis{T}(12)

    @testset "$(rpad("Generic derived set",80))" begin
        test_generic_set_interface(BasisFunctions.ConcreteDerivedDict(b2))
    end

    @testset "$(rpad("Linear mapped sets",80))" begin
        test_generic_set_interface(rescale(b1, -T(1), T(2)))
        test_generic_set_interface(rescale(b2, -T(2), T(3)))
    end

    @testset "$(rpad("A simple subset",80))" begin
        test_generic_set_interface(b1[2:6]) end

    @testset "$(rpad("Operated sets",80))" begin
        test_generic_set_interface(OperatedDict(differentiation_operator(Span(b1)))) end


    @testset "$(rpad("Multiple sets",80))" begin
        # Test sets with internal representation as vector and as tuple separately
        test_generic_set_interface(multidict(b1,rescale(b2, 0, 1)))
        test_generic_set_interface(MultiDict((b1,rescale(b2, 0, 1)))) end

    @testset "$(rpad("Weighted sets",80))" begin
        # Try a functor
        test_generic_set_interface(BF.Cos() * b1)
        # as well as a regular function
        test_generic_set_interface(cos * b1)
        # and a 2D example
        # (not just yet, fix 2D for BigFloat first)
#        test_generic_set_interface( ((x,y) -> cos(x+y)) * tensorproduct(b1, 2) )
    end

    @testset "$(rpad("A multiple and weighted set combination",80))" begin
        s = rescale(b1, 1/2, 1)
        test_generic_set_interface(multidict(s,Log()*s)) end

    @testset "$(rpad("A complicated subset",80))" begin
        s = rescale(b1, 1/2, 1)
        test_generic_set_interface(s[1:5]) end

    @testset "$(rpad("Piecewise sets",80))" begin
        part = PiecewiseInterval(T(0), T(10), 10)
        pw = PiecewiseDict(b2, part)
        test_generic_set_interface(pw) end

    # @testset "$(rpad("A tensor product of MultiDict's",80))" begin
    # b = multidict(b1,b2)
    # c = b ⊗ b
    # test_generic_set_interface(c) end
end
