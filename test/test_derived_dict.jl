# test_derived_dict.jl

function test_derived_dicts(T)
    b1 = FourierBasis{T}(11)
    b2 = ChebyshevBasis{T}(12)

    @testset "$(rpad("Generic derived dictionary",80))" begin
        test_generic_dict_interface(BasisFunctions.ConcreteDerivedDict(b2))
    end

    @testset "$(rpad("Linear mapped dictionaries",80))" begin
        test_generic_dict_interface(rescale(b1, -T(1), T(2)))
        test_generic_dict_interface(rescale(b2, -T(2), T(3)))
    end

    @testset "$(rpad("A simple subdict",80))" begin
        test_generic_dict_interface(b1[2:6]) end

    @testset "$(rpad("Operated dictionaries",80))" begin
        test_generic_dict_interface(OperatedDict(differentiation_operator(Span(b1)))) end


    @testset "$(rpad("Multiple dictionaries",80))" begin
        # Test sets with internal representation as vector and as tuple separately
        test_generic_dict_interface(multidict(b1,rescale(b2, 0, 1)))
        test_generic_dict_interface(MultiDict((b1,rescale(b2, 0, 1)))) end

    @testset "$(rpad("Weighted dictionaries",80))" begin
        # Try a functor
        test_generic_dict_interface(BF.Cos() * b1)
        # as well as a regular function
        test_generic_dict_interface(cos * b1)
        # and a 2D example
        # (not just yet, fix 2D for BigFloat first)
#        test_generic_dict_interface( ((x,y) -> cos(x+y)) * tensorproduct(b1, 2) )
    end

    @testset "$(rpad("A multiple and weighted dict combination",80))" begin
        s = rescale(b1, 1/2, 1)
        test_generic_dict_interface(multidict(s,Log()*s)) end

    @testset "$(rpad("A complicated subdict",80))" begin
        s = rescale(b1, 1/2, 1)
        test_generic_dict_interface(s[1:5]) end

    @testset "$(rpad("Piecewise dictionaries",80))" begin
        part = PiecewiseInterval(T(0), T(10), 10)
        pw = PiecewiseDict(b2, part)
        test_generic_dict_interface(pw) end

    # @testset "$(rpad("A tensor product of MultiDict's",80))" begin
    # b = multidict(b1,b2)
    # c = b ⊗ b
    # test_generic_dict_interface(c) end
end
