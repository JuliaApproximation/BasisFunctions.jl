using Calculus
function test_derived_dicts(T)
    b1 = Fourier{T}(11)
    b2 = ChebyshevT{T}(12)

    @testset "$(rpad("Generic derived dictionary",80))" begin
        test_generic_dict_interface(BasisFunctions.ConcreteDerivedDict(b2))
    end

    @testset "$(rpad("Complexified dictionary",80))" begin
        # Something is wrong with BigFloat, the idct isn't right. Consult with Daan.
        b3 = ChebyshevT{Float64}(12)
        # First, check that a complexified dict is made
        b3c = complex(b3)
        @test b3c isa BasisFunctions.ComplexifiedDict
        @test coefficienttype(b3c) == Complex{Float64}
        # and check that it doesn't create a nested ComplexifiedDict
        @test complex(b3c) isa BasisFunctions.ComplexifiedDict{<:ChebyshevT}
        test_generic_dict_interface(b3c)
    end
    @testset "$(rpad("Linear mapped dictionaries",80))" begin
        test_generic_dict_interface(rescale(b1, -T(1), T(2)))
        test_generic_dict_interface(rescale(b2, -T(2), T(3)))
    end

    @testset "$(rpad("A simple subdict",80))" begin
        test_generic_dict_interface(b1[2:6]) end

    @testset "$(rpad("Operated dictionaries",80))" begin
        D1 = DiagonalOperator(b1, rand(length(b1)))
        test_generic_dict_interface(D1*b1)
        D2 = ArrayOperator(rand(length(b1),length(b1)), b1, b1)
        test_generic_dict_interface(D2*b1)
        D3 = differentiation(b2)
        test_generic_dict_interface(D3*b2)
    end


    @testset "$(rpad("Multiple dictionaries",80))" begin
        # Test sets with internal representation as vector and as tuple separately
        test_generic_dict_interface(multidict(b1,rescale(b2, 0, 1)))
        #test_generic_dict_interface(MultiDict((b1,rescale(b2, 0, 1)))) end
        end

    @testset "$(rpad("Weighted dictionaries",80))" begin
        # as well as a regular function
        test_generic_dict_interface(cos * b1)
        # and a 2D example
        # (not just yet, fix 2D for BigFloat first)
#        test_generic_dict_interface( ((x,y) -> cos(x+y)) * tensorproduct(b1, 2) )
    end

    @testset "$(rpad("A multiple and weighted dict combination",80))" begin
        s = rescale(b1, 1/2, 1)
        test_generic_dict_interface(multidict(s,log*s)) end

    @testset "$(rpad("A complicated subdict",80))" begin
        s = rescale(b1, 1/2, 1)
        test_generic_dict_interface(s[1:5]) end

    @testset "$(rpad("Piecewise dictionaries",80))" begin
        # Changed the number of partitions to 7 (from 10) in order to avoid alignment of a
        # grid point in one of the tests with the boundary of a subinterval.
        part = PiecewiseInterval(T(0), T(10), 7)
        pw = PiecewiseDict(b2, part)
        test_generic_dict_interface(pw) end

    # @testset "$(rpad("A tensor product of MultiDict's",80))" begin
    # b = multidict(b1,b2)
    # c = b ⊗ b
    # test_generic_dict_interface(c) end
end
