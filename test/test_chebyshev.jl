
chebyshev_types = (Float32, Float64, BigFloat)

function test_chebyshev_expansions(T)
    n = 10
    b = ChebyshevT{T}(n)
    if T == Float32 || T == Float64
        g = BasisFunctions.secondgrid(b)
        BasisFunctions.Test.test_generic_dict_transform(b, g)
        BasisFunctions.Test.test_generic_dict_transform(complex(b), g)
    end

    @test support(b) == ChebyshevInterval{T}()
end

for T in chebyshev_types
    @testset "$(rpad("Chebyshev expansions ($T)",80))" begin
        test_chebyshev_expansions(T)
    end
end
