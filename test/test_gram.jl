using BasisFunctions, Test, DoubleFloats

for T in (Float64,Double64)

    @testset "Discrete Gauss OPS Rule" begin
        for d in (Legendre{T}(10), Laguerre(10,rand(T)), Hermite{T}(10), ChebyshevT{T}(10),ChebyshevU{T}(10),Jacobi(10,rand(T),rand(T)),Fourier{T}(10))
            g = interpolation_grid(d)
            gr = gauss_rule(d)
            @test g ≈ points(gr)
            @test isorthogonal(d, gr)
            @test iscompatible(d, g)
            @test gram(d, gr) isa DiagonalOperator

            test_orthogonality_orthonormality(d, true, d isa Fourier, gr)

            @test BasisFunctions.quadweights(gauss_rule(d),measure(d)) ≈ ones(T, 10)
        end


        B = Fourier(10)
        d = B^2
        g = interpolation_grid(d)
        gr = gauss_rule(d)
        @test g ≈ points(gr)
        @test isorthogonal(d, gr)
        @test iscompatible(d, g)
        @test gram(d, gr) isa TensorProductOperator && all(map(x->isa(x, DiagonalOperator), elements(gram(d,gr))))
        @test BasisFunctions.quadweights(gauss_rule(d),measure(d)) ≈ BasisFunctions.OuterProductArray(map(ones, axes(B^2))...)

    end
end
