test_orthogonality_orthonormality(dict::Dictionary, measure::BasisFunctions.Measure=measure(dict)) =
    test_orthogonality_orthonormality(dict, isorthogonal(dict, measure), isorthonormal(dict, measure), measure)


function test_orthogonality_orthonormality(dict::Dictionary, isorthog::Bool, isorthonorm::Bool, measure::BasisFunctions.Measure=measure(dict); warnslow=false,atol=1e-6,rtol=1e-6, opts...)
    isorthonormal(dict, measure) && (@test isorthogonal(dict, measure))
    @test isorthogonal(dict, measure) == isorthog
    @test isorthonormal(dict, measure) == isorthonorm

    G = gramoperator(dict, measure; warnslow=warnslow)
    @test eltype(G) == coefficienttype(dict)
    M = Matrix(BasisFunctions.default_gramoperator(dict, measure;warnslow=warnslow,atol=atol,rtol=rtol,opts...))

    if isorthog
        @test isdiag(G)
    else
        @test !isdiag(G)
        @test norm(M - Diagonal(diag(M))) > 1e-3 || any(abs.(diag(M)).< 1e-5)
    end
    if isorthonorm
        @test G isa IdentityOperator
    else
        @test !(G isa IdentityOperator)
        @test !(diag(M) ≈ Ones(length(dict)))
    end

    @test norm(M - Matrix(G)) < 1e-5
end
