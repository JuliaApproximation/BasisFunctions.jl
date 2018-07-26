
function test_mapped_dicts()
    b = FourierBasis(20)
    x1 = 1.0+im
    x2 = 2.0+3im
    m1 = interval_map(0.0, 1.0, x1, x2)
    m2 = embedding_map(Float64, Complex{Float64})
    m = m1 ∘ m2
    c = mapped_dict(b, m)

    @test !in_support(c, 10, 0.5)
    @test in_support(c, 10, x1)
    @test in_support(c, 10, (x1+x2)/2)

    u = approximate(c, exp)
    z = 1/3*x1+2/3*x2
    @test abs(u(z)-exp(z)) < 0.2

    d = circle()
    m = parameterization(d)
    b = FourierBasis(20)
    c = mapped_dict(b, m)
    f1(x,y) = exp(x+im*y)
    u = approximate(c, f1)
    @test in_support(c, 1, SVector(1.0, 0.0))
    @test abs(u(SVector(cos(10),sin(10)))-f1(cos(10),sin(10))) < 1e-6
end
