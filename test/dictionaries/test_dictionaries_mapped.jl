
function test_mapped_dicts()
    d = UnitCircle()
    m = parameterization(d)
    b = Fourier(20)
    c = mapped_dict(b, m)
    f1(x,y) = exp(x+im*y)
    u = approximate(c, f1)
    @test in_support(c, 1, SVector(1.0, 0.0))
    @test abs(u(SVector(cos(10),sin(10)))-f1(cos(10),sin(10))) < 1e-6
end
