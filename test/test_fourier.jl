# test_fourier.jl

#####
# Fourier series
#####
function test_fourier_series(T)

    ## Even length
    n = 12
    a = -T(1.2)
    b = T(3.4)
    fb = rescale(FourierBasis(n,T), a, b)
    @test ~isreal(fb)

    @test left(fb) ≈ a
    @test right(fb) ≈ b

    @test grid(fb) == PeriodicEquispacedGrid(n, a, b)

    # Take a random point in the domain
    x = T(a+rand()*(b-a))
    y = (x-a)/(b-a)

    # Is the 0-index basis function the constant 1?
    freq = 0
    idx = frequency2idx(set(fb), freq)
    @test fb[idx](x) ≈ 1

    # Evaluate in a point in the interior
    freq = 3
    idx = frequency2idx(set(fb), freq)
    @test fb[idx](x) ≈ exp(2*T(pi)*1im*freq*y)

    # Evaluate the largest frequency, which is a cosine in this case
    freq = n >> 1
    idx = frequency2idx(set(fb), freq)
    @test fb[idx](x) ≈ cos(2*T(pi)*freq*y)

    # Evaluate an expansion
    coef = T[1; 2; 3; 4] * (1+im)
    e = SetExpansion(rescale(FourierBasis(4,T), a, b), coef)
    @test e(x) ≈ coef[1]*T(1) + coef[2]*exp(2*T(pi)*im*y) + coef[3]*cos(4*T(pi)*y) + coef[4]*exp(-2*T(pi)*im*y)

    # Check type promotion: evaluate at an integer and at a rational point
    for i in [1 2]
        @test typeof(call_set(fb, i, 0)) == Complex{T}
        @test typeof(call_set(fb, i, 1//2)) == Complex{T}
    end

    # Try an extension
    n = 12
    coef = map(complexify(T), rand(n))
    b1 = rescale(FourierBasis(n,T), a, b)
    b2 = rescale(FourierBasis(n+1,T), a, b)
    b3 = rescale(FourierBasis(n+15,T), a, b)
    E2 = extension_operator(b1, b2)
    E3 = extension_operator(b1, b3)
    e1 = SetExpansion(b1, coef)
    e2 = SetExpansion(b2, E2*coef)
    e3 = SetExpansion(b3, E3*coef)
    x = T(2//10)
    @test e1(x) ≈ e2(x)
    @test e1(x) ≈ e3(x)


    # Differentiation test
    coef = map(complexify(T), rand(Float64, size(fb)))
    D = differentiation_operator(fb)
    coef2 = D*coef
    e1 = SetExpansion(fb, coef)
    e2 = SetExpansion(rescale(FourierBasis(length(fb)+1,T),left(fb),right(fb)), coef2)


    x = T(2//10)
    delta = sqrt(eps(T))
    @test abs( (e1(x+delta)-e1(x))/delta - e2(x) ) / abs(e2(x)) < 100delta



    ## Odd length
    fbo = rescale(FourierBasis(13,T), a, b)

    @test ~isreal(fbo)

    # Is the 0-index basis function the constant 1?
    freq = 0
    idx = frequency2idx(set(fbo), freq)
    @test fbo[idx](T(2//10)) ≈ 1

    # Evaluate in a point in the interior
    freq = 3
    idx = frequency2idx(set(fbo), freq)
    x = T(2//10)
    y = (x-a)/(b-a)
    @test fbo[idx](x) ≈ exp(2*T(pi)*1im*freq*y)

    # Evaluate an expansion
    coef = [one(T)+im; 2*one(T)-im; 3*one(T)+2im]
    e = SetExpansion(FourierBasis(3, a, b), coef)
    x = T(2//10)
    y = (x-a)/(b-a)
    @test e(x) ≈ coef[1]*one(T) + coef[2]*exp(2*T(pi)*im*y) + coef[3]*exp(-2*T(pi)*im*y)
    # evaluate on a grid
    g = grid(set(e))
    result = e(g)
    # Don't compare to zero with isapprox because the default absolute tolerance is zero.
    # So: add 1 and compare to 1
    @test sum([abs(result[i] - e(g[i])) for i in 1:length(g)]) + 1 ≈ 1

    # Try an extension
    n = 13
    coef = map(complexify(T), rand(n))
    b1 = FourierBasis(n, T)
    b2 = FourierBasis(n+1, T)
    b3 = FourierBasis(n+15, T)
    E2 = Extension(b1, b2)
    E3 = Extension(b1, b3)
    e1 = SetExpansion(b1, coef)
    e2 = SetExpansion(b2, E2*coef)
    e3 = SetExpansion(b3, E3*coef)
    x = T(2//10)
    @test e1(x) ≈ e2(x)
    @test e1(x) ≈ e3(x)

    # Restriction
    n = 14
    b1 = FourierBasis(n, T)
    b2 = FourierBasis(n-1, T)
    b3 = FourierBasis(n-5, T)
    E1 = Restriction(b1, b2)    # source has even length
    E2 = Restriction(b2, b3)    # source has odd length
    coef1 = map(complexify(T), rand(length(b1)))
    coef2 = E1*coef1
    coef3 = E2*coef2
    @test reduce(&, [ coef2[i+1] == coef1[i+1] for i=0:BasisFunctions.nhalf(b2) ] )
    @test reduce(&, [ coef2[end-i+1] == coef1[end-i+1] for i=1:BasisFunctions.nhalf(b2) ] )
    @test reduce(&, [ coef3[i+1] == coef2[i+1] for i=0:BasisFunctions.nhalf(b3) ] )
    @test reduce(&, [ coef3[end-i+1] == coef2[end-i+1] for i=1:BasisFunctions.nhalf(b3) ] )

    # Differentiation test
    coef = map(complexify(T), rand(Float64, size(fbo)))
    D = differentiation_operator(fbo)
    coef2 = D*coef
    e1 = SetExpansion(fbo, coef)
    e2 = SetExpansion(fbo, coef2)

    x = T(2//10)
    delta = sqrt(eps(T))
    @test abs( (e1(x+delta)-e1(x))/delta - e2(x) ) / abs(e2(x)) < 150delta

    # Transforms
    b1 = FourierBasis(161, T)
    A = approximation_operator(b1)
    f = x -> 1/(2+cos(pi*x))
    e = approximate(b1, f)
    x0 = T(1//2)
    @test abs(e(T(x0))-f(x0)) < sqrt(eps(T))
end
