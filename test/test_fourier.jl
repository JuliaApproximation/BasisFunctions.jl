# test_fourier.jl
using BasisFunctions, BasisFunctions.Test, DomainSets

using Test
types = (Float64, BigFloat)

#####
# Fourier series
#####
function test_fourier_series(T)

    ### Test bounds checking
    fb0 = FourierBasis{T}(5)
    @test ~in_support(fb0, 1, -one(T)/10)
    @test in_support(fb0, 1, zero(T))
    @test in_support(fb0, 1, one(T)/2)
    @test in_support(fb0, 1, one(T))
    @test in_support(fb0, 1, zero(T)-0.9*sqrt(eps(T)))
    @test in_support(fb0, 1, one(T)+0.9*sqrt(eps(T)))

    @test support(fb0) == UnitInterval{T}()

    ## Even length
    n = 12
    a = -T(1.2)
    b = T(3.4)
    fb = rescale(FourierBasis{T}(n), a, b)
    @test ~isreal(fb)

    @test infimum(support(fb)) ≈ a
    @test supremum(support(fb)) ≈ b

    g = grid(fb)
    @test typeof(g) <: PeriodicEquispacedGrid
    @test leftendpoint(g) ≈ a
    @test rightendpoint(g) ≈ b
    @test length(g) == length(fb)

    # Take a random point in the domain
    x = T(a+rand()*(b-a))
    y = (x-a)/(b-a)

    # Is the 0-index basis function the constant 1?
    freq = 0
    idx = frequency2idx(superdict(fb), freq)
    @test fb[idx](x) ≈ 1

    # Evaluate in a point in the interior
    freq = 3
    idx = frequency2idx(superdict(fb), freq)
    @test fb[idx](x) ≈ exp(2*T(pi)*1im*freq*y)

    # Evaluate the largest frequency, which is a cosine in this case
    freq = n >> 1
    idx = frequency2idx(superdict(fb), freq)
    @test fb[idx](x) ≈ cos(2*T(pi)*freq*y)

    # Evaluate an expansion
    coef = T[1; 2; 3; 4] * (1+im)
    e = Expansion(rescale(FourierBasis{T}(4), a, b), coef)
    @test e(x) ≈ coef[1]*T(1) + coef[2]*exp(2*T(pi)*im*y) + coef[3]*cos(4*T(pi)*y) + coef[4]*exp(-2*T(pi)*im*y)

    # Check type promotion: evaluate at an integer and at a rational point
    for i in [1 2]
        @test typeof(BasisFunctions.unsafe_eval_element(fb, i, 0)) == Complex{T}
        @test typeof(BasisFunctions.unsafe_eval_element(fb, i, 1//2)) == Complex{T}
    end

    # Try an extension
    n = 12
    coef = map(complex(T), rand(n))
    b1 = rescale(FourierBasis{T}(n), a, b)
    b2 = rescale(FourierBasis{T}(n+1), a, b)
    b3 = rescale(FourierBasis{T}(n+15), a, b)
    E2 = extension_operator(b1, b2)
    E3 = extension_operator(b1, b3)
    e1 = Expansion(b1, coef)
    e2 = Expansion(b2, E2*coef)
    e3 = Expansion(b3, E3*coef)
    x = T(2//10)
    @test e1(x) ≈ e2(x)
    @test e1(x) ≈ e3(x)


    # Differentiation test
    coef = map(complex(T), rand(Float64, size(fb)))
    D = differentiation_operator(fb)
    coef2 = D*coef
    e1 = Expansion(fb, coef)
    e2 = Expansion(rescale(FourierBasis{T}(length(fb)+1),support(fb)), coef2)

    zero_orderD = pseudodifferential_operator(fb,x->1)
    pseudoD = pseudodifferential_operator(fb,x->x^2+x)

    x = T(2//10)
    delta = sqrt(eps(T))
    @test abs( (e1(x+delta)-e1(x))/delta - e2(x) ) / abs(e2(x)) < 100delta



    ## Odd length
    fbo = rescale(FourierBasis{T}(13), a, b)

    @test ~isreal(fbo)

    # Is the 0-index basis function the constant 1?
    freq = 0
    idx = frequency2idx(superdict(fbo), freq)
    @test fbo[idx](T(2//10)) ≈ 1

    # Evaluate in a point in the interior
    freq = 3
    idx = frequency2idx(superdict(fbo), freq)
    x = T(2//10)
    y = (x-a)/(b-a)
    @test fbo[idx](x) ≈ exp(2*T(pi)*1im*freq*y)

    # Evaluate an expansion
    coef = [one(T)+im; 2*one(T)-im; 3*one(T)+2im]
    e = Expansion(FourierBasis{T}(3, a, b), coef)
    x = T(2//10)
    y = (x-a)/(b-a)
    @test e(x) ≈ coef[1]*one(T) + coef[2]*exp(2*T(pi)*im*y) + coef[3]*exp(-2*T(pi)*im*y)
    # evaluate on a grid
    g = grid(dictionary(e))
    result = e(g)
    # Don't compare to zero with isapprox because the default absolute tolerance is zero.
    # So: add 1 and compare to 1
    @test sum([abs(result[i] - e(g[i])) for i in 1:length(g)]) + 1 ≈ 1

    # Try an extension
    n = 13
    coef = map(complex(T), rand(n))
    b1 = FourierBasis{T}(n)
    b2 = FourierBasis{T}(n+1)
    b3 = FourierBasis{T}(n+15)
    E2 = extension_operator(b1, b2)
    E3 = extension_operator(b1, b3)
    e1 = Expansion(b1, coef)
    e2 = Expansion(b2, E2*coef)
    e3 = Expansion(b3, E3*coef)
    x = T(2//10)
    @test e1(x) ≈ e2(x)
    @test e1(x) ≈ e3(x)

    # Restriction
    n = 14
    b1 = FourierBasis{T}(n)
    b2 = FourierBasis{T}(n-1)
    b3 = FourierBasis{T}(n-5)
    E1 = restriction_operator(b1, b2)    # source has even length
    E2 = restriction_operator(b2, b3)    # source has odd length
    coef1 = map(complex(T), rand(length(b1)))
    coef2 = E1*coef1
    coef3 = E2*coef2
    @test reduce(&, [ coef2[i+1] == coef1[i+1] for i=0:BasisFunctions.nhalf(b2) ] )
    @test reduce(&, [ coef2[end-i+1] == coef1[end-i+1] for i=1:BasisFunctions.nhalf(b2) ] )
    @test reduce(&, [ coef3[i+1] == coef2[i+1] for i=0:BasisFunctions.nhalf(b3) ] )
    @test reduce(&, [ coef3[end-i+1] == coef2[end-i+1] for i=1:BasisFunctions.nhalf(b3) ] )

    # Differentiation test
    coef = map(complex(T), rand(Float64, size(fbo)))
    D = differentiation_operator(fbo)
    coef2 = D*coef
    e1 = Expansion(fbo, coef)
    e2 = Expansion(fbo, coef2)

    x = T(2//10)
    delta = sqrt(eps(T))
    @test abs( (e1(x+delta)-e1(x))/delta - e2(x) ) / abs(e2(x)) < 150delta

    # Transforms
    b1 = FourierBasis{T}(161)
    A = approximation_operator(b1)
    f = x -> 1/(2+cos(2*T(pi)*x))
    e = approximate(b1, f)
    x0 = T(1//2)
    @test abs(e(T(x0))-f(x0)) < sqrt(eps(T))

    # Arithmetic

    b2 = FourierBasis{T}(162)
    f2 = x -> 1/(2+cos(2*T(pi)*x))
    e2 = approximate(b2, f2)
    x0 = T(1//2)
    @test abs((e*e2)(T(x0))-f(x0)*f2(x0)) < sqrt(eps(T))
    @test abs((e+2*e2)(T(x0))-(f(x0)+2*f2(x0))) < sqrt(eps(T))
    @test abs((3*e-e2)(T(x0))-(3*f(x0)-f2(x0))) < sqrt(eps(T))

    # Discrete Gram
    b = FourierBasis{T}(11)

    G = DiscreteGram(b)
    DG = DiscreteDualGram(b)
    MG = DiscreteMixedGram(b)

    e = coefficients(random_expansion(b))
    @test G*e ≈ e
    @test DG*e ≈ e
    @test MG*e ≈ e

    G = Gram(b)
    DG = DualGram(b)
    MG = MixedGram(b)

    e = coefficients(random_expansion(b))
    @test G*e ≈ e
    @test DG*e ≈ e
    @test MG*e ≈ e
end

for T in types
    @testset "$(rpad("Fourier expansions ($T)",80))" begin
        println("Fourier expansions ($T):")
        test_fourier_series(T)
    end
    println()
end
