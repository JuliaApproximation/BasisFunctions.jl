# integral.jl


function integral(f, domain::AbstractInterval{T}; atol = 0, rtol = sqrt(eps(T)), verbose=false, options...) where T
    I,e = QuadGK.quadgk(f, leftendpoint(domain), rightendpoint(domain); rtol=rtol, atol=atol)
    if verbose && (e > sqrt(rtol))
        warn("Numerical evaluation of integral did not converge")
    end
    I
end

integral(f, domain::Domain; options...) =
    error("Don't know how to compute an integral on domain: $(domain).")

weighted_integral(f, weight, domain::Domain; options...) =
    integral(x -> weight(x)*f(x), domain; options...)

integral(f, measure::Measure; options...) =
    weighted_integral(f, unsafe_weightfunction(measure), support(measure); options...)

integral(f, measure::LebesgueMeasure; options...) =
    integral(f, support(measure); options...)

integral(f, measure::DiracMeasure; options...) =
    f(point(measure))

innerproduct(f, g, m::Measure; options...) = integral(x->conj(f(x))*g(x), m; options...)

innerproduct(f, g, d::Domain; options...) = integral(x->conj(f(x))*g(x), d; options...)
