# Routines for evaluating integrals

integral(f, domain::AbstractInterval; options...) =
    numerical_integral(f, leftendpoint(domain), rightendpoint(domain); options...)

integral(f, domain::DomainSets.FullSpace{T}; options...) where {T <: Real} =
    numerical_integral(f, -convert(T, Inf), convert(T, Inf); options...)

function numerical_integral(f, a::T, b::T; atol = 0, rtol = sqrt(eps(T)), verbose=false, options...) where {T}
    I,e = QuadGK.quadgk(f, a, b; rtol=rtol, atol=atol)
    if verbose && (e > sqrt(rtol))
        @warn "Numerical evaluation of integral did not converge"
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

# TODO DiracCombMeasure

# ChebyshevT: apply cosine map to the integral.
# Weight function times Jacobian becomes identity.
integral(f, measure::ChebyshevTMeasure{T}; options...) where {T} =
    convert(T,pi)*integral(x->f(cos(convert(T,pi)*x)), UnitInterval{T}(); options...)

# ChebyshevU: apply cosine map to the integral.
# Weight function and Jacobian are both equal to sin(pi*x).
integral(f, measure::ChebyshevUMeasure{T}; options...) where {T} =
    convert(T,pi)*integral(x->f(cos(convert(T,pi)*x))*sin(convert(T,pi)*x)^2, UnitInterval{T}(); options...)

# For mapped measures, we can undo the map and leave out the jacobian in the
# weight function of the measure by going to the supermeasure
integral(f, measure::MappedMeasure; options...) =
    integral(x->f(applymap(mapping(measure),x)), supermeasure(measure); options...)

innerproduct(f, g, measure; options...) = integral(x->conj(f(x))*g(x), measure; options...)
