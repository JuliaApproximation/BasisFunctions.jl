# Routines for evaluating integrals

# function integral(f, domain::AbstractInterval; options...)
#     if isempty(domain)
#         return zero(f(zero(eltype(domain))))
#     end
#     numerical_integral(f, infimum(domain), supremum(domain); options...)
# end

# integral(f, domain::UnionDomain; options...) =
#     sum([integral(f, d; options...) for d in elements(domain)])

# integral(f, domain::DomainSets.FullSpace{T}; options...) where {T <: Real} =
#     numerical_integral(f, -convert(T, Inf), convert(T, Inf); options...)

function numerical_integral(f, a::T, b::T; atol = 0, rtol = sqrt(eps(T)), verbose=false, numquad = false, overquad = 2, options...) where {T}
    if numquad
        x, w = options[:quadrule]
        sum(w .* f.(x))
    else
        if overquad == 2
            nodes = (a,b)
        else
            nodes = LinRange(a,b,overquad)
        end
        I,e = QuadGK.quadgk(f, nodes...; rtol=rtol, atol=atol)
        if verbose && (e > sqrt(rtol))
            @warn "Numerical evaluation of integral did not converge"
        end
        I
    end
end

# integral(f, domain::Domain; options...) =
#     error("Don't know how to compute an integral on domain: $(domain).")

# weighted_integral(f, weight, domain::Domain; options...) =
#     integral(x -> weight(x)*f(x), domain; options...)

# integral(f, measure::Measure; options...) =
#     weighted_integral(f, unsafe_weightfunction(measure), support(measure); options...)

# integral(f, measure::LebesgueMeasure; options...) =
#     integral(f, support(measure); options...)

# integral(f, measure::DiracMeasure; options...) =
#     f(point(measure))

function integral(f, measure::DiscreteMeasure; options...)
    @warn "This routine is deprecated"
    T = prectype(measure)
    r = zero(T)
    # For some reason, the code below allocates memory:
    # for (xi,x) in enumerate(grid(measure))
    #     r += unsafe_discrete_weight(measure,xi)*f(x)
    # end
    # So we do it more explicitly:
    w = weights(measure)
    g = grid(measure)
    for i in 1:length(g)
        r += w[i] * f(g[i])
    end
    r
end

# function integral(f, measure::DiscreteMeasure, domain::Domain; options...)
#     T = prectype(measure)
#     r = zero(T)
#     w = weights(measure)
#     g = grid(measure)
#     for i in 1:length(g)
#         if g[i] ∈ domain
#             r += w[i] * f(g[i])
#         end
#     end
#     r
# end

# ChebyshevT: apply cosine map to the integral.
# Weight function times Jacobian becomes identity.
integral(f, measure::ChebyshevTMeasure{T}) where {T} =
    convert(T,pi)*integral(x->f(cos(convert(T,pi)*x)), UnitInterval{T}())

# ChebyshevU: apply cosine map to the integral.
# Weight function and Jacobian are both equal to sin(pi*x).
integral(f, measure::ChebyshevUMeasure{T}) where {T} =
    convert(T,pi)*integral(x->f(cos(convert(T,pi)*x))*sin(convert(T,pi)*x)^2, UnitInterval{T}())

# For mapped measures, we can undo the map and leave out the jacobian in the
# weight function of the measure by going to the supermeasure
integral(f, measure::MappedMeasure) =
    integral(x->f(applymap(mapping(measure),x)), supermeasure(measure))

innerproduct(f, g, measure; options...) = integral(x->conj(f(x))*g(x), measure)
