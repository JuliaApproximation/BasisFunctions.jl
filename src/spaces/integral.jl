# For mapped measures, we can undo the map and leave out the jacobian in the
# weight function of the measure by going to the supermeasure

function DomainIntegrals.quadrature_m(qs, integrand, domain::DomainSets.MappedDomain, μ::MappedMeasure, sing)
    # TODO: think about what to do with the singularity here
    # -> Move MappedMeasure into DomainIntegrals.jl and implement the logic there
    m1 = mapping(μ)
    m2 = mapping(domain)
    if iscompatible(m1,m2)
        integrand2 = x -> integrand(applymap(m, x))
        DomainIntegrals.quadrature(qs, integrand2, superdomain(domain), supermeasure(μ), sing)
    else
        DomainIntegrals.quadrature_d(qs, integrand, domain, μ, sing)
    end
end

function DomainIntegrals.quadrature_m(qs, integrand, domain, μ::MappedMeasure, sing)
    m = mapping(μ)
    integrand2 = x -> integrand(applymap(m, x))
    DomainIntegrals.quadrature(qs, integrand2, inv(m)*domain, supermeasure(μ), sing)
end
