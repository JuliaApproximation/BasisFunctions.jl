
"Add a support and measure to a dictionary."
struct DomainSupportDict{S,T} <: HighlySimilarDerivedDict{S,T}
    support     ::  Domain{S}
    superdict   ::  Dictionary{S,T}
end

support(Φ::DomainSupportDict) = Φ.support

modifiersymbol(Φ::DomainSupportDict) = "supported"

hasmeasure(Φ::DomainSupportDict) = true
measure(Φ::DomainSupportDict) = lebesguemeasure(support(Φ))

plotgrid(Φ::DomainSupportDict, n) = plotgrid_domain(n, support(Φ))
