
gaussweights(dmeasure::DiscreteMeasure, measure::Measure) =
    gaussweights(grid(dmeasure), weights(dmeasure), measure)

gaussweights(grid::AbstractGrid, weights, measure::Measure) =
    gaussweights(grid, measure) ./ weights

gaussweights(grid::AbstractGrid, ::Ones, measure::Measure) =
    gaussweights(grid, measure)

gaussweights(grid::AbstractSubGrid, measure::SubMeasure) =
    gaussweights(supergrid(grid), supermeasure(measure))[subindices(grid)]

function gaussweights(grid::MappedGrid, measure::MappedMeasure; options...)
    @assert iscompatible(mapping(grid), mapping(measure))
    sgrid = supergrid(grid)
    gaussweights(sgrid, supermeasure(measure); options...)
end

gaussweights(grid::ProductGrid, measure::ProductMeasure) =
    OuterProductArray(map(gaussweights, elements(grid), elements(measure))...)

gaussweights(grid::ChebyshevTNodes{T}, ::ChebyshevTMeasure{T}) where {T} =
    ChebyshevTWeights{T}(length(grid))

gaussweights(grid::ChebyshevUNodes{T}, ::ChebyshevUMeasure{T}) where {T} =
    ChebyshevUWeights{T}(length(grid))

gaussweights(grid::LegendreNodes{T}, ::LegendreMeasure{T}) where {T} =
    weights(LegendreGaussMeasure{T}(length(grid)))

gaussweights(grid::LaguerreNodes{T}, measure::LaguerreMeasure{T}) where {T} =
    (@assert grid.α ≈ measure.α; weights(LaguerreGaussMeasure(length(grid),grid.α)))

gaussweights(grid::HermiteNodes{T}, ::HermiteMeasure{T}) where {T} =
    weights(HermiteGaussMeasure{T}(length(grid)))

gaussweights(grid::JacobiNodes{T}, measure::JacobiMeasure{T}) where {T} =
    (@assert (grid.α ≈ measure.α) & (grid.β ≈ measure.β); weights(JacobiGaussMeasure(length(grid), grid.α, grid.β)))

gaussweights(grid::FourierGrid{T}, measure::FourierMeasure{T}) where {T} =
    ProbabilityArray{T}(length(grid))

gaussweights(grid::PeriodicEquispacedGrid{T}, measure::FourierMeasure{T}) where {T} =
    (@assert support(grid)≈support(measure);ProbabilityArray{T}(length(grid)))

gaussweights(grid::MidpointEquispacedGrid{T}, measure::FourierMeasure{T}) where {T} =
    (@assert support(grid)≈support(measure);ProbabilityArray{T}(length(grid)))

gaussweights(grid::PeriodicEquispacedGrid{T}, measure::LebesgueMeasure{T}) where {T} =
    (@assert support(grid)≈support(measure); duration(support(measure))*ProbabilityArray{T}(length(grid)))

gaussweights(grid::MidpointEquispacedGrid{T}, measure::LebesgueMeasure{T}) where {T} =
    (@assert support(grid)≈support(measure); duration(support(measure))*ProbabilityArray{T}(length(grid)))

function gaussweights(grid::ChebyshevNodes{T}, ::LegendreMeasure{T}) where {T}
	x, w = fejer_first_rule(length(grid), T)
	w
end

function gaussweights(grid::ChebyshevExtremae{T}, ::LegendreMeasure{T}) where {T}
	x, w = clenshaw_curtis(length(grid)-1, T)
	w
end

gaussweights(grid::ChebyshevNodes{T}, measure::LebesgueMeasure{T}) where {T} =
    (@assert ≈(map(support, (grid,measure))); gaussweights(grid, LegendreMeasure{T}()))

gaussweights(grid::ChebyshevExtremae{T}, measure::LebesgueMeasure{T}) where {T} =
    (@assert ≈(map(support, (grid,measure))); gaussweights(grid, LegendreMeasure{T}()))

gaussweights(grid::AbstractGrid, measure::Measure; options...) =
	default_quadratureweights(grid, measure; options...)

function default_quadratureweights(grid, measure; options...)
	@debug "No known quadrature normalization available for grid $(typeof(grid)) with measure $(typeof(measure)) \n choosing Riemann sum normalization"
	riemann_normalization(grid, measure; options...)
end

# Fall back in 1d. We don't know any special properties of the grid: we use a
# Riemann sum and multiply by the weight function of the measure on the grid.
function riemann_normalization(grid::AbstractGrid1d, measure; T=prectype(grid), options...)
	a = infimum(support(measure))
	b = supremum(support(measure))
	riemannsum_weights(grid, a, b; T=T) .*  measuresampling(grid, measure)
end

function riemannsum_weights(grid::AbstractGrid, a = leftendpoint(support(grid)), b = rightendpoint(support(grid)); T)
	M = length(grid)
	L = b-a
	t = [grid[end] - L; collect(grid); grid[1]+L]
	weights = zeros(T, M)
	for m in 1:M
		weights[m] = (t[m+2]-t[m]) / 2
	end
	weights
end


# For any Lebesgue measure this will be the identity operator
measuresampling(grid::AbstractGrid, measure::LebesgueMeasure; T=prectype(grid), opts...) =
	Ones{T}(size(grid))

"""
	measuresampling(grid::AbstractGrid, ::Array, measure::Measure)

Return an array with a weight function of a measure evaluated on a grid.
"""
measuresampling(grid::AbstractGrid, measure::Measure; T=prectype(grid)) =
    [convert(T,unsafe_weight(measure, xi)) for xi in grid]
