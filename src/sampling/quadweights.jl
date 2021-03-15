
@deprecate gaussweights quadweights

quadweights(dmeasure::DiscreteWeight, measure::Weight) =
    quadweights(points(dmeasure), weights(dmeasure), measure)

quadweights(grid::AbstractGrid, weights, measure::Weight) =
    quadweights(grid, measure) ./ weights

quadweights(grid::AbstractGrid, ::Ones, measure::Weight) =
    quadweights(grid, measure)

function quadweights(grid::MappedGrid, measure::MappedWeight)
    @assert iscompatible(mapping(grid), mapping(measure))
    sgrid = supergrid(grid)
    quadweights(sgrid, supermeasure(measure))
end

quadweights(grid::ProductGrid, measure::ProductWeight) =
    OuterProductArray(map(quadweights, elements(grid), elements(measure))...)

quadweights(grid::ChebyshevTNodes{T}, ::ChebyshevTWeight{T}) where {T} =
    GridArrays.ChebyshevTWeights{T}(length(grid))

quadweights(grid::ChebyshevUNodes{T}, ::ChebyshevUWeight{T}) where {T} =
    GridArrays.ChebyshevUWeights{T}(length(grid))

quadweights(grid::LegendreNodes{T}, ::LegendreWeight{T}) where {T} =
    weights(GaussLegendre{T}(length(grid)))

function quadweights(grid::LaguerreNodes{T}, measure::LaguerreWeight{T}) where {T}
	@assert grid.α ≈ measure.α
    weights(GaussLaguerre(length(grid),grid.α))
end

quadweights(grid::HermiteNodes{T}, ::HermiteWeight{T}) where {T} =
    weights(GaussHermite{T}(length(grid)))

function quadweights(grid::JacobiNodes{T}, measure::JacobiWeight{T}) where {T}
    @assert (grid.α ≈ measure.α) & (grid.β ≈ measure.β)
	weights(GaussJacobi(length(grid), grid.α, grid.β))
end

quadweights(grid::FourierGrid{T}, measure::FourierWeight{T}) where {T} =
    NormalizedArray{T}(length(grid))

function quadweights(grid::PeriodicEquispacedGrid{T}, measure::FourierWeight{T}) where {T}
    @assert coverdomain(grid)≈support(measure)
	NormalizedArray{T}(length(grid))
end

function quadweights(grid::MidpointEquispacedGrid{T}, measure::FourierWeight{T}) where {T}
    @assert coverdomain(grid)≈support(measure)
	NormalizedArray{T}(length(grid))
end

function quadweights(grid::PeriodicEquispacedGrid{T}, measure::LebesgueMeasure{T}) where {T}
    @assert coverdomain(grid)≈support(measure)
	DomainSets.width(support(measure))*NormalizedArray{T}(length(grid))
end

function quadweights(grid::MidpointEquispacedGrid{T}, measure::LebesgueMeasure{T}) where {T}
    @assert coverdomain(grid)≈support(measure)
	DomainSets.width(support(measure))*NormalizedArray{T}(length(grid))
end

function quadweights(grid::ChebyshevNodes{T}, ::LegendreWeight{T}) where {T}
	x, w = fejer_first_rule(length(grid), T)
	w
end

function quadweights(grid::ChebyshevExtremae{T}, ::LegendreWeight{T}) where {T}
	x, w = clenshaw_curtis(length(grid)-1, T)
	w
end

function quadweights(grid::ChebyshevNodes{T}, measure::LebesgueMeasure{T}) where {T}
    @assert ≈(map(support, (grid,measure))...)
	quadweights(grid, LegendreWeight{T}())
end

function quadweights(grid::ChebyshevExtremae{T}, measure::LebesgueMeasure{T}) where {T}
    @assert ≈(map(support, (grid,measure))...)
	quadweights(grid, LegendreWeight{T}())
end

quadweights(grid::AbstractGrid, measure::Weight) =
	default_quadratureweights(grid, measure)

function default_quadratureweights(grid, measure)
	@debug "No known quadrature normalization available for grid $(typeof(grid)) with measure $(typeof(measure)) \n choosing Riemann sum normalization"
	riemann_normalization(grid, measure)
end

# Fall back in 1d. We don't know any special properties of the grid: we use a
# Riemann sum and multiply by the weight function of the measure on the grid.
function riemann_normalization(grid::AbstractGrid1d, measure)
	a = infimum(support(measure))
	b = supremum(support(measure))
	riemannsum_weights(grid, a, b) .*  measuresampling(grid, measure)
end

function riemannsum_weights(grid::AbstractGrid, a = leftendpoint(coverdomain(grid)), b = rightendpoint(coverdomain(grid)))
	M = length(grid)
	L = b-a
	t = [grid[end] - L; collect(grid); grid[1]+L]
	T = prectype(grid)
	weights = zeros(T, M)
	for m in 1:M
		weights[m] = (t[m+2]-t[m]) / 2
	end
	weights
end


# For any Lebesgue measure this will be the identity operator
measuresampling(grid::AbstractGrid, measure::LebesgueMeasure) =
	Ones{prectype(grid)}(size(grid))

"""
	measuresampling(grid::AbstractGrid, ::Array, measure::Weight)

Return an array with a weight function of a measure evaluated on a grid.
"""
measuresampling(grid::AbstractGrid, measure::Weight) =
    [unsafe_weightfun(measure, xi) for xi in grid]
