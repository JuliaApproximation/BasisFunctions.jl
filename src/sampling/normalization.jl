
# Routines having to do with the normalizing function values on a grid with respect
# to a given measure.

"""
Compute an operator that approximates a discretization of the given measure on the
given sampling grid.

It is an approximation in the following sense. Assume a function `f` is given
with integral `∫f dμ`. If `S` is the sampling operator on the grid, and if
`N` is the associated quadraturenormalization (with respect to `dμ`), then:
`∫f dμ ≈ sum(N * S(f))`.
"""
quadraturenormalization(op::GridSampling, measure) =
	quadraturenormalization(dest(op), measure)

quadraturenormalization(gb::GridBasis, measure) =
	quadraturenormalization(gb, grid(gb), measure)

quadraturenormalization(gb, grid::PeriodicEquispacedGrid, ::LebesgueMeasure) =
	ScalingOperator(gb, stepsize(grid))

quadraturenormalization(gb, grid::MidpointEquispacedGrid, ::LebesgueMeasure) =
	ScalingOperator(gb, stepsize(grid))

quadraturenormalization(gb, grid::FourierGrid, ::LebesgueMeasure) =
	ScalingOperator(gb, stepsize(grid))

quadraturenormalization(gb::GridBasis, grid::ChebyshevNodes, ::ChebyshevTMeasure) =
	ScalingOperator(gb, convert(coefficienttype(gb), pi)/length(grid))

function quadraturenormalization(gb, grid::ChebyshevNodes, ::LegendreMeasure)
	T = coefficienttype(gb)
	x, w = fejer_first_rule(length(grid), real(T))
	DiagonalOperator(gb, w)
end

function quadraturenormalization(gb, grid::ChebyshevExtremae, ::LegendreMeasure)
	T = coefficienttype(gb)
	x, w = clenshaw_curtis(length(grid)-1, real(T))
	DiagonalOperator(gb, w)
end

function quadraturenormalization(gb, grid::MappedGrid, measure::MappedMeasure)
	@assert iscompatible(mapping(grid), mapping(measure))
	T = coefficienttype(gb)
	sgrid = supergrid(grid)
	sgb = GridBasis{T}(sgrid)
	wrap_operator(gb, gb, quadraturenormalization(sgb, sgrid, supermeasure(measure)))
end

quadraturenormalization(gb, grid::ProductGrid, measure::ProductMeasure) =
	tensorproduct( map( (g,m) -> quadraturenormalization(g, m), elements(gb), elements(measure))...)

function quadraturenormalization(gb, grid::AbstractSubGrid, measure::SubMeasure)
	T = coefficienttype(gb)
	sgrid = supergrid(grid)
	sgb = GridBasis{T}(sgrid)

	O = quadraturenormalization(sgb, supermeasure(measure))
	if isa(O, ScalingOperator)
		# If the operator of the supergrid is a ScalingOperator, we can take
		# its value and make a ScalingOperator for the subgrid
		val = scalar(O)
		ScalingOperator(gb, val)
	else
		# Else we have to do it manually. This may be suboptimal, because the
		# composite operator constructed below will often be diagonal.
		T = coefficienttype(gb)
		sgb = GridBasis{T}(sgrid)
		restriction_operator(sgb, gb) * O * extension_operator(gb, sgb)
	end
end


"Return a diagonal operator with a weight function of a measure evaluated on a grid."
weightoperator(gb::GridBasis, measure::Measure) = weightoperator(gb, grid(gb), measure)

# For any Lebesgue measure this will be the identity operator
weightoperator(gb::GridBasis, grid::AbstractGrid, measure::LebesgueMeasure) = IdentityOperator(gb)

function weightoperator(gb::GridBasis, grid::AbstractGrid, measure::Measure)
	diag = zeros(gb)
	for i in eachindex(grid)
		diag[i] = unsafe_weight(measure, grid[i])
	end
	DiagonalOperator(gb, diag)
end


quadraturenormalization(gb, grid, measure) = riemann_normalization(gb, grid, measure)

# Fall back in 1d. We don't know any special properties of the grid: we use a
# Riemann sum and multiply by the weight function of the measure on the grid.
function riemann_normalization(gb, grid::AbstractGrid1d, measure)
	a = infimum(support(measure))
	b = supremum(support(measure))
	riemannsum_normalization(grid, a, b; T=coefficienttype(gb)) *  weightoperator(gb, measure)
end


function riemannsum_normalization(grid::AbstractGrid, a = leftendpoint(grid), b = rightendpoint(grid); T)
	M = length(grid)
	L = b-a
	t = [grid[end] - L; collect(grid); grid[1]+L]
	weights = zeros(T, M)
	for m in 1:M
		weights[m] = sqrt( (t[m+2]-t[m]) / 2)
	end
	DiagonalOperator(GridBasis{T}(grid), weights)
end
