
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
quadraturenormalization(S::GridSampling, dμ; options...) =
	quadraturenormalization(dest(S), dμ; options...)

"""
Return a normalization for a sampling operator on a grid. The normalization ensures
a norm equivalence between the continuous norm of the function space associated
with the measure, and the discrete norm of the sampling vector.

In practice, this amounts to a diagonal scaling with the square roots of the
weights of a quadrature rule on the grid that is exact on the span of the dictionary
in the space.
"""
sampling_normalization(gb::GridBasis, discretemeasure, measure; options...) =
	DiagonalOperator(gb, gb, sqrt.(gaussweights(discretemeasure, measure)))

quadraturenormalization(discretemeasure::DiscreteMeasure, measure::Measure; options...) =
	Diagonal(gaussweights(discretemeasure, measure))

# quadraturenormalization(grid::ProductGrid, weights, measure::Union{ProductMeasure,DiscreteTensorSubMeasure}; options...) =
# 	tensorproduct( map( (g,w,m) -> quadraturenormalization(g, w, m; options...), elements(grid), elements(weights), elements(measure))...)
#
# function quadraturenormalization(discretemeasure::DiscreteSubMeasure, measure::SubMeasure; T=prectype(grid(discretemeasure)), options...)
# 	sdmeasure = supermeasure(discretemeasure)
# 	smeasure = supermeasure(measure)
#
# 	O = quadraturenormalization(sdmeasure, smeasure; T=T, options...)
# 	sgb = GridBasis{T}(grid(sdmeasure))
# 	if O isa ScalingOperator
# 		# If the operator of the supergrid is a ScalingOperator, we can take
# 		# its value and make a ScalingOperator for the subgrid
# 		val = scalar(O)
# 		ScalingOperator(sgb, convert(T,val))
# 	else
# 		# Else we have to do it manually. This may be suboptimal, because the
# 		# composite operator constructed below will often be diagonal.
# 		gb = GridBasis{T}(grid(discretemeasure))
# 		restriction_operator(sgb, gb; options...) * O * extension_operator(gb, sgb; options...)
# 	end
# end
#
# function quadraturenormalization(grid::AbstractSubGrid, weights, measure::LebesgueMeasure;options...)
# 	submeasure = SubMeasure(BasisFunctions.GenericLebesgueMeasure(support(supergrid(grid))),support(measure))
# 	quadraturenormalization(grid, weights, submeasure;options...)
# end
