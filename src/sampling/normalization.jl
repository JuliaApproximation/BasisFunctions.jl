
# Routines having to do with the normalizing function values on a grid with respect
# to a given measure.

"""
Return a normalization for a sampling operator on a grid. The normalization ensures
a norm equivalence between the continuous norm of the function space associated
with the measure, and the discrete norm of the sampling vector.

In practice, this amounts to a diagonal scaling with the square roots of the
weights of a quadrature rule on the grid that is exact on the span of the dictionary
in the space.
"""
sampling_normalization(gb::GridBasis{T}, measure) where T =
	DiagonalOperator(gb, gb, sqrt.(quadweights(grid(gb), measure)))
