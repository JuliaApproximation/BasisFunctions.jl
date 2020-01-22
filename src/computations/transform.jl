
#####################
# Generic transforms
#####################

# A function set can have several associated transforms. The default transform is
# associated with the grid of the set, e.g. the FFT and the DCTII for Chebyshev expansions
# which convert between coefficient space and value space. In this case, the
# transform maps coefficients to or from a GridBasis.
#
# transform_operator takes two arguments, a source and destination set, in order
# to allow for different transforms.
#
# An important case is where the source or destination is a GridBasis. In that case,
# the routines transform_from_grid and transform_to_grid are invoked,
# with the grid as a third argument.
# Dictionary's can intercept these functions and define their coefficient-to-value
# and value-to-coefficient transforms.


# The default transform space is the space associated with the grid of the set
transform_dict(s::Dictionary; options...) = GridBasis(s)

transform_operator(src::Dictionary; options...) = transform_operator(src, transform_dict(src); options...)

# If source or destination is a GridBasis, specialize
transform_operator(src::Dictionary, dest::GridBasis; options...) =
    transform_to_grid(src, dest, grid(dest); options...)
transform_operator(src::GridBasis, dest::Dictionary; options...) =
    transform_from_grid(src, dest, grid(src); options...)

transform_to_grid(src::Dictionary, dest::GridBasis; options...) =
    transform_to_grid(src, dest, grid(dest); options...)
transform_from_grid(src::GridBasis, dest::Dictionary; options...) =
    transform_from_grid(src, dest, grid(src); options...)

# Convert grid to GridBasis
transform_operator(src::Dictionary, grid::AbstractGrid; T = coefficienttype(src), options...) =
    transform_to_grid(src, GridBasis{T}(grid); options...)
transform_operator(grid::AbstractGrid, dest::Dictionary; T = coefficienttype(dest), options...) =
    transform_from_grid(GridBasis{T}(grid), dest; options...)
