
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
transform_dict(Φ::Dictionary; options...) = GridBasis(Φ)

transform_operator(src::Dictionary; options...) = transform_operator(src, transform_dict(src); options...)

transform_operator(src::Dictionary, dest::Dictionary; options...) =
    transform_operator(operatoreltype(src, dest), src, dest; options...)

# If source or destination is a GridBasis, specialize
transform_operator(T, src::Dictionary, dest::GridBasis; options...) =
    transform_to_grid(T, src, dest, grid(dest); options...)
transform_operator(T, src::GridBasis, dest::Dictionary; options...) =
    transform_from_grid(T, src, dest, grid(src); options...)

transform_to_grid(T, src::Dictionary, dest::GridBasis; options...) =
    transform_to_grid(T, src, dest, grid(dest); options...)
transform_from_grid(T, src::GridBasis, dest::Dictionary; options...) =
    transform_from_grid(T, src, dest, grid(src); options...)

# Convert grid to GridBasis
transform_operator(T, src::Dictionary, grid::AbstractGrid; options...) =
    transform_to_grid(T, src, GridBasis{T}(grid); options...)
transform_operator(T, grid::AbstractGrid, dest::Dictionary; options...) =
    transform_from_grid(T, GridBasis{T}(grid), dest; options...)
