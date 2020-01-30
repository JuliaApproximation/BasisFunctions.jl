
####################################
# Generic extension and restriction
####################################
"""
An extension operator is an operator that can be used to extend a representation in a set s1 to a
representation in a larger set s2.
"""
function extension_operator(s1::Dictionary, s2::Dictionary; options...) end


"""
A restriction operator does the opposite of what the extension operator does: it restricts
a representation in a set s1 to a representation in a smaller set s2. Loss of accuracy may result
from the restriction.
"""
function restriction_operator(s1::Dictionary, s2::Dictionary; options...) end


# For convenience with dispatch, add the grids as extra arguments when only
# GridBasis's are involved
extension_operator(src::GridBasis, dest::GridBasis; options...) =
    grid_restriction_operator(dest, src, grid(dest), grid(src); options...)'

# For convenience with dispatch, add the grids as extra arguments when only
# GridBasis's are involved
restriction_operator(src::GridBasis, dest::GridBasis; options...) =
    grid_restriction_operator(src, dest, grid(src), grid(dest); options...)

function grid_restriction_operator(src, dest, src_grid, dest_grid; options...)
    @show src, dest, src_grid, dest_grid
    default_restriction_operator(src, dest; options...)
end

function grid_restriction_operator(src::Dictionary, dest::Dictionary, src_grid::G, dest_grid::GridArrays.MaskedGrid{G,M,I,T}; options...) where {G<:AbstractGrid,M,I,T}
    @assert supergrid(dest_grid) == src_grid
    IndexRestriction(src, dest, subindices(dest_grid))
end


hasextension(dg::GridBasis{T,G}) where {T,G <: GridArrays.AbstractSubGrid} = true
hasextension(dg::GridBasis{T,G}) where {T,G <: GridArrays.TensorSubGrid} = true



"""
Return a suitable length to restrict to, for example one such that the corresponding grids are nested
and function evaluations can be shared. The default is half the length of the current set.
"""
restriction_size(s::Dictionary) = length(s)>>1

restrict(s::Dictionary) = resize(s, restriction_size(s))

restriction_operator(s1::Dictionary; options...) =
    restriction_operator(s1, restrict(s1); options...)

# Transforming between dictionaries with the same type is the same as restricting
hastransform(src::D, dest::D) where {D <: Dictionary} = true
