
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

"""
Return a suitable length to extend to, for example one such that the corresponding grids are nested
and function evaluations can be shared. The default is twice the length of the current set.
"""
extension_size(s::Dictionary) = 2*length(s)

extend(s::Dictionary) = resize(s, extension_size(s))

extension_operator(s1::Dictionary; options...) =
    extension_operator(s1, extend(s1); options...)


# For convenience with dispatch, add the grids as extra arguments when only
# GridBasis's are involved
extension_operator(src::GridBasis, dest::GridBasis; options...) =
    grid_restriction_operator(dest, src, grid(dest), grid(src); options...)'

# For convenience with dispatch, add the grids as extra arguments when only
# GridBasis's are involved
restriction_operator(src::GridBasis, dest::GridBasis; options...) =
    grid_restriction_operator(src, dest, grid(src), grid(dest); options...)

grid_restriction_operator(src, dest, src_grid, dest_grid; options...) =
    default_restriction_operator(src, dest; options...)


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
