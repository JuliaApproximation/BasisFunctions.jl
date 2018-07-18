# extension.jl

####################################
# Generic extension and restriction
####################################

# A dictionary can often be extended or restricted to a larger or smaller one.
# Extension and restriction operators always have a source and destination set.
# For introspection, you can ask a set what a suitable extensize size would be,
# and use a resized set as the destination of the operator:
#
#   larger_size = extension_size(some_set)
#   larger_set = resize(some_set, larger_size)
#   E = extension_operator(some_set, larger_set)
#
# and similarly for restriction. This is the default if no destination set is
# supplied to extension_operator.
#
# If a set does not need to store data for its extension operator, it can simply
# choose to implement the action of the Extension operator, e.g.:
#
# apply!(op::Extension, s1::SomeSet, s2::SomeSet, coef_dest, coef_src) = ...

struct Extension{SRC <: Dictionary,DEST <: Dictionary,ELT} <: DictionaryOperator{ELT}
    src     ::  SRC
    dest    ::  DEST
end

Extension(src::Dictionary, dest::Dictionary) =
    Extension{typeof(src),typeof(dest),op_eltype(src,dest)}(src, dest)

function Extension(::Type{ELT}, src::Dictionary, dest::Dictionary) where {ELT}
    S, D, A = op_eltypes(src, dest, ELT)
    new_src = promote_coeftype(src,S)
    new_dest = promote_coeftype(dest,D)
    Extension(new_src, new_dest)
end

similar_operator(op::Extension, ::Type{S}, src::Dictionary, dest::Dictionary) where {S} =
    Extension(S, src, dest)

"""
An extension operator is an operator that can be used to extend a representation in a set s1 to a
representation in a larger set s2. The default extension operator is of type Extension with s1 and
s2 as source and destination.
"""
function extension_operator(s1::Dictionary, s2::Dictionary; options...)
    @assert has_extension(s1)
    default_extension_operator(s1, s2; options...)
end

# default_extension_operator(s1::Dictionary, s2::Dictionary; options...) =
#     IndexExtensionOperator(s1, s2, 1:length(s1))
default_extension_operator(s1::Dictionary, s2::Dictionary; options...) =
    Extension(s1, s2)


"""
Return a suitable length to extend to, for example one such that the corresponding grids are nested
and function evaluations can be shared. The default is twice the length of the current set.
"""
extension_size(s::Dictionary) = 2*length(s)

extend(s::Dictionary) = resize(s, extension_size(s))

extension_operator(s1::Dictionary; options...) =
    extension_operator(s1, extend(s1); options...)


struct Restriction{SRC <: Dictionary,DEST <: Dictionary,ELT} <: DictionaryOperator{ELT}
    src     ::  SRC
    dest    ::  DEST
end

Restriction(src::Dictionary, dest::Dictionary) =
    Restriction{typeof(src),typeof(dest),op_eltype(src,dest)}(src, dest)

function Restriction(::Type{ELT}, src::Dictionary, dest::Dictionary) where {ELT}
    S, D, A = op_eltypes(src, dest, ELT)
    new_src = promote_coeftype(src,S)
    new_dest = promote_coeftype(dest,D)
    Restriction(new_src, new_dest)
end

similar_operator(op::Restriction, ::Type{S}, src::Dictionary, dest::Dictionary) where {S} =
    Restriction(S, src, dest)


"""
A restriction operator does the opposite of what the extension operator does: it restricts
a representation in a set s1 to a representation in a smaller set s2. Loss of accuracy may result
from the restriction. The default restriction_operator is of type Restriction with sets s1 and
s2 as source and destination.
"""
function restriction_operator(s1::Dictionary, s2::Dictionary; options...)
    @assert has_extension(s2)
    default_restriction_operator(s1, s2; options...)
end

default_restriction_operator(s1::Dictionary, s2::Dictionary; options...) =
    Restriction(s1, s2)


# For convenience with dispatch, add the grids as extra arguments when only
# GridBasis's are involved
extension_operator(src::GridBasis, dest::GridBasis; options...) =
    grid_extension_operator(src, dest, grid(src), grid(dest); options...)

grid_extension_operator(src, dest, src_grid, dest_grid; options...) =
    default_extension_operator(src, dest; options...)

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

if VERSION < v"0.7-"
    ctranspose(op::Extension) = restriction_operator(dest(op), src(op))
    ctranspose(op::Restriction) = extension_operator(dest(op), src(op))
else
    adjoint(op::Extension) = restriction_operator(dest(op), src(op))
    adjoint(op::Restriction) = extension_operator(dest(op), src(op))
end

# Transforming between dictionaries with the same type is the same as restricting
has_transform(src::D, dest::D) where {D <: Dictionary} = true

function transform_operator(src::D, dest::D) where {D <: Dictionary}
    if length(src) > length(dest)
        return Restriction(src, dest)
    elseif length(src) < length(dest)
        return Extension(src, dest)
    else
        return Identity(src, dest)
    end
end
