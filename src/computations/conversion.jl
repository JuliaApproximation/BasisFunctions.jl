
export conversion,
    extend,
    extension,
    extensionsize,
    restrict,
    restriction,
    restrictionsize

"""
```
conversion(::Type{T}, src::Dictionary, dest::Dictionary; options...)
```

Construct an operator with element type `T` that converts coefficients from the
source dictionary to coefficients of the destination dictionary.
"""
conversion

operatoreltype(Φ::Dictionary...) = promote_type(map(coefficienttype, Φ)...)

for op in (:conversion, :extension, :restriction, :evaluation, :approximation)
    # Only dictionaries are given: compute the eltype
    @eval $op(dicts::Dictionary...; options...) = $op(operatoreltype(dicts...), dicts...; options...)
end



############################
# Extension and restriction
############################

# Conversion between dictionaries of the same type: decide between extension and restriction.
conversion(::Type{T}, src::D, dest::D; options...) where {T,D <: Dictionary} =
    extension_restriction(T, src, dest; options...)

function extension_restriction(T, src::Dictionary, dest::Dictionary; options...)
    if dimensions(src) == dimensions(dest)
        IdentityOperator{T}(src, dest)
    elseif length(src) < length(dest)
        extension(T, src, dest; options...)
    elseif length(src) > length(dest)
        restriction(T, src, dest; options...)
    else
        # lengths are equal but dimensions are not
        error("Don't know how to convert between two `$(name(src))` dictionaries with different structure.")
        # Perhaps we can do something elementwise
    end
end

# Transforming between dictionaries with the same type is the same as restricting
hastransform(src::D, dest::D) where {D <: Dictionary} = true


extension(T, src::D, dest::D; options...) where {D} =
    error("Don't know how to extend dictionary $(name(src)) from size $(size(src)) to size $(size(dest))")

extension(T, src::Dictionary, dest::Dictionary; options...) =
    error("Can't do extension between dictionaries of different type")


restriction(T, src::D, dest::D; options...) where {D} =
        error("Don't know how to extend dictionary $(name(src)) from size $(size(src)) to size $(size(dest))")

restriction(T, src::Dictionary, dest::Dictionary; options...) =
    error("Can't do restriction between dictionaries of different type")

"""
Return a suitable length to extend to, for example one such that the corresponding grids are nested
and function evaluations can be shared. The default is twice the length of the current set.
"""
extensionsize(s::Dictionary) = 2*length(s)

extend(Φ::Dictionary) = resize(Φ, extensionsize(Φ))

# only a single dictionary is given
extension(T, src::Dictionary; options...) = extension(T, src, extend(src); options...)

"""
Return a suitable length to restrict to, for example one such that the corresponding grids are nested
and function evaluations can be shared. The default is half the length of the current set.
"""
restrictionsize(Φ::Dictionary) = length(Φ)>>1

restrict(Φ::Dictionary) = resize(Φ, restrictionsize(Φ))

restriction(T, src::Dictionary; options...) = restriction(T, src, restrict(src); options...)


# For convenience with dispatch, add the grids as extra arguments when only
# GridBasis's are involved
extension(::Type{T}, src::GridBasis, dest::GridBasis; options...) where {T} =
    gridrestriction(T, dest, src, grid(dest), grid(src); options...)'

restriction(::Type{T}, src::GridBasis, dest::GridBasis; options...) where {T} =
    gridrestriction(T, src, dest, grid(src), grid(dest); options...)


function gridrestriction(::Type{T}, src::Dictionary, dest::Dictionary, src_grid::G, dest_grid::GridArrays.MaskedGrid{G,M,I,S}; options...) where {T,G<:AbstractGrid,M,I,S}
    @assert supergrid(dest_grid) == src_grid
    IndexRestriction{T}(src, dest, subindices(dest_grid))
end


hasextension(dg::GridBasis{T,G}) where {T,G <: GridArrays.AbstractSubGrid} = true
hasextension(dg::GridBasis{T,G}) where {T,G <: GridArrays.TensorSubGrid} = true


###############################
# Converting to grids and back
###############################

# Convert to and from grids.
# To grid: we invoke `evaluation`
# From grid: we invoke `approximation`. This defaults to inverting the corresponding
#  `evaluation` operator.

conversion(T, src::Dictionary, dest::GridBasis; options...) =
    evaluation(T, src, dest; options...)

evaluation(T, src::Dictionary, dest::GridBasis; options...) =
    grid_evaluation(T, src, dest, grid(dest); options...)

conversion(T, src::GridBasis, dest::Dictionary; options...) =
    approximation(T, src, dest; options...)

# Resolve ambiguity by the above methods
conversion(T, src::GridBasis, dest::GridBasis; options...) =
    extension_restriction(T, src, dest; options...)

approximation(T, src, dest; options) = pinv(evaluation(T, dest, src; options...))
