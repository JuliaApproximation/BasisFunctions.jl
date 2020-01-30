
export conversion,
    extend,
    extension,
    restriction

"""
```
conversion(::Type{T}, src::Dictionary, dest::Dictionary; options...)
```

Construct an operator with element type `T` that converts coefficients from the
source dictionary to coefficients of the destination dictionary.
"""
conversion

operatoreltype(Φ::Dictionary...) = promote_type(map(coefficienttype, Φ)...)

conversion(src::Dictionary, dest::Dictionary; options...) =
    conversion(operatoreltype(src, dest), src, dest; options...)


############################
# Extension and restriction
############################

# Conversion between dictionaries of the same type: decide between extension and restriction.
function conversion(::Type{T}, src::D, dest::D; options...) where {T,D <: Dictionary}
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

# Some convenience functions follow

# only dictionaries are given
extension(dicts::Dictionary...; options...) = extension(operatoreltype(dicts...), dicts...; options...)
restriction(dicts::Dictionary...; options...) = restriction(operatoreltype(dicts...), dicts...; options...)

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



###############################
# Converting to grids and back
###############################


# Convert to and from grids.
conversion(T, src::Dictionary, dest::GridBasis; options...) =
    togrid(T, src, dest; options...)

conversion(T, src::GridBasis, dest::Dictionary; options...) =
    fromgrid(T, src, dest; options...)
