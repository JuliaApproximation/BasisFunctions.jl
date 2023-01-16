
export conversion,
    extend,
    extension,
    extensionsize,
    restrict,
    restriction,
    restrictionsize,
    evaluation,
    approximation,
    change_basis

"""
```
conversion(::Type{T}, src::Dictionary, dest::Dictionary; options...)
```

Construct an operator with element type `T` that converts coefficients from the
source dictionary to coefficients of the destination dictionary.

The conversion is often exact, but this can not always be achieved with
certainty. For example, it is hard to know in advance whether converting to
evaluations in a grid is invertible.
"""
conversion

operatoreltype(Φ::Dictionary...) = promote_type(map(coefficienttype, Φ)...)

for op in (:conversion, :extension, :restriction, :evaluation)
    # Only dictionaries are given: compute the eltype
    @eval $op(dicts::Dictionary...; options...) = $op(operatoreltype(dicts...), dicts...; options...)
    # Only grids are given: make a GridBasis dictionary
    @eval $op(::Type{T}, src::AbstractGrid, dest::AbstractGrid; options...) where {T} =
        $op(T, GridBasis{T}(src), GridBasis{T}(dest); options...)
end



############################
# Extension and restriction
############################

# Conversion between dictionaries of the same type: decide between extension and restriction.
function conversion(::Type{T}, src::D, dest::D; options...) where {T,D <: Dictionary}
    if src == dest
        IdentityOperator{T}(src)
    else
        extension(T, src, dest; options...)
    end
end

function extension_restriction(T, src::Dictionary, dest::Dictionary; options...)
    if dimensions(src) == dimensions(dest)
        IdentityOperator{T}(src, dest)
    elseif length(src) < length(dest)
        extension(T, src, dest; options...)
    elseif length(src) > length(dest)
        restriction(T, src, dest; options...)
    else
        # lengths are equal but dimensions are not
        error("Don't know how to convert between two `$(repr(src))` dictionaries with different structure.")
        # Perhaps we can do something elementwise
    end
end

# Transforming between dictionaries with the same type is the same as restricting
hastransform(src::D, dest::D) where {D <: Dictionary} = true


extension(::Type{T}, src::D, dest::D; options...) where {T,D} =
    error("Don't know how to extend dictionary $(name(src)) from size $(size(src)) to size $(size(dest))")

extension(::Type{T}, src::Dictionary, dest::Dictionary; options...) where {T} =
    error("Can't do extension between dictionaries of different type")


restriction(::Type{T}, src::D, dest::D; options...) where {T,D} =
        error("Don't know how to extend dictionary $(name(src)) from size $(size(src)) to size $(size(dest))")

restriction(::Type{T}, src::Dictionary, dest::Dictionary; options...) where {T} =
    error("Can't do restriction between dictionaries of different type")

"""
Return a suitable length to extend to, for example one such that the corresponding grids are nested
and function evaluations can be shared. The default is twice the length of the current set.
"""
extensionsize(s::Dictionary) = 2*length(s)

extend(Φ::Dictionary) = resize(Φ, extensionsize(Φ))

# only a single dictionary is given
extension(::Type{T}, src::Dictionary; options...) where {T} =
    extension(T, src, extend(src); options...)

"""
Return a suitable length to restrict to, for example one such that the corresponding grids are nested
and function evaluations can be shared. The default is half the length of the current set.
"""
restrictionsize(Φ::Dictionary) = length(Φ)>>1

restrict(Φ::Dictionary) = resize(Φ, restrictionsize(Φ))

restriction(::Type{T}, src::Dictionary; options...) where {T} =
    restriction(T, src, restrict(src); options...)


# For convenience with dispatch, add the grids as extra arguments when only
# GridBasis's are involved, and support AbstractGrid arguments.
extension(::Type{T}, src::GridBasis, dest::GridBasis; options...) where {T} =
    gridrestriction(T, dest, src, grid(dest), grid(src); options...)'

restriction(::Type{T}, src::GridBasis, dest::GridBasis; options...) where {T} =
    gridrestriction(T, src, dest, grid(src), grid(dest); options...)

function gridrestriction(::Type{T}, src::Dictionary, dest::Dictionary, src_grid::G, dest_grid::GridArrays.MaskedGrid{G,M,I,S}; options...) where {T,G<:AbstractGrid,M,I,S}
    @assert supergrid(dest_grid) == src_grid
    IndexRestriction{T}(src, dest, subindices(dest_grid))
end


hasextension(dg::GridBasis{T,G}) where {T,G <: GridArrays.SubGrid} = true
hasextension(dg::GridBasis{T,G}) where {T,G <: GridArrays.ProductSubGrid} = true


###############
# Projection
###############

projection(src::Dictionary, dest::Dictionary, args...; options...) =
    projection(operatoreltype(src,dest), src, dest, args...; options...)

projection(::Type{T}, src, dest, μ = measure(dest); options...) where {T} =
    projection1(T, src, dest, μ; options...)
projection1(T, src::Dictionary, dest, μ; options...) =
    projection2(T, src, dest, μ; options...)
projection2(T, src, dest::Dictionary, μ; options...) =
    projection3(T, src, dest, μ; options...)
projection3(T, src, dest, μ::Measure; options...) =
    default_projection(T, src, dest, μ; options...)

default_projection(src::Dictionary, dest::Dictionary, μ = measure(dest); options...) =
    default_projection(operatoreltype(src,dest), src, dest, μ; options...)
function default_projection(T, src, dest, μ = measure(dest); options...)
    if src == dest
        IdentityOperator{T}(src, dest)
    else
        G = mixedgram(T, dest, src, μ; options...)
        D = Diagonal([1/norm(bf)^2 for bf in dest])
        ArrayOperator(D*matrix(G), src, dest)
    end
end


###############################
# Generic conversion
###############################

change_basis(F1::Expansion; dest::Dictionary) =
    conversion(dictionary(F1), dest) * F1

conversion(::Type{T}, src, dest; options...) where {T} =
    conversion1(T, src, dest; options...)
conversion1(T, src::Dictionary, dest; options...) =
    conversion2(T, src, dest; options...)
function conversion2(T, src, dest::Dictionary; verbose=false, options...)
    if issubset(Span(src), Span(dest))
        verbose && println("WARN: using default conversion from $(src) to $(dest)")
        default_conversion(T, src, dest; verbose, options...)
    else
        error("No known exact conversion from $(src) to $(dest)")
    end
end

default_conversion(T, src, dest; options...) =
    projection(T, src, dest; options...)

# Convert to and from grids.
# To grid: we invoke `evaluation`
# From grid: we invoke `interpolation`.
# Problem is that we don't know in general whether these would be exact.
conversion(::Type{T}, src::Dictionary, dest::GridBasis; options...) where {T} =
    evaluation(T, src, dest; options...)

conversion(::Type{T}, src::GridBasis, dest::Dictionary; options...) where {T} =
    interpolation(T, src, dest; options...)

# Resolve ambiguity by the above methods
conversion(::Type{T}, src::GridBasis, dest::GridBasis; options...) where {T} =
    extension(T, src, dest; options...)
