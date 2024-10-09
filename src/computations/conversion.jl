
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
conversion([::Type{T}, ]src::Dictionary, dest::Dictionary; options...)
```

Construct an operator that converts coefficients from the source dictionary to
coefficients of the destination dictionary.

The conversion is exact, up to side-effects of finite-precision calculations.
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
    error("Don't know how to extend dictionary $(src) from size $(size(src)) to size $(size(dest))")

extension(::Type{T}, src::Dictionary, dest::Dictionary; options...) where {T} =
    error("Can't do extension between dictionaries of different type")


restriction(::Type{T}, src::D, dest::D; options...) where {T,D} =
        error("Don't know how to extend dictionary $(src) from size $(size(src)) to size $(size(dest))")

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

function gridrestriction(::Type{T}, src::Dictionary, dest::Dictionary, src_grid, dest_grid::GridArrays.MaskedGrid; options...) where {T}
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

noconversion(src, dest) = error("No known exact conversion from $(src) to $(dest).")

conversion(::Type{T}, src, dest; options...) where T =
    conversion1(T, src, dest; options...)
conversion1(T, src, dest; options...) = conversion2(T, src, dest; options...)
conversion2(T, src, dest; options...) = default_conversion(T, src, dest; options...)

function default_conversion(::Type{T}, src, dest; verbose=false, options...) where T
    if isequaldict(src, dest)
        IdentityOperator{T}(src, dest)
    elseif issubset(Span(src), Span(dest))
        verbose && println("WARN: using default conversion from $(src) to $(dest)")
        explicit_conversion(T, src, dest; verbose, options...)
    else
        noconversion(src, dest)
    end
end

function explicit_conversion(::Type{T}, src, dest; options...) where T
    if hasmeasure(dest)
        projection(T, src, dest, measure(dest); options...)
    elseif hasinterpolationgrid(dest)
        pts = interpolation_grid(dest)
        I = interpolation(T, dest; options...)
        A = evaluation(T, src, pts; options...)
        I*A
    elseif hasinterpolationgrid(src)
        pts = interpolation_grid(src)
        I = interpolation(T, dest, pts; options...)
        A = evaluation(T, src, pts; options...)
        I*A
    else
        noconversion(src, dest)
    end
end

# Convert to and from grids.
# To grid: we invoke `evaluation`
# From grid: we invoke `interpolation`.
# Problem is that we don't know in general whether these would be exact.
# TODO: these should be removed (breaking change)
conversion(::Type{T}, src::Dictionary, dest::GridBasis; options...) where {T} =
    evaluation(T, src, dest; options...)

conversion(::Type{T}, src::GridBasis, dest::Dictionary; options...) where {T} =
    interpolation(T, src, dest; options...)

# Resolve ambiguity by the above methods
conversion(::Type{T}, src::GridBasis, dest::GridBasis; options...) where {T} =
    extension(T, src, dest; options...)


## Normalization

function isnormalized(Φ::Dictionary, μ = measure(Φ))
	for i in eachindex(Φ)
		if !(norm(Φ[i], μ) ≈ 1)
			return false
		end
	end
	return true
end

normalize(Φ::Dictionary, μ = measure(Φ)) =
	isnormalized(Φ, μ) ? Φ : Diagonal([inv(norm(φ, μ)) for φ in Φ]) * Φ

## Orthogonalization

"Compute the square root of a symmetric and positive definite matrix."
spd_matrix_sqrt(A::AbstractArray{T}) where {T<:Base.IEEEFloat}= sqrt(A)
function spd_matrix_sqrt(A::AbstractArray)
	# at the time of writing sqrt(A) is not supported for generic numbers
	# but svd is (in the GenericLinearAlgebra package)
	u,s,v = svd(A)
	u * sqrt(Diagonal(s)) * u'
end

orthogonalize(Φ::Dictionary, μ = measure(Φ)) = orthogonalize1(Φ, μ)
# enable dispatch on the first argument without ambiguity
orthogonalize1(Φ::Dictionary, μ) = orthogonalize2(Φ, μ)
# enable dispatch on the second argument without ambiguity
function orthogonalize2(Φ, μ)
	if isorthonormal(Φ, μ)
		Φ
	else
		default_orthogonalize(Φ, μ)
	end
end

function default_orthogonalize(Φ::Dictionary, μ = measure(Φ))
	A = inv(matrix(gram(Φ, μ)))
	spd_matrix_sqrt(A) * Φ
end
