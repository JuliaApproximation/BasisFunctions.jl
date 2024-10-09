# Functionality for the promotion of dictionaries.

export promote_domaintype,
    promote_coefficienttype

# Convert the domaintype of the dictionary to `S`
convert(::Type{Dictionary{S}}, dict::Dictionary{S}) where {S} = dict
convert(::Type{Dictionary{S}}, dict::Dictionary) where {S} = similar(dict, S)


promote(d1::Dictionary, d2::Dictionary) =
    promote_coefficienttype(promote_domaintype(d1, d2)...)


## Domain type

"Promote the domain type of the dictionaries to a common type."
promote_domaintype(dict::Dictionary) = (dict,)

function promote_domaintype(d1::Dictionary, d2::Dictionary, d3::Dictionary...)
    d1b, d2b = promote_domaintype(d1, d2)
    (d1b, promote_domaintype(d2b, d3...))
end

promote_domaintype(d1::Dictionary, d2::Dictionary) =
    _promote_domaintype(d1, d2, promote_type(domaintype(d1), domaintype(d2)))
_promote_domaintype(d1, d2, ::Type{S}) where {S} =
    convert(Dictionary{S}, d1), convert(Dictionary{S}, d2)


## Coefficient type

"Promote the coefficient type of the dictionaries to a common type."
promote_coefficienttype(dict::Dictionary) = (dict,)

function promote_coefficienttype(d1::Dictionary, d2::Dictionary, d3::Dictionary...)
    d1b, d2b = promote_coefficienttype(d1, d2)
    (d1b, promote_coefficienttype(d2b, d3...))
end

promote_coefficienttype(d1::Dictionary, d2::Dictionary) =
    _promote_coefficienttype(d1, d2, coefficienttype(d1), coefficienttype(d2))

# Types are the same
_promote_coefficienttype(d1, d2, ::Type{T}, ::Type{T}) where {T} = (d1,d2)
# Types differ: compute a joined supertype
_promote_coefficienttype(d1, d2, ::Type{S}, ::Type{T}) where {S,T} =
    _promote_coefficienttype(d1, d2, S, T, promote_type(S,T))

_promote_coefficienttype(d1, d2, ::Type{S}, ::Type{T}, ::Type{U}) where {S,T,U} =
    ensure_coefficienttype(U, d1), ensure_coefficienttype(U, d2)

"Ensure that the dictionary (or dictionaries) have coefficient type `T` or wider."
ensure_coefficienttype(::Type{C}, dict::Dictionary) where {C} =
    _ensure_coefficienttype(C, ensure_prectype(prectype(C),dict))
_ensure_coefficienttype(::Type{C}, dict::Dictionary) where {C} =
    _ensure_coefficienttype(dict, C, coefficienttype(dict))
_ensure_coefficienttype(dict::Dictionary, ::Type{T}, ::Type{T}) where {T} = dict
# No fallback is provided, in order to generate an error when a coefficient type
# is requested that can not be achieved. Complexification of a dictionary is
# achieved by ComplexifiedDict.

ensure_coefficienttype(::Type{T}, dicts::Dictionary...) where {T} =
    ensure_coefficienttype.(T, dicts)

## Precision type

"Ensure that the dictionary has precision type `T` or a wider type."
ensure_prectype(::Type{T}, dict::Dictionary) where {T} =
    _ensure_prectype(dict, T, prectype(dict))
_ensure_prectype(dict::Dictionary, ::Type{T}, ::Type{T}) where {T} = dict
_ensure_prectype(dict::Dictionary, ::Type{S}, ::Type{T}) where {S,T} =
    _ensure_prectype(dict, S, T, promote_type(S,T))

_ensure_prectype(dict::Dictionary, S, T, ::Type{U}) where {U} =
    similar(dict, widen_prectype(domaintype(dict), U))

ensure_prectype(::Type{T}, dicts::Dictionary...) where {T} =
    ensure_prectype.(T, dicts)


"Widen the type such that its precision type matches `U`."
widen_prectype(::Type{Complex{T}}, ::Type{U}) where {T,U} = Complex{U}
widen_prectype(::Type{<:Number}, ::Type{U}) where {U<:Number} = U
widen_prectype(::Type{SVector{T,N}}, ::Type{U}) where {T,N,U} = SVector{widen_prectype(T,U),N}
widen_prectype(::Type{Array{T,N}}, ::Type{U}) where {T,N,U} = Array{widen_prectype(T,U),N}
widen_prectype(::Type{NTuple{N,T}}, ::Type{U}) where {T,N,U} = NTuple{N,widen_prectype(T,U)}
widen_prectype(::Type{Tuple{A}}, ::Type{U}) where {A,U} = Tuple{widen_prectype(A,U)}
widen_prectype(::Type{Tuple{A,B}}, ::Type{U}) where {A,B,U} =
    Tuple{widen_prectype(A,U),widen_prectype(B,U)}
widen_prectype(::Type{Tuple{A,B,C}}, ::Type{U}) where {A,B,C,U} =
    Tuple{widen_prectype(A,U),widen_prectype(B,U),widen_prectype(C,U)}


## Promotion with conversion

"""
Promote two dictionaries to a common family, in such a way that expansions in both given
dictionaries can be converted to expansions in the promoted dictionaries.
"""
promote_convertible(d1, d2) = promote_convertible1(promote_domaintype(d1, d2)...)
promote_convertible1(d1, d2) = promote_convertible2(d1, d2)
promote_convertible2(d1, d2) = default_promote_convertible(d1, d2)

nopromotion(d1,d2) = error("Don't know how to promote and convert between dictionaries $(d1) and $(d2).")

default_promote_convertible(d1::D, d2::D) where D = d1, d2
default_promote_convertible(d1, d2) =
    isequaldict(d1,d2) ? d1 : nopromotion(d1,d2)
