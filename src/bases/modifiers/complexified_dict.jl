
"""
A 'ComplexifiedDict' is a dictionary for which the coefficient type is the complex
version of the original dictionary. It is obtained by calling `promote` or
`ensure_coefficienttype` on a dictionary or dictionaries.
"""
struct ComplexifiedDict{D,S,T} <: HighlySimilarDerivedDict{S,T}
    superdict   :: D
end

const ComplexifiedDict1d{D,S<:Number,T<:Number} = ComplexifiedDict{D,S,T}

ComplexifiedDict(d::Dictionary{S,T}) where {S,T<:Real} = ComplexifiedDict{typeof(d),S,T}(d)
ComplexifiedDict(d::ComplexifiedDict) = d

Base.complex(dict::Dictionary) = ensure_coefficienttype(complex(coefficienttype(dict)), dict)
function Base.real(dict::Dictionary)
    @assert isreal(dict)
    dict
end
Base.real(dict::ComplexifiedDict) = superdict(dict)

similardictionary(s::ComplexifiedDict, s2::Dictionary) = ComplexifiedDict(s2)

components(dict::ComplexifiedDict) = map(ComplexifiedDict, components(superdict(dict)))
component(dict::ComplexifiedDict, i) = ComplexifiedDict(component(superdict(dict), i))

# This is the general rule to complexify a dictionary
_ensure_coefficienttype(dict::Dictionary, ::Type{Complex{T}}, ::Type{T}) where {T} =
    ComplexifiedDict(dict)

coefficienttype(dict::ComplexifiedDict) = complex(coefficienttype(superdict(dict)))

transform_dict(dict::ComplexifiedDict) = complex(transform_dict(superdict(dict)))

evaluation(::Type{T}, dict::ComplexifiedDict, gb::GridBasis, grid; options...) where {T} =
    evaluation(T, superdict(dict), gb, grid; options...)


hasmeasure(dict::ComplexifiedDict) = hasmeasure(superdict(dict))
measure(dict::ComplexifiedDict) = measure(superdict(dict))

gram1(T, dict::BasisFunctions.ComplexifiedDict, measure; options...) =
    wrap_operator(dict, dict, gram(T, superdict(dict), measure; options...))

dict_innerproduct1(d1::ComplexifiedDict, i, d2, j, measure; options...) =
    dict_innerproduct(superdict(d1), i, d2, j, measure; options...)
dict_innerproduct2(d1, i, d2::ComplexifiedDict, j, measure; options...) =
    dict_innerproduct(d1, i, superdict(d2), j, measure; options...)

span_isequal(d1::ComplexifiedDict, d2::ComplexifiedDict) =
    span_isequal(superdict(d1), superdict(d2))

span_issubset(d1::ComplexifiedDict, d2::ComplexifiedDict) =
    span_issubset(superdict(d1), superdict(d2))
span_issubset2(d1, d2::ComplexifiedDict) = span_issubset(d1, superdict(d2))

iscompatible(d1::Dictionary, d2::ComplexifiedDict) = iscompatible(d1, superdict(d2))
iscompatible(d1::ComplexifiedDict, d2::Dictionary) = iscompatible(superdict(d1), d2)
iscompatible(d1::ComplexifiedDict, d2::ComplexifiedDict) = iscompatible(superdict(d1), superdict(d2))
function conversion2(T, d1, d2::ComplexifiedDict; options...)
    @assert !isreal(T)
    op = conversion(T, d1, superdict(d2); options...)
    wrap_operator(d1, d2, op)
end

hasconstant(d::ComplexifiedDict) = hasconstant(superdict(d))
hasx(d::ComplexifiedDict) = hasx(superdict(d))
coefficients_of_one(dict::ComplexifiedDict) =
    complex(coefficients_of_one(superdict(dict)))
coefficients_of_x(dict::ComplexifiedDict) =
    complex(coefficients_of_x(superdict(dict)))

## Printing

show(io::IO, d::ComplexifiedDict) = print(io, "complex($(repr(superdict(d))))")
modifiersymbol(d::ComplexifiedDict) = "complex"
