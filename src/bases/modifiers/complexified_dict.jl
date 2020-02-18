
"""
A 'ComplexifiedDict' is a dictionary for which the coefficient type is the complex
version of the original dictionary. It is obtained by calling `promote` or
`ensure_coefficienttype` on a dictionary or dictionaries.
"""
struct ComplexifiedDict{D,S,T} <: DerivedDict{S,T}
    superdict   :: D
end

const ComplexifiedDict1d{D,S<:Number,T<:Number} = ComplexifiedDict{D,S,T}

ComplexifiedDict(d::Dictionary{S,T}) where {S,T<:Real} = ComplexifiedDict{typeof(d),S,T}(d)

Base.complex(dict::Dictionary) = ensure_coefficienttype(complex(coefficienttype(dict)), dict)

similardictionary(s::ComplexifiedDict, s2::Dictionary) = ComplexifiedDict(s2)

elements(dict::ComplexifiedDict) = map(ComplexifiedDict, elements(superdict(dict)))
element(dict::ComplexifiedDict, i) = ComplexifiedDict(element(superdict(dict), i))

# This is the general rule to complexify a dictionary
_ensure_coefficienttype(dict::Dictionary, ::Type{Complex{T}}, ::Type{T}) where {T} =
    ComplexifiedDict(dict)

apply_map(dict::ComplexifiedDict, map) = mapped_dict(dict, map)

coefficienttype(dict::ComplexifiedDict) = complex(coefficienttype(superdict(dict)))

transform_dict(dict::ComplexifiedDict) = complex(transform_dict(superdict(dict)))

evaluation(::Type{T}, dict::ComplexifiedDict, gb::GridBasis, grid::AbstractGrid; options...) where {T} =
    evaluation(T, superdict(dict), gb, grid; options...)


hasmeasure(dict::ComplexifiedDict) = hasmeasure(superdict(dict))

measure(dict::ComplexifiedDict) = measure(superdict(dict))

gram1(T, dict::BasisFunctions.ComplexifiedDict, measure; options...) =
    wrap_operator(dict, dict, gram(T, superdict(dict), measure; options...))

innerproduct1(d1::ComplexifiedDict, i, d2, j, measure; options...) =
    innerproduct(superdict(d1), i, d2, j, measure; options...)
innerproduct2(d1, i, d2::ComplexifiedDict, j, measure; options...) =
    innerproduct(d1, i, superdict(d2), j, measure; options...)


## Printing

name(dict::ComplexifiedDict) = "Complex " * name(superdict(dict))

modifiersymbol(dict::ComplexifiedDict) = PrettyPrintSymbol{:ℂ}()
name(::PrettyPrintSymbol{:ℂ}) = "Complexification"
