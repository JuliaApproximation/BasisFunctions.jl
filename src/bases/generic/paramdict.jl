"""
A `ParamDict` is a dictionary combined with a map, which is the parameterization
of some domain that is the image of the support of the dictionary.

If `Ω` is a domain, parameterized by the function `κ : P → Ω`, then the `ParamDict`
is a dictionary defined on `P`, with each point `t ∈ P` identified with
`x = κ(t) ∈ Ω`. The `ParamDict` acts like any other dictionary on `P`.

Note the difference with a `MapppedDict`, which is defined on `Ω`. Operations
on mapped dictionaries require an easily invertible map.
"""
struct ParamDict{D,M,S,T} <: DerivedDict{S,T}
    superdict   ::  D
    map         ::  M
    image       ::  Domain
end

# In the constructor we check the domain and codomain types.
# - The domain of the ParamDict is the same as the domain of the map.
# - The map maps to the same type as elements of the domain
ParamDict(dict::Dictionary{S,T}, map::AbstractMap{S,U}, imag::Domain{U}) where {S,T,U} =
    ParamDict{typeof(dict),typeof(map),S,T}(dict, map, image)

param_dict(dict::Dictionary, map::AbstractMap) = ParamDict(dict, map)

mapping(dict::ParamDict) = dict.map
image(dict::ParamDict) = dict.image

similar_dictionary(dict::ParamDict, dict2::Dictionary) = ParamDict(dict2, mapping(dict), image(dict))


iscompatible(dict1::ParamDict, dict2::ParamDict) =
    iscompatible(mapping(dict1),mapping(dict2)) && iscompatible(superdict(dict1),superdict(dict2))


## Printing

name(dict::ParamDict) = "Parametric " * name(superdict(dict))

modifiersymbol(dict::ParamDict) = PrettyPrintSymbol{:L}(dict)

name(::PrettyPrintSymbol{:P}) = "Lifting"
string(s::PrettyPrintSymbol{:L}) = _string(s, s.object)
_string(s::PrettyPrintSymbol{:L}, dict::ParamDict) =
    "Lifting from $(support(superdict(dict))) to $(image(dict))"
