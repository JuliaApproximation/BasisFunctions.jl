"""
A `ParamDict` is a dictionary combined with a map, which is the parameterization
of some domain that is the image of the support of the dictionary.

If `Ω` is a domain, parameterized by the function `κ : P → Ω`, then the `ParamDict`
is a dictionary defined on `P`, with each point `t ∈ P` identified with
`x = κ(t) ∈ Ω`. The `ParamDict` acts like any other dictionary on `P`.

Note the difference with a `MappedDict`, which is defined on `Ω`. Operations
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
ParamDict(dict::Dictionary{S,T}, map::Map{S}, image::Domain) where {S,T} =
    ParamDict{typeof(dict),typeof(map),S,T}(dict, map, image)

function ParamDict(dict::Dictionary, map)
    image = DomainSets.parametric_domain(map, support(dict))
    ParamDict(dict, map, image)
end

param_dict(dict::Dictionary, map::Map) = ParamDict(dict, map)

forward_map(dict::ParamDict) = dict.map
image(dict::ParamDict) = dict.image

similardictionary(dict::ParamDict, dict2::Dictionary) =
    ParamDict(dict2, forward_map(dict), image(dict))


iscompatible(dict1::ParamDict, dict2::ParamDict) =
    iscompatible(forward_map(dict1),forward_map(dict2)) && iscompatible(superdict(dict1),superdict(dict2))


## Printing

Display.displaystencil(d::ParamDict) = ["ParamDict(", superdict(d), ", ", forward_map(d), ", ", image(d), ")"]
