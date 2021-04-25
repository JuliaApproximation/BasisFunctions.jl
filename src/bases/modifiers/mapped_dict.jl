
export →,
    rescale,
    MappedDict,
    mapped_dict,
    forward_map

"""
A `MappedDict` has a dictionary and a map. The domain of the dictionary is
mapped to a different one. Evaluating the `MappedDict` in a point uses the
inverse map to evaluate the underlying dictionary element in the corresponding
point.
"""
abstract type MappedDict{S,T} <: DerivedDict{S,T} end

const MappedDict1d{S <: Number,T <: Number} = MappedDict{S,T}
const MappedDict2d{S <: SVector{2},T <: Number} = MappedDict{S,T}

mapped_dict(dict::Dictionary, map) = MappedDict(dict, map)
mapped_dict(dict::MappedDict, map) = mapped_dict(superdict(dict), map ∘ forward_map(dict))

# Convenience function, similar to apply_map for grids etcetera
@deprecate apply_map(dict::Dictionary, map) mapped_dict(dict, map)

# In the constructor we check the domain and codomain types.
# The domain of the MappedDict is defined by the range of the map, because the
# domain of the underlying dict is mapped to the domain of the MappedDict.
# Hence, the domain type of the map has to equal the domain type of the dictionary.
MappedDict(dict::Dictionary{T1,T}, map::Map{T1}) where {T1,T} =
    MappedDict{codomaintype(map,T1),T}(dict, map)
MappedDict(dict::Dictionary{S1,T1}, map::Map{S2}) where {S1,S2,T1} =
    MappedDict(dict, convert(Map{S1}, map))

MappedDict{S,T}(dict::Dictionary, map) where {S,T} = GenericMappedDict{S,T}(dict, map)

MappedDict{S,T}(dict::Dictionary, map::ScalarAffineMap{S}) where {S,T} =
    ScalarMappedDict{S,T}(dict, map)

"Concrete mapped dictionary for any dictionary and any map."
struct GenericMappedDict{S,T,D,M} <: MappedDict{S,T}
    superdict   ::  D
    map         ::  M
end

GenericMappedDict{S,T}(dict::Dictionary, map) where {S,T} =
    GenericMappedDict{S,T,typeof(dict),typeof(map)}(dict, map)


"Concrete mapped dictionary for any dictionary with a scalar map."
struct ScalarMappedDict{S,T,D} <: MappedDict{S,T}
    superdict   ::  D
    map         ::  ScalarAffineMap{S}
end

ScalarMappedDict{S,T}(dict::Dictionary, map) where {S,T} =
    GenericMappedDict{S,T,typeof(dict),typeof(map)}(dict, map)



forward_map(dict::MappedDict) = dict.map
inverse_map(dict::MappedDict) = inverse(forward_map(dict))

similardictionary(dict::MappedDict, dict2::Dictionary) = mapped_dict(dict2, forward_map(dict))

hasderivative(dict::MappedDict) = false
hasantiderivative(dict::MappedDict) = false

hasderivative(dict::MappedDict1d) =
    hasderivative(superdict(dict)) && isaffine(forward_map(dict))
hasantiderivative(dict::MappedDict1d) =
    hasantiderivative(superdict(dict)) && isaffine(forward_map(dict))

interpolation_grid(dict::MappedDict) = _grid(dict, superdict(dict), forward_map(dict))
_grid(dict::MappedDict, sdict, map) = map_grid(interpolation_grid(sdict), map)


function unmap_grid(dict::MappedDict, grid::MappedGrid)
    if iscompatible(forward_map(dict), forward_map(grid))
        supergrid(grid)
    else
        apply_map(grid, inverse_map(dict))
    end
end

unmap_grid(dict::MappedDict, grid::AbstractGrid) = apply_map(grid, inverse_map(dict))


isreal(s::MappedDict) = isreal(superdict(s)) && isreal(forward_map(s))

unsafe_eval_element(s::MappedDict, idx, y) =
    unsafe_eval_element(superdict(s), idx, leftinverse(forward_map(s), y))

function unsafe_eval_element_derivative(s::MappedDict1d, idx, y, order)
    @assert order == 1
    x = leftinverse(forward_map(s), y)
    d = unsafe_eval_element_derivative(superdict(s), idx, x, order)
    z = d / jacdet(forward_map(s), y)
end

function unsafe_eval_element_derivative(dict::MappedDict, idx, y, order)
    @assert maximum(order) <= 1
    m = forward_map(dict)
    x = leftinverse(m, y)
    J = jacobian(m, y)
    g = eval_gradient(superdict(dict), idx, x)
    sum(order .* (J' \ g))
end

function eval_expansion(s::MappedDict{S,T}, coef, y::S) where {S,T}
    if in_support(s, first(eachindex(s)), y)
        eval_expansion(superdict(s), coef, leftinverse(forward_map(s), y))
    else
        zero(codomaintype(s))
    end
end

support(dict::MappedDict) = DomainSets.map_domain(forward_map(dict), support(superdict(dict)))

support(dict::MappedDict, idx) = forward_map(dict).(support(superdict(dict), idx))

function dict_in_support(set::MappedDict, idx, y, threshold = default_threshold(y))
    x = leftinverse(forward_map(set), y)
    y1 = forward_map(set)(x)
    if norm(y-y1) < threshold
        in_support(superdict(set), idx, x)
    else
        false
    end
end

iscompatible(s1::MappedDict, s2::MappedDict) =
    iscompatible(forward_map(s1),forward_map(s2)) && iscompatible(superdict(s1),superdict(s2))

measure(dict::MappedDict) = apply_map(measure(superdict(dict)), forward_map(dict))

hasmeasure(dict::MappedDict) = hasmeasure(superdict(dict))

for f in (:isorthonormal, :isorthogonal, :isbiorthogonal)
    @eval $f(dict::MappedDict, measure::MappedWeight) =
        iscompatible(forward_map(dict),forward_map(measure)) && $f(superdict(dict), supermeasure(measure))
end



function innerproduct_native(d1::MappedDict, i, d2::MappedDict, j, measure::MappedWeight; options...)
    if iscompatible(d1,d2) && iscompatible(forward_map(d1),forward_map(measure))
        innerproduct(superdict(d1), i, superdict(d2), j, supermeasure(measure); options...)
    else
        innerproduct1(d1, i, d2, j, measure; options...)
    end
end

function gram(::Type{T}, dict::MappedDict, m::MappedWeight; options...) where {T}
    if iscompatible(forward_map(dict), forward_map(m))
        wrap_operator(dict, dict, gram(T, superdict(dict), supermeasure(m); options...))
    else
        default_gram(T, dict, m; options...)
    end
end

function gram(::Type{T}, dict::MappedDict, m::DiscreteWeight, grid::MappedGrid, weights; options...) where {T}
    if iscompatible(forward_map(grid), forward_map(dict))
        wrap_operator(dict, dict, gram(T, superdict(dict), supermeasure(m); options...))
    else
        default_gram(T, dict, m; options...)
    end
end



###############
# Transforms
###############

# We assume that maps do not affect a transform, as long as both the source and
# destination are mapped in the same way. If a source or destination grid is not
# mapped, we attempt to apply the inverse map to the grid and continue.
# For example, a mapped Fourier basis may have a PeriodicEquispacedGrid on a
# general interval. It is not necessarily a mapped grid.

transform_dict(s::MappedDict; options...) = mapped_dict(transform_dict(superdict(s); options...), forward_map(s))

function hasgrid_transform(dict::MappedDict, gb::GridBasis, grid::AbstractGrid)
    sgrid = unmap_grid(dict, grid)
    T = codomaintype(gb)
    hasgrid_transform(superdict(dict), GridBasis{T}(sgrid), sgrid)
end

function transform_from_grid(::Type{T}, s1::GridBasis, s2::MappedDict, grid; options...) where {T}
    sgrid = unmap_grid(s2, grid)
    op = transform_from_grid(T, GridBasis{T}(sgrid), superdict(s2), sgrid; options...)
    wrap_operator(s1, s2, op)
end

function transform_to_grid(::Type{T}, s1::MappedDict, s2::GridBasis, grid; options...) where {T}
    sgrid = unmap_grid(s1, grid)
    op = transform_to_grid(T, superdict(s1), GridBasis{T}(sgrid), sgrid; options...)
    wrap_operator(s1, s2, op)
end



###################
# Evaluation
###################


function evaluation(::Type{T}, dict::MappedDict, gb::GridBasis, grid; options...) where {T}
    sgrid = unmap_grid(dict, grid)
    op = evaluation(T, superdict(dict), GridBasis{T}(sgrid), sgrid; options...)
    wrap_operator(dict, gb, op)
end


###################
# Differentiation
###################

function derivative_dict(Φ::MappedDict, order; options...)
    map = forward_map(Φ)
    superdiff = similardictionary(Φ, derivative_dict(superdict(Φ), order; options...))
    if isaffine(map)
        superdiff
    elseif jacobian(map) isa ConstantMap
        superdiff
    else
        # TODO: this is only correct in 1D
        @assert order == 1
        mi = inverse(map)
        (t -> 1/jacdet(map, mi(t))) * superdiff
    end
end

function antiderivative_dict(Φ::MappedDict, order; options...)
    map = forward_map(Φ)
    if isaffine(map)
        similardictionary(Φ, antiderivative_dict(superdict(Φ), order; options...))
    elseif jacobian(map) isa ConstantMap
        similardictionary(Φ, antiderivative_dict(superdict(Φ), order; options...))
    else
        error("Don't know the antiderivative of a dictionary mapped by $(map)")
    end
end

# TODO: generalize to other orders
function differentiation(::Type{T}, dsrc::MappedDict1d, ddest::MappedDict1d, order::Int; options...) where {T}
    @assert isaffine(forward_map(dsrc))
    D = differentiation(T, superdict(dsrc), superdict(ddest), order; options...)
    S = ScalingOperator{T}(dest(D), jacobian(forward_map(dsrc),1)^(-order))
    wrap_operator(dsrc, ddest, S*D)
end

function differentiation(::Type{T}, dsrc::MappedDict1d, ddest::DerivedDict, order; options...) where {T}
    @assert order == 1
    @assert ddest isa WeightedDict1d
    @assert iscompatible(forward_map(dsrc), forward_map(superdict(ddest)))
    D = differentiation(T, superdict(dsrc), superdict(superdict(ddest)), order; options...)
    wrap_operator(dsrc, ddest, D)
end

function antidifferentiation(::Type{T}, dsrc::MappedDict1d, ddest::MappedDict1d, order::Int; options...) where {T}
    @assert isaffine(forward_map(dsrc))
    D = antidifferentiation(T, superdict(dsrc), superdict(ddest), order; options...)
    S = ScalingOperator{T}(dest(D), jacobian(forward_map(dsrc),1)^(order))
    wrap_operator(dsrc, ddest, S*D)
end


#################
# Special cases
#################

mapped_dict(dict::GridBasis, map) = GridBasis(map_grid(grid(s), map), coefficienttype(s))

"Rescale a function set to an interval [a,b]."
function rescale(s::Dictionary1d, a::Number, b::Number)
    T = promote_type(domaintype(s), typeof(a), typeof(b))
    if abs(a-infimum(support(s))) < 10eps(T) && abs(b-supremum(support(s))) < 10eps(T)
        s
    else
        m = mapto(support(s), T(a)..T(b))
        mapped_dict(s, m)
    end
end

rescale(s::Dictionary, d::AbstractInterval) = rescale(s, infimum(d), supremum(d))

"Map a dictionary to a domain"
(→)(Φ::Dictionary, domain::Domain) = rescale(Φ, domain)
# The symbol is \to

(∘)(Φ::Dictionary, map::AbstractMap) = mapped_dict(Φ, map)

# "Preserve Tensor Product Structure"
function rescale(s::TensorProductDict, a::SVector{N}, b::SVector{N}) where {N}
    scaled_sets = [ rescale(component(s,i), a[i], b[i]) for i in 1:N]
    tensorproduct(scaled_sets...)
end

plotgrid(S::MappedDict, n) = apply_map(plotgrid(superdict(S),n), forward_map(S))


#################
# Arithmetic
#################

function (*)(dict1::MappedDict, dict2::MappedDict, coef_src1, coef_src2)
    @assert iscompatible(superdict(dict1),superdict(dict2))
    mset,mcoef = (*)(superdict(dict1),superdict(dict2),coef_src1,coef_src2)
    mapped_dict(mset, forward_map(dict1)), mcoef
end


## Printing

name(dict::MappedDict) = "Mapped " * name(superdict(dict))

modifiersymbol(dict::MappedDict) = PrettyPrintSymbol{:M}(dict)

name(::PrettyPrintSymbol{:M}) = "Mapping"
string(s::PrettyPrintSymbol{:M}) = _string(s, s.object)
_string(s::PrettyPrintSymbol{:M}, dict::MappedDict) =
    "Mapping from $(support(superdict(dict))) to $(support(dict))"
