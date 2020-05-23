
export →,
    rescale,
    MappedDict,
    mapped_dict,
    mapping

"""
A `MappedDict` has a dictionary and a map. The domain of the dictionary is
mapped to a different one. Evaluating the `MappedDict` in a point uses the
inverse map to evaluate the underlying dictionary element in the corresponding
point.
"""
struct MappedDict{D,M,S,T} <: DerivedDict{S,T}
    superdict   ::  D
    map         ::  M
end

const MappedDict1d{D,M,S <: Number,T <: Number} = MappedDict{D,M,S,T}
const MappedDict2d{D,M,S <: SVector{2},T <: Number} = MappedDict{D,M,S,T}

# In the constructor we check the domain and codomain types.
# The domain of the MappedDict is defined by the range of the map, because the
# domain of the underlying dict is mapped to the domain of the MappedDict.
# Hence, the domain type of the map has to equal the domain type of the dictionary.
MappedDict(dict::Dictionary{T1,T}, map::Map{T1}) where {T1,T} =
    MappedDict{typeof(dict),typeof(map),codomaintype(map,T1),T}(dict, map)

MappedDict(dict::Dictionary{S1,T1}, map::Map{S2}) where {S1,S2,T1} =
    MappedDict(dict, convert(Map{S1}, map))

mapped_dict(dict::Dictionary, map::AbstractMap) = MappedDict(dict, map)

# Convenience function, similar to apply_map for grids etcetera
apply_map(dict::Dictionary, map) = mapped_dict(dict, map)

apply_map(dict::MappedDict, map) = apply_map(superdict(dict), map ∘ mapping(dict))

mapping(dict::MappedDict) = dict.map

similardictionary(s::MappedDict, s2::Dictionary) = MappedDict(s2, mapping(s))

hasderivative(dict::MappedDict) = false
hasantiderivative(dict::MappedDict) = false

hasderivative(dict::MappedDict1d) =
    hasderivative(superdict(dict)) && isaffine(mapping(dict))
hasantiderivative(dict::MappedDict1d) =
    hasantiderivative(superdict(dict)) && isaffine(mapping(dict))

interpolation_grid(dict::MappedDict) = _grid(dict, superdict(dict), mapping(dict))
_grid(dict::MappedDict, sdict, map) = mapped_grid(interpolation_grid(sdict), map)


function unmap_grid(dict::MappedDict, grid::MappedGrid)
    if iscompatible(mapping(dict), mapping(grid))
        supergrid(grid)
    else
        apply_map(grid, inv(mapping(dict)))
    end
end

unmap_grid(dict::MappedDict, grid::AbstractGrid) = apply_map(grid, inv(mapping(dict)))


isreal(s::MappedDict) = isreal(superdict(s)) && isreal(mapping(s))

unsafe_eval_element(s::MappedDict, idx, y) =
    unsafe_eval_element(superdict(s), idx, leftinverse(mapping(s), y))

function unsafe_eval_element_derivative(s::MappedDict1d, idx, y, order)
    @assert order == 1
    x = leftinverse(mapping(s), y)
    d = unsafe_eval_element_derivative(superdict(s), idx, x, order)
    z = d / jacdet(mapping(s), y)
end

function unsafe_eval_element_derivative(dict::MappedDict, idx, y, order)
    @assert maximum(order) <= 1
    m = mapping(dict)
    x = leftinverse(m, y)
    J = jacobian(m, y)
    g = eval_gradient(superdict(dict), idx, x)
    sum(order .* (J' \ g))
end

function eval_expansion(s::MappedDict{D,M,S,T}, coef, y::S) where {D,M,S,T}
    if in_support(s, first(eachindex(s)), y)
        eval_expansion(superdict(s), coef, leftinverse(mapping(s), y))
    else
        zero(codomaintype(s))
    end
end

support(dict::MappedDict) = DomainSets.map_domain(mapping(dict), support(superdict(dict)))

support(dict::MappedDict, idx) = mapping(dict).(support(superdict(dict), idx))

function dict_in_support(set::MappedDict, idx, y, threshold = default_threshold(y))
    x = leftinverse(mapping(set), y)
    y1 = mapping(set)(x)
    if norm(y-y1) < threshold
        in_support(superdict(set), idx, x)
    else
        false
    end
end

iscompatible(s1::MappedDict, s2::MappedDict) =
    iscompatible(mapping(s1),mapping(s2)) && iscompatible(superdict(s1),superdict(s2))

measure(dict::MappedDict) = apply_map(measure(superdict(dict)), mapping(dict))

hasmeasure(dict::MappedDict) = hasmeasure(superdict(dict))

for f in (:isorthonormal, :isorthogonal, :isbiorthogonal)
    @eval $f(dict::MappedDict, measure::MappedMeasure) =
        iscompatible(mapping(dict),mapping(measure)) && $f(superdict(dict), supermeasure(measure))
end



function innerproduct_native(d1::MappedDict, i, d2::MappedDict, j, measure::MappedMeasure; options...)
    if iscompatible(d1,d2) && iscompatible(mapping(d1),mapping(measure))
        innerproduct(superdict(d1), i, superdict(d2), j, supermeasure(measure); options...)
    else
        innerproduct1(d1, i, d2, j, measure; options...)
    end
end

function gram(::Type{T}, dict::MappedDict, m::MappedMeasure; options...) where {T}
    if iscompatible(mapping(dict), mapping(m))
        wrap_operator(dict, dict, gram(T, superdict(dict), supermeasure(m); options...))
    else
        default_gram(T, dict, m; options...)
    end
end

function gram(::Type{T}, dict::MappedDict, m::DiscreteMeasure, grid::MappedGrid, weights; options...) where {T}
    if iscompatible(mapping(grid), mapping(dict))
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

transform_dict(s::MappedDict; options...) = apply_map(transform_dict(superdict(s); options...), mapping(s))

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
    map = mapping(Φ)
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
    map = mapping(Φ)
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
    @assert isaffine(mapping(dsrc))
    D = differentiation(T, superdict(dsrc), superdict(ddest), order; options...)
    S = ScalingOperator{T}(dest(D), jacobian(mapping(dsrc),1)^(-order))
    wrap_operator(dsrc, ddest, S*D)
end

function differentiation(::Type{T}, dsrc::MappedDict1d, ddest::DerivedDict, order; options...) where {T}
    @assert order == 1
    @assert ddest isa WeightedDict1d
    @assert iscompatible(mapping(dsrc), mapping(superdict(ddest)))
    D = differentiation(T, superdict(dsrc), superdict(superdict(ddest)), order; options...)
    wrap_operator(dsrc, ddest, D)
end

function antidifferentiation(::Type{T}, dsrc::MappedDict1d, ddest::MappedDict1d, order::Int; options...) where {T}
    @assert isaffine(mapping(dsrc))
    D = antidifferentiation(T, superdict(dsrc), superdict(ddest), order; options...)
    S = ScalingOperator{T}(dest(D), jacobian(mapping(dsrc),1)^(order))
    wrap_operator(dsrc, ddest, S*D)
end


#################
# Special cases
#################

# TODO: check for promotions here
mapped_dict(s::MappedDict, map::AbstractMap) = MappedDict(superdict(s), map ∘ mapping(s))

mapped_dict(s::GridBasis, map::AbstractMap) = GridBasis(mapped_grid(grid(s), map), coefficienttype(s))

"Rescale a function set to an interval [a,b]."
function rescale(s::Dictionary1d, a::Number, b::Number)
    T = promote_type(domaintype(s), typeof(a), typeof(b))
    if abs(a-infimum(support(s))) < 10eps(T) && abs(b-supremum(support(s))) < 10eps(T)
        s
    else
        m = interval_map(infimum(support(s)), supremum(support(s)), T(a), T(b))
        apply_map(s, m)
    end
end

rescale(s::Dictionary, d::AbstractInterval) = rescale(s, infimum(d), supremum(d))

"Map a dictionary to a domain"
(→)(Φ::Dictionary, domain::Domain) = rescale(Φ, domain)
# The symbol is \to

(∘)(Φ::Dictionary, map::AbstractMap) = mapped_dict(Φ, map)

# "Preserve Tensor Product Structure"
function rescale(s::TensorProductDict, a::SVector{N}, b::SVector{N}) where {N}
    scaled_sets = [ rescale(element(s,i), a[i], b[i]) for i in 1:N]
    tensorproduct(scaled_sets...)
end

plotgrid(S::MappedDict, n) = apply_map(plotgrid(superdict(S),n), mapping(S))


#################
# Arithmetic
#################

function (*)(s1::MappedDict, s2::MappedDict, coef_src1, coef_src2)
    @assert iscompatible(superdict(s1),superdict(s2))
    (mset,mcoef) = (*)(superdict(s1),superdict(s2),coef_src1, coef_src2)
    (MappedDict(mset, mapping(s1)), mcoef)
end


## Printing

name(dict::MappedDict) = "Mapped " * name(superdict(dict))

modifiersymbol(dict::MappedDict) = PrettyPrintSymbol{:M}(dict)

name(::PrettyPrintSymbol{:M}) = "Mapping"
string(s::PrettyPrintSymbol{:M}) = _string(s, s.object)
_string(s::PrettyPrintSymbol{:M}, dict::MappedDict) =
    "Mapping from $(support(superdict(dict))) to $(support(dict))"
