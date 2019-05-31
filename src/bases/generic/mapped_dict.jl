
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

# In the constructor we check the domain and codomain types.
# The domain of the MappedDict is defined by the range of the map, because the
# domain of the underlying dict is mapped to the domain of the MappedDict.
# Hence, the domain type of the map has to equal the domain type of the dictionary.
MappedDict(dict::Dictionary{T1,T}, map::AbstractMap{T1,S}) where {S,T1,T} =
    MappedDict{typeof(dict),typeof(map),S,T}(dict, map)

# If the parameters don't match, we may have to promote the map.
# This does not (currently) work for all maps.
function MappedDict(dict::Dictionary{S1,T1}, map::AbstractMap{S2,T2}) where {S1,S2,T1,T2}
    S = promote_type(S1,S2)
    MappedDict(promote_domaintype(dict, S), DomainSets.update_eltype(map, S))
end

mapped_dict(dict::Dictionary, map::AbstractMap) = MappedDict(dict, map)

# Convenience function, similar to apply_map for grids etcetera
apply_map(dict::Dictionary, map) = mapped_dict(dict, map)

apply_map(dict::MappedDict, map) = apply_map(superdict(dict), map ∘ mapping(dict))

mapping(dict::MappedDict) = dict.map

similar_dictionary(s::MappedDict, s2::Dictionary) = MappedDict(s2, mapping(s))

hasderivative(dict::MappedDict) =
    hasderivative(superdict(dict)) && islinear(mapping(dict))
hasantiderivative(dict::MappedDict) =
    hasantiderivative(superdict(dict)) && islinear(mapping(dict))

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
    unsafe_eval_element(superdict(s), idx, apply_left_inverse(mapping(s),y))

function unsafe_eval_element_derivative(s::MappedDict1d, idx, y)
    x = apply_left_inverse(mapping(s), y)
    d = unsafe_eval_element_derivative(superdict(s), idx, x)
    z = d / jacobian(mapping(s), y)
end

function eval_expansion(s::MappedDict{D,M,S,T}, coef, y::S) where {D,M,S,T}
    if in_support(s, first(eachindex(s)), y)
        eval_expansion(superdict(s), coef, apply_left_inverse(mapping(s),y))
    else
        zero(codomaintype(s))
    end
end

support(dict::MappedDict) = mapping(dict)*support(superdict(dict))

support(dict::MappedDict, idx) = mapping(dict)*support(superdict(dict), idx)

function dict_in_support(set::MappedDict, idx, y, threshold = default_threshold(y))
    x = apply_left_inverse(mapping(set), y)
    y1 = applymap(mapping(set), x)
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

function gramoperator(dict::MappedDict, m::MappedMeasure; T = coefficienttype(dict), options...)
    if iscompatible(mapping(dict), mapping(m))
        wrap_operator(dict, dict, gramoperator(superdict(dict), supermeasure(m); T=T, options...))
    else
        default_gramoperator(dict, m; T=T, options...)
    end
end

function gramoperator(dict::MappedDict, m::DiscreteMeasure, grid::MappedGrid, weights; T = coefficienttype(dict), options...)
    if iscompatible(mapping(grid), mapping(dict))
        wrap_operator(dict, dict, gramoperator(superdict(dict), supermeasure(m); T=T, options...))
    else
        default_gramoperator(dict, m; T=T, options...)
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

function transform_from_grid(s1::GridBasis, s2::MappedDict, grid; T = op_eltype(s1,s2), options...)
    sgrid = unmap_grid(s2, grid)
    op = transform_from_grid(GridBasis{T}(sgrid), superdict(s2), sgrid; T=T, options...)
    wrap_operator(s1, s2, op)
end

function transform_to_grid(s1::MappedDict, s2::GridBasis, grid; T = op_eltype(s1,s2), options...)
    sgrid = unmap_grid(s1, grid)
    op = transform_to_grid(superdict(s1), GridBasis{T}(sgrid), sgrid; T=T, options...)
    wrap_operator(s1, s2, op)
end



###################
# Evaluation
###################


function grid_evaluation_operator(dict::MappedDict, gb::GridBasis, grid; T = op_eltype(dict, gb), options...)
    sgrid = unmap_grid(dict, grid)
    op = grid_evaluation_operator(superdict(dict), GridBasis{T}(sgrid), sgrid; T=T, options...)
    wrap_operator(dict, gb, op)
end


###################
# Differentiation
###################

for op in (:derivative_dict, :antiderivative_dict)
    @eval $op(s::MappedDict1d, order::Int; options...) =
        (@assert islinear(mapping(s)); apply_map( $op(superdict(s), order; options...), mapping(s) ))
end

function differentiation_operator(s1::MappedDict1d, s2::MappedDict1d, order; T=op_eltype(s1,s2), options...)
    @assert islinear(mapping(s1))
    D = differentiation_operator(superdict(s1), superdict(s2), order; T=T, options...)
    S = ScalingOperator(dest(D), jacobian(mapping(s1),1)^(-order); T=T)
    wrap_operator(s1, s2, S*D)
end

function antidifferentiation_operator(s1::MappedDict1d, s2::MappedDict1d, order; T=op_eltype(s1,s2), options...)
    @assert islinear(mapping(s1))
    D = antidifferentiation_operator(superdict(s1), superdict(s2), order; T=T, options...)
    S = ScalingOperator(dest(D), jacobian(mapping(s1),1)^(order); T=T)
    wrap_operator(s1, s2, S*D)
end


#################
# Special cases
#################

# TODO: check for promotions here
mapped_dict(s::MappedDict, map::AbstractMap) = MappedDict(superdict(s), map*mapping(s))

mapped_dict(s::GridBasis, map::AbstractMap) = GridBasis(mapped_grid(grid(s), map), coefficienttype(s))

"Rescale a function set to an interval [a,b]."
function rescale(s::Dictionary1d, a, b)
    T = domaintype(s)
    if abs(a-infimum(support(s))) < 10eps(T) && abs(b-supremum(support(s))) < 10eps(T)
        s
    else
        m = interval_map(infimum(support(s)), supremum(support(s)), T(a), T(b))
        apply_map(s, m)
    end
end

rescale(s::Dictionary, d::Domain) = rescale(s, infimum(d), supremum(d))


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
