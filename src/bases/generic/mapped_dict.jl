# mapped_dict.jl

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

const MappedSpan{A,S,T,D <: MappedDict} = Span{A,S,T,D}
const MappedSpan1d{A,S,T,D <: MappedDict1d} = Span{A,S,T,D}

# In the constructor we check the domain and codomain types.
# The domain of the MappedDict is defined by the range of the map, because the
# domain of the underlying dict is mapped to the domain of the MappedDict.
# Hence, the domain type of the map has to equal the domain type of the dictionary.
# Confusingly, the domain and codomain types of dictionaries and Maps are
# defined in different order ({S,T} and {T,S}): below, parameter T1 has to equal.
MappedDict(dict::Dictionary{T1,T}, map::AbstractMap{S,T1}) where {S,T1,T} =
    MappedDict{typeof(dict),typeof(map),S,T}(dict, map)

# If the parameters don't match, we may have to promote the map.
# This does not (currently) work for all maps.
function MappedDict(dict::Dictionary{S1,T1}, map::AbstractMap{T2,S2}) where {S1,S2,T1,T2}
    S = promote_type(S1,S2)
    MappedDict(promote_domaintype(dict, S), update_eltype(map, S))
end

mapped_dict(dict::Dictionary, map::AbstractMap) = MappedDict(dict, map)

# Convenience function, similar to apply_map for grids etcetera
apply_map(dict::Dictionary, map) = mapped_dict(dict, map)

apply_map(dict::MappedDict, map) = apply_map(superdict(dict), map*mapping(dict))

apply_map(span::Span, map) = Span(apply_map(dictionary(span), map), coeftype(span))

mapping(dict::MappedDict) = dict.map

similar_dictionary(s::MappedDict, s2::Dictionary) = MappedDict(s2, mapping(s))

has_derivative(s::MappedDict) = has_derivative(superdict(s)) && islinear(mapping(s))
has_antiderivative(s::MappedDict) = has_antiderivative(superdict(s)) && islinear(mapping(s))

grid(s::MappedDict) = _grid(s, superdict(s), mapping(s))
_grid(s::MappedDict1d, set, map) = mapped_grid(grid(set), map)

for op in (:left, :right)
    @eval $op(s::MappedDict1d) = applymap( mapping(s), $op(superdict(s)) )
    @eval $op(s::MappedDict1d, idx) = applymap( mapping(s), $op(superdict(s), idx) )
end

name(s::MappedDict) = _name(s, superdict(s), mapping(s))
_name(s::MappedDict, set, map) = "A mapped set based on " * name(set)
_name(s::MappedDict1d, set, map) = name(set) * ", mapped to [ $(left(s))  ,  $(right(s)) ]"

isreal(s::MappedDict) = isreal(superdict(s)) && isreal(mapping(s))

eval_element(s::MappedDict, idx, y) = eval_element(superdict(s), idx, apply_inverse(mapping(s),y))

function eval_element_derivative(s::MappedDict1d, idx, y)
    x = apply_inverse(mapping(s), y)
    d = eval_element_derivative(superdict(s), idx, x)
    z = d / jacobian(mapping(s), y)
end

eval_expansion(s::MappedDict, coef, y::Number) = eval_expansion(superdict(s), coef, apply_inverse(mapping(s),y))

#eval_expansion(s::MappedDict, coef, grid::AbstractGrid) = eval_expansion(superdict(s), coef, apply_map(grid, inv(mapping(s))))

in_support(set::MappedDict, idx, y) = in_support(superdict(set), idx, apply_inverse(mapping(set), y))

is_compatible(s1::MappedDict, s2::MappedDict) = is_compatible(mapping(s1),mapping(s2)) && is_compatible(superdict(s1),superdict(s2))


###############
# Transforms
###############

# We assume that maps do not affect a transform, as long as both the source and
# destination are mapped in the same way. If a source or destination grid is not
# mapped, we attempt to apply the inverse map to the grid and continue.
# For example, a mapped Fourier basis may have a PeriodicEquispacedGrid on a
# general interval. It is not necessarily a mapped grid.

transform_space(s::MappedSpan; options...) = apply_map(transform_space(superspan(s); options...), mapping(s))

has_grid_transform(s::MappedDict, gb, g::MappedGrid) =
    is_compatible(mapping(s), mapping(g)) &&
        has_transform(superdict(s), gridbasis(supergrid(g), codomaintype(gb)))

function has_grid_transform(s::MappedDict, gb, g::AbstractGrid)
    g2 = apply_map(g, inv(mapping(s)))
    has_grid_transform(superdict(s), gridbasis(g2, codomaintype(gb)), g2)
end


function simplify_transform_pair(s::MappedDict, g::MappedGrid)
    if is_compatible(mapping(s), mapping(g))
        superdict(s), supergrid(g)
    else
        s, g
    end
end

function simplify_transform_pair(s::MappedDict, g::AbstractGrid)
    g2 = apply_map(g, inv(mapping(s)))
    simplify_transform_pair(superdict(s), g2)
end


###################
# Evaluation
###################

mapping(s::MappedSpan) = mapping(dictionary(s))

# If the set is mapped and the grid is mapped, and if the maps are identical,
# we can use the evaluation operator of the underlying set and grid
function grid_evaluation_operator(s::MappedSpan, dgs::DiscreteGridSpace, g::MappedGrid; options...)
    if is_compatible(mapping(s), mapping(g))
        E = evaluation_operator(superspan(s), supergrid(g); options...)
        wrap_operator(s, dgs, E)
    else
        default_evaluation_operator(s, dgs; options...)
    end
end

# If the grid is not mapped, we proceed by performing the inverse map on the grid,
# like we do for transforms above
function grid_evaluation_operator(s::MappedSpan, dgs::DiscreteGridSpace, g::AbstractGrid; options...)
    g2 = apply_map(g, inv(mapping(s)))
    E = evaluation_operator(superspan(s), gridspace(superspan(s), g2); options...)
    wrap_operator(s, dgs, E)
end

# We have to intercept the case of a subgrid, because there is a general rule
# for subgrids and abstract Dictionary's in generic/evaluation that causes an
# ambiguity. We proceed here by applying the inverse map to the underlying grid
# of the subgrid.
function grid_evaluation_operator(s::MappedSpan, dgs::DiscreteGridSpace, g::AbstractSubGrid; options...)
    mapped_supergrid = apply_map(supergrid(g), inv(mapping(s)))
    g2 = similar_subgrid(g, mapped_supergrid)
    g2_dgs = gridspace(superspan(s), g2)
    E = evaluation_operator(superspan(s), g2_dgs; options...)
    wrap_operator(s, dgs, E)
end

###################
# Differentiation
###################

for op in (:derivative_space, :antiderivative_space)
    @eval $op(s::MappedSpan1d, order::Int; options...) =
        (@assert islinear(mapping(s)); apply_map( $op(superspan(s), order; options...), mapping(s) ))
end

function differentiation_operator(s1::MappedSpan1d, s2::MappedSpan1d, order::Int; options...)
    @assert islinear(mapping(s1))
    D = differentiation_operator(superspan(s1), superspan(s2), order; options...)
    S = ScalingOperator(dest(D), jacobian(mapping(s1),1)^(-order))
    wrap_operator(s1, s2, S*D)
end

function antidifferentiation_operator(s1::MappedSpan1d, s2::MappedSpan1d, order::Int; options...)
    @assert islinear(mapping(s1))
    D = antidifferentiation_operator(superspan(s1), superspan(s2), order; options...)
    S = ScalingOperator(dest(D), jacobian(mapping(s1),1)^(order))
    wrap_operator(s1, s2, S*D)
end


#################
# Special cases
#################

# TODO: check for promotions here
mapped_dict(s::MappedDict, map::AbstractMap) = MappedDict(superdict(s), map*mapping(s))

mapped_dict(s::DiscreteGridSpace, map::AbstractMap) = DiscreteGridSpace(mapped_grid(grid(s), map), eltype(s))

"Rescale a function set to an interval [a,b]."
function rescale(s::Dictionary1d, a, b)
    T = domaintype(s)
    if abs(a-left(s)) < 10eps(T) && abs(b-right(s)) < 10eps(T)
        s
    else
        m = interval_map(left(s), right(s), T(a), T(b))
        apply_map(s, m)
    end
end

# "Preserve Tensor Product Structure"
function rescale{N}(s::TensorProductDict, a::SVector{N}, b::SVector{N})
    scaled_sets = [ rescale(element(s,i), a[i], b[i]) for i in 1:N]
    tensorproduct(scaled_sets...)
end

#################
# Arithmetic
#################


function (*)(s1::MappedDict, s2::MappedDict, coef_src1, coef_src2)
    @assert is_compatible(superdict(s1),superdict(s2))
    (mset,mcoef) = (*)(superdict(s1),superdict(s2),coef_src1, coef_src2)
    (MappedDict(mset, mapping(s1)), mcoef)
end

Gram(s::MappedSpan; options...) = wrap_operator(s, s, _gram(superspan(s), mapping(s); options...))

_gram(s::Span, map::AffineMap; options...) = jacobian(map, nothing)*Gram(s; options...)

dot(s::MappedSpan, f1::Function, f2::Function, nodes::Array=native_nodes(dictionary(s)); options...) =
    _dot(superspan(s), mapping(s), f1, f2, nodes; options...)

_dot(s::Span1d, map::AffineMap, f1::Function, f2::Function, nodes::Array; options...) =
    jacobian(map, nothing)*dot(s, x->f1(applymap(map,x)), x->f2(applymap(map,x)), apply_inverse(map,nodes); options...)

native_nodes(s::MappedDict) = _native_nodes(superdict(s), mapping(s))
_native_nodes(s::Dictionary, map::AffineMap) = applymap(map, native_nodes(s))
