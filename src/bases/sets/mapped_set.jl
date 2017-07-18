# mapped_set.jl

"""
A MappedSet has a set and a map. The domain of the set is mapped to a different
one. Evaluating the MappedSet in a point uses the inverse map to evaluate the
underlying set in the corresponding point.
"""
struct MappedSet{S,M,T} <: DerivedSet{T}
    superset    ::  S
    map         ::  M

    function MappedSet{S,M,T}(set::FunctionSet{T}, map) where {S,M,T}
        new(set, map)
    end
end

const MappedSet1d{S,M,T <: Number} = MappedSet{S,M,T}

const MappedSpan{A, F <: MappedSet} = Span{A,F}
const MappedSpan1d{A, F <: MappedSet1d} = Span{A,F}

MappedSet(set::FunctionSet{T}, map::AbstractMap) where {T} =
    MappedSet{typeof(set),typeof(map),T}(set, map)

mapped_set(set::FunctionSet, map::AbstractMap) = MappedSet(set, map)

# Convenience function, similar to apply_map for grids etcetera
apply_map(set::FunctionSet, map) = mapped_set(set, map)

apply_map(set::MappedSet, map) = apply_map(superset(set), map*mapping(set))

apply_map(span::Span, map) = Span(apply_map(set(span), map), coeftype(span))

mapping(set::MappedSet) = set.map

similar_set(s::MappedSet, s2::FunctionSet) = MappedSet(s2, mapping(s))

has_derivative(s::MappedSet) = has_derivative(superset(s)) && islinear(mapping(s))
has_antiderivative(s::MappedSet) = has_antiderivative(superset(s)) && islinear(mapping(s))

grid(s::MappedSet) = _grid(s, superset(s), mapping(s))
_grid(s::MappedSet1d, set, map) = mapped_grid(grid(set), map)

for op in (:left, :right)
    @eval $op(s::MappedSet1d) = applymap( mapping(s), $op(superset(s)) )
    @eval $op(s::MappedSet1d, idx) = applymap( mapping(s), $op(superset(s), idx) )
end

name(s::MappedSet) = _name(s, superset(s), mapping(s))
_name(s::MappedSet, set, map) = "A mapped set based on " * name(set)
_name(s::MappedSet1d, set, map) = name(set) * ", mapped to [ $(left(s))  ,  $(right(s)) ]"

isreal(s::MappedSet) = isreal(superset(s)) && isreal(mapping(s))

eval_element(s::MappedSet, idx, y) = eval_element(superset(s), idx, apply_inverse(mapping(s),y))

eval_expansion(s::MappedSet, coef, y::Number) = eval_expansion(superset(s), coef, apply_inverse(mapping(s),y))

#eval_expansion(s::MappedSet, coef, grid::AbstractGrid) = eval_expansion(superset(s), coef, apply_map(grid, inv(mapping(s))))

in_support(set::MappedSet, idx, y) = in_support(superset(set), idx, apply_inverse(mapping(set), y))

is_compatible(s1::MappedSet, s2::MappedSet) = is_compatible(mapping(s1),mapping(s2)) && is_compatible(superset(s1),superset(s2))


###############
# Transforms
###############

# We assume that maps do not affect a transform, as long as both the source and
# destination are mapped in the same way. If a source or destination grid is not
# mapped, we attempt to apply the inverse map to the grid and continue.
# For example, a mapped Fourier basis may have a PeriodicEquispacedGrid on a
# general interval. It is not necessarily a mapped grid.

transform_space(s::MappedSpan; options...) = apply_map(transform_space(superspan(s); options...), mapping(s))

has_grid_transform(s::MappedSet, gs, g::MappedGrid) =
    is_compatible(mapping(s), mapping(g)) && has_transform(superset(s), gridset(supergrid(g)))

function has_grid_transform(s::MappedSet, gs, g::AbstractGrid)
    g2 = apply_map(g, inv(mapping(s)))
    has_grid_transform(superset(s), gridset(g2), g2)
end


function simplify_transform_pair(s::MappedSet, g::MappedGrid)
    if is_compatible(mapping(s), mapping(g))
        superset(s), supergrid(g)
    else
        s, g
    end
end

function simplify_transform_pair(s::MappedSet, g::AbstractGrid)
    g2 = apply_map(g, inv(mapping(s)))
    simplify_transform_pair(superset(s), g2)
end


###################
# Evaluation
###################

mapping(s::MappedSpan) = mapping(set(s))

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
# for subgrids and abstract FunctionSet's in generic/evaluation that causes an
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
mapped_set(s::MappedSet, map::AbstractMap) = MappedSet(superset(s), map*mapping(s))

mapped_set(s::DiscreteGridSpace, map::AbstractMap) = DiscreteGridSpace(mapped_grid(grid(s), map), eltype(s))

"Rescale a function set to an interval [a,b]."
function rescale(s::FunctionSet1d, a, b)
    T = domaintype(s)
    if abs(a-left(s)) < 10eps(T) && abs(b-right(s)) < 10eps(T)
        s
    else
        m = interval_map(left(s), right(s), T(a), T(b))
        apply_map(s, m)
    end
end

# "Preserve Tensor Product Structure"
function rescale{N}(s::TensorProductSet, a::SVector{N}, b::SVector{N})
    scaled_sets = [ rescale(element(s,i), a[i], b[i]) for i in 1:N]
    tensorproduct(scaled_sets...)
end

#################
# Arithmetic
#################


function (*)(s1::MappedSet, s2::MappedSet, coef_src1, coef_src2)
    @assert is_compatible(superset(s1),superset(s2))
    (mset,mcoef) = (*)(superset(s1),superset(s2),coef_src1, coef_src2)
    (MappedSet(mset, mapping(s1)), mcoef)
end

Gram(s::MappedSpan; options...) = wrap_operator(s, s, _gram(superspan(s), mapping(s); options...))

_gram(s::Span, map::AffineMap; options...) = jacobian(map, nothing)*Gram(s; options...)

dot(s::MappedSpan, f1::Function, f2::Function, nodes::Array=native_nodes(set(s)); options...) =
    _dot(superspan(s), mapping(s), f1, f2, nodes; options...)

_dot(s::Span1d, map::AffineMap, f1::Function, f2::Function, nodes::Array; options...) =
    jacobian(map, nothing)*dot(s, x->f1(applymap(map,x)), x->f2(applymap(map,x)), apply_inverse(map,nodes); options...)

native_nodes(s::MappedSet) = _native_nodes(superset(s), mapping(s))
_native_nodes(s::FunctionSet, map::AffineMap) = applymap(map, native_nodes(s))
