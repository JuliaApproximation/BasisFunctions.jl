# mapped_set.jl

"""
A MappedSet has a set and a map. The domain of the set is mapped to a different
one. Evaluating the MappedSet in a point uses the inverse map to evaluate the
underlying set in the corresponding point.
"""
immutable MappedSet{S,M,N,T} <: DerivedSet{N,T}
    set     ::  S
    map     ::  M

    function MappedSet(set::FunctionSet{N,T}, map)
        new(set, map)
    end
end

typealias MappedSet1d{S,M,T} MappedSet{S,M,1,T}

MappedSet{N,T}(set::FunctionSet{N,T}, map::AbstractMap) =
    MappedSet{typeof(set),typeof(map),N,T}(set, map)

mapped_set(set::FunctionSet, map::AbstractMap) = MappedSet(set, map)

# Convenience function, similar to apply_map for grids etcetera
apply_map(set::FunctionSet, map) = mapped_set(set, map)

mapping(set::MappedSet) = set.map

similar_set(s::MappedSet, s2::FunctionSet) = MappedSet(s2, mapping(s))

has_derivative(s::MappedSet) = has_derivative(set(s)) && is_linear(mapping(s))
has_antiderivative(s::MappedSet) = has_antiderivative(set(s)) && is_linear(mapping(s))

grid(s::MappedSet) = _grid(s, set(s), mapping(s))
_grid(s::MappedSet1d, set, map) = mapped_grid(grid(set), map)

for op in (:left, :right)
    @eval $op(s::MappedSet1d) = forward_map( mapping(s), $op(set(s)) )
    @eval $op(s::MappedSet1d, idx) = forward_map( mapping(s), $op(set(s), idx) )
end

name(s::MappedSet) = _name(s, set(s), mapping(s))
_name(s::MappedSet, set, map) = "A mapped set based on " * name(set)
_name(s::MappedSet1d, set, map) = name(set) * ", mapped to [ $(left(s))  ,  $(right(s)) ]"

isreal(s::MappedSet) = isreal(set(s)) && isreal(mapping(s))

eval_element(s::MappedSet, idx, y) = eval_element(set(s), idx, inverse_map(mapping(s),y))

is_compatible(s1::MappedSet, s2::MappedSet) = is_compatible(mapping(s1),mapping(s2)) && is_compatible(set(s1),set(s2))


###############
# Transforms
###############

# We assume that maps do not affect a transform, as long as both the source and
# destination are mapped in the same way. If a source or destination grid is not
# mapped, we attempt to apply the inverse map to the grid and continue.
# For example, a mapped Fourier basis may have a PeriodicEquispacedGrid on a
# general interval. It is not necessarily a mapped grid.

transform_set(s::MappedSet; options...) = apply_map(transform_set(set(s); options...), mapping(s))

has_grid_transform(s::MappedSet, dgs, g::MappedGrid) =
    is_compatible(mapping(s), mapping(g)) && has_transform(set(s), DiscreteGridSpace(grid(g), eltype(dgs)))

function has_grid_transform(s::MappedSet, dgs, g::AbstractGrid)
    g2 = apply_map(g, inv(mapping(s)))
    has_grid_transform(set(s), DiscreteGridSpace(g2, eltype(dgs)), g2)
end


function simplify_transform_pair(s::MappedSet, g::MappedGrid)
    if is_compatible(mapping(s), mapping(g))
        set(s), grid(g)
    else
        s, g
    end
end

function simplify_transform_pair(s::MappedSet, g::AbstractGrid)
    g2 = apply_map(g, inv(mapping(s)))
    simplify_transform_pair(set(s), g2)
end


###################
# Differentiation
###################

for op in (:derivative_set, :antiderivative_set)
    @eval $op(s::MappedSet1d, order::Int; options...) =
        (@assert is_linear(mapping(s)); apply_map( $op(set(s), order; options...), mapping(s) ))
end

function differentiation_operator(s1::MappedSet1d, s2::MappedSet1d, order::Int; options...)
    @assert is_linear(mapping(s1))
    D = differentiation_operator(set(s1), set(s2), order; options...)
    S = ScalingOperator(dest(D), jacobian(mapping(s1),1)^(-order))
    wrap_operator( s1, s2, S*D )
end

function antidifferentiation_operator(s1::MappedSet1d, s2::MappedSet1d, order::Int; options...)
    @assert is_linear(mapping(s1))
    D = antidifferentiation_operator(set(s1), set(s2), order; options...)
    S = ScalingOperator(dest(D), jacobian(mapping(s1),1)^(order))
    wrap_operator( s1, s2, S*D )
end


#################
# Special cases
#################

# TODO: check for promotions here
mapped_set(s::MappedSet, map::AbstractMap) = MappedSet(set(s), map*mapping(s))

mapped_set(s::DiscreteGridSpace, map::AbstractMap) = DiscreteGridSpace(mapped_grid(grid(s), map), eltype(s))

"Rescale a function set to an interval [a,b]."
function rescale(s::FunctionSet1d, a, b)
    T = numtype(s)
    if abs(a-left(s)) < 10eps(T) && abs(b-right(s)) < 10eps(T)
        s
    else
        m = interval_map(left(s), right(s), a, b)
        apply_map(s, m)
    end
end

#################
# Arithmetic
#################


function (*)(s1::MappedSet, s2::MappedSet, coef_src1, coef_src2)
    @assert is_compatible(set(s1),set(s2))
    (mset,mcoef) = (*)(set(s1),set(s2),coef_src1, coef_src2)
    (MappedSet(mset, mapping(s1)), mcoef)
end
