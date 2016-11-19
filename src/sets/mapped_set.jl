# mapped_set.jl

"""
A MappedSet has a set and a map. The domain of the set is mapped to a different
one. Evaluating the MappedSet in a point uses the inverse map to evaluate the
underlying set in the corresponding point.
"""
immutable MappedSet{S,M,N,T} <: DerivedSet{N,T}
    set     ::  S
    map     ::  M
end

typealias MappedSet1d{S,M,T} MappedSet{S,M,1,T}

MappedSet{N,T}(set::FunctionSet{N,T}, map::AbstractMap) =
    MappedSet{typeof(set),typeof(map),N,T}(set, map)

mapped_set(set::FunctionSet, map::AbstractMap) = MappedSet(set, map)

mapping(set::MappedSet) = set.map

similar_set(s::MappedSet, s2::FunctionSet) = MappedSet(s2, mapping(s))

has_derivative(s::MappedSet) = has_derivative(set(s)) && is_linear(mapping(s))
has_antiderivative(s::MappedSet) = false

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
# destination are mapped in the same way.
# We have to wrap the operators appropriately though.

# Checks for compatibility of maps are not yet included below

for op in (:transform_operator,)
    # Both sets are mapped: undo the maps
    @eval $op(s1::MappedSet, s2::MappedSet; options...) =
        wrap_operator(s1, s2, $op(set(s1), set(s2); options...) )
    # The destination is not mapped: try to apply the inverse map to it.
    @eval $op(s1::MappedSet, s2::DiscreteGridSpace; options...) =
        wrap_operator(s1, s2, $op(set(s1), mapped_set(s2, inv(mapping(s1))) ; options...) )
    # The source is not mapped: try to apply the inverse map to it.
    @eval $op(s1::DiscreteGridSpace, s2::MappedSet; options...) =
        wrap_operator(s1, s2, $op(mapped_set(s1, inv(mapping(s2))), set(s2); options...) )
end

# We have to do these by hand, because the spaces are not the same: s1 is source and destination
# of the transform_pre_operator. The post operation only acts on s2.
transform_pre_operator(s1::MappedSet, s2::MappedSet; options...) =
    wrap_operator(s1, s1, transform_pre_operator(set(s1), set(s2); options...))
transform_pre_operator(s1::MappedSet, s2::DiscreteGridSpace; options...) =
    wrap_operator(s1, s1, transform_pre_operator(set(s1), mapped_set(s2, inv(mapping(s1))); options...) )
transform_pre_operator(s1::DiscreteGridSpace, s2::MappedSet; options...) =
    wrap_operator(s1, s1, transform_pre_operator(mapped_set(s1, inv(mapping(s2))), set(s2); options...))

transform_post_operator(s1::MappedSet, s2::MappedSet; options...) =
    wrap_operator(s2, s2, transform_post_operator(set(s1), set(s2); options...))
transform_post_operator(s1::MappedSet, s2::DiscreteGridSpace; options...) =
    wrap_operator(s2, s2, transform_post_operator(set(s1), mapped_set(s2, inv(mapping(s1))); options...))
transform_post_operator(s1::DiscreteGridSpace, s2::MappedSet; options...) =
    wrap_operator(s2, s2, transform_post_operator(mapped_set(s1, inv(mapping(s2))), set(s2); options...))


###################
# Differentiation
###################

for op in (:derivative_set, :antiderivative_set)
    @eval $op(s::MappedSet1d, order::Int; options...) = mapped_set( $op(set(s), order; options...), mapping(s) )
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

mapped_set(s::MappedSet, map::AbstractMap) = MappedSet(set(s), map*mapping(s))

mapped_set(s::DiscreteGridSpace, map::AbstractMap) = DiscreteGridSpace(mapped_grid(grid(s), map))

"Rescale a function set to an interval [a,b]."
function rescale(s::FunctionSet1d, a, b)
    T = numtype(s)
    if abs(a-left(s)) < 10eps(T) && abs(b-right(s)) < 10eps(T)
        s
    else
        m = interval_map(left(s), right(s), a, b)
        mapped_set(s, m)
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
