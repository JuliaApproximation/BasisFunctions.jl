# multiple_set.jl


"""
A MultiSet is the concatenation of several function sets. The function sets
may be the same (but scaled to different intervals, say) or they can be different.
The sets are contained in an indexable set, such as a tuple or an array. In case
of an array, the number of sets may be large.

The native representation of a MultiSet is a MultiArray, of which each element
is the native representation of the corresponding element of the multiset.
"""
immutable MultiSet{SETS,N,T} <: CompositeSet{N,T}
    sets    ::  SETS
    offsets ::  Array{Int,1}

    function MultiSet(sets)
        offsets = compute_offsets(sets)
        new(sets, offsets)
    end
end

# Is this constructor type-stable? Probably not, even if T is given, because
# of the use of ndims below.
function MultiSet(sets, T)
    for set in sets
        # Is this the right check here?
        @assert promote_type(eltype(set), T) == T
    end
    MultiSet{typeof(sets),ndims(sets[1]),T}(sets)
end

function MultiSet(sets)
    T = reduce(promote_type, map(eltype, sets))
    MultiSet(map(s->promote_eltype(s,T), sets), T)
end


similar_set(set::MultiSet, sets, T = eltype(set)) = MultiSet(sets, T)

multiset(set::FunctionSet) = set

# When manipulating multisets, we create Array's of sets by default
multiset(s1::FunctionSet, s2::FunctionSet) = MultiSet([s1,s2])
multiset(s1::MultiSet, s2::MultiSet) = MultiSet(vcat(elements(s1), elements(s2)))
multiset(s1::MultiSet, s2::FunctionSet) = MultiSet(vcat(elements(s1), s2))
multiset(s1::FunctionSet, s2::MultiSet) = MultiSet(vcat(s1, elements(s2)))

multiset(s1::FunctionSet, s2::FunctionSet, s3::FunctionSet...) =
    multiset(multiset(s1,s2), s3...)

function multiset(sets::AbstractArray)
    if length(sets) == 1
        multiset(sets[1])
    else
        MultiSet(flatten(MultiSet, sets, FunctionSet{ndims(sets[1]),eltype(sets[1])}))
    end
end

# Perhaps we don't want this behaviour, that [b;b] creates a MultiSet, rather
# than an array of FunctionSet's
# vcat(s1::FunctionSet, s2::FunctionSet) = multiset(s1,s2)

⊕(s1::FunctionSet, s2::FunctionSet) = multiset(s1, s2)

name(s::MultiSet) = "A set consisting of $(composite_length(s)) sets"

for op in (:is_orthogonal, :is_biorthogonal, :is_basis, :is_frame)
    # Redirect the calls to multiple_is_basis with the elements as extra arguments,
    # and that method can decide whether the property holds for the multiset.
    fname = Symbol("multiple_$(op)")
    @eval $op(s::MultiSet) = ($fname)(s, elements(s)...)
    # By default, multisets do not have these properties:
    @eval ($fname)(s, elements...) = false
end

for op in (:has_grid, :has_transform)
    fname = Symbol("multiple_$(op)")
    @eval $op(s::MultiSet) = ($fname)(s, elements(s)...)
    @eval ($fname)(s, elements...) = false
end


# Try to return ranges of an underlying set, if possible
function subset(s::MultiSet, idx::OrdinalRange{Int})
    i1 = multilinear_index(s, first(idx))
    i2 = multilinear_index(s, last(idx))
    # Check whether the range lies fully in one set
    if i1[1] == i2[1]
        subset(element(s, i1[1]), i1[2]:step(idx):i2[2])
    else
        FunctionSubSet(s, idx)
    end
end



for op in [:left, :right, :moment, :norm]
    @eval $op(set::MultiSet, idx::Int) = $op(set, multilinear_index(set, idx))
    # Pass along a linear or a native index to the subset
    @eval function $op(set::MultiSet, idx::Union{MultiLinearIndex,Tuple{Int,Any}})
        i,j = idx
        $op(set.sets[i], j)
    end
end

left(set::MultiSet) = minimum(map(left, elements(set)))
right(set::MultiSet) = maximum(map(right, elements(set)))


## Differentiation

# derivative_set(s::MultiSet, order; options...) =
#     MultiSet(map(b-> derivative_set(b, order; options...), elements(s)))
#
# antiderivative_set(s::MultiSet, order; options...) =
#     MultiSet(map(b-> antiderivative_set(b, order; options...), elements(s)))

for op in [:differentiation_operator, :antidifferentiation_operator]
    @eval function $op(s1::MultiSet, s2::MultiSet, order; options...)
        if composite_length(s1) == composite_length(s2)
            BlockDiagonalOperator(AbstractOperator{eltype(s1)}[$op(element(s1,i), element(s2, i), order; options...) for i in 1:composite_length(s1)], s1, s2)
        else
            # We have a situation because the sizes of the multisets don't match.
            # The derivative set may have been a nested multiset that was flattened. This
            # case occurs for example in multisets involving an AugmentedSet, because the
            # derivative of a single AugmentedSet may be a MultiSet.
            # The problem is we don't know which elements of s1 to match with which elements of s2.
            # Resolve the situation by looking at the standard derivative sets of each element of s1.
            # This may not be correct if one of the elements has multiple derivative sets, and
            # the user had chosen a non-standard one.
            ops = AbstractOperator{eltype(s1)}[$op(el; options...) for el in elements(s1)]
            BlockDiagonalOperator(ops, s1, s2)
        end
    end
end

evaluation_operator(set::MultiSet, dgs::DiscreteGridSpace; options...) =
    block_row_operator( AbstractOperator{eltype(set)}[evaluation_operator(el,dgs; options...) for el in elements(set)], set, dgs)

## Rescaling

rescale(s::MultiSet, a, b) = multiset(map( t-> rescale(t, a, b), elements(s)))
