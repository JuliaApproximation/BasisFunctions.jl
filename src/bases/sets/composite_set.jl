# composite_set.jl

"""
CompositeSet is the abstract supertype of a function set that consists of several
sets. The CompositeSet type defines common routines for indexing and iteration.

The representation of a CompositeSet is a MultiArray. The outer array of this
MultiArray adopts the structure of the elements of the CompositeSet: if the elements
are stored in a tuple, the outer array will be a tuple. If the elements are
stored in an array, the outer array will be an array as well.

CompositeSet's that store elements in an array can have a large number of elements.
However, for good efficiency they should have the same type. Tuples are suitable
to hold mixed-type elements with good efficiency, however in this case there can't
be too many.

The concrete subtypes differ in what evaluation means. Examples include:
- MultiSet, where evaluation is the sum of the evaluation of the subsets
- VectorSet, where evaluation is a vector (or tuple) of the evaluations of the
subsets
- PiecewiseSet, where evaluation depends on the location of the evaluation point
"""
# The type parameter S is the rangetype.
abstract type CompositeSet{S,T} <: FunctionSet{T}
end

const CompositeSetSpan{A,F <: CompositeSet} = Span{A,F}

# We assume that every subset has an indexable field called sets
is_composite(set::CompositeSet) = true
elements(set::CompositeSet) = set.sets
element(set::CompositeSet, j::Int) = set.sets[j]

similar_compositespan(s::CompositeSetSpan, spans) =
    Span(similar_set(set(s), map(set, spans)), coeftype(s))

# TODO: think about these and verify correctness. Maybe the types depend on the
# nature of the composite set.
rangetype(::Type{CompositeSet{S,T}}) where {S,T} = S

# For a generic implementation of range indexing, we need a 'similar_set' function
# to create a new set of the same type as the given set.
element(set::CompositeSet, range::Range) = similar_set(set, set.sets[range])

tail(set::CompositeSet) = nb_elements(set) == 2 ? element(set, 2) : element(set, 2:nb_elements(set))

# We compute offsets of the individual sets using a cumulative sum
compute_offsets(sets::Array) = [0; cumsum(map(length, sets))]

# Convert a tuple to an array in order to use cumsum like above
compute_offsets(sets::NTuple{N,Any}) where {N} = compute_offsets([set for set in sets])

# Implement equality in terms of equality of the elements.
==(s1::CompositeSet, s2::CompositeSet) = (elements(s1) == elements(s2))

length(s::CompositeSet) = s.offsets[end]

length(s::CompositeSet, i::Int) = length(element(s,i))

# Concrete subtypes should override similar_set and call their own constructor

# Using map in the definitions below ensures that a tuple is created when
# elements(set) is a tuple, and an array when elements(set) is an array.
resize(set::CompositeSet, n) =
    similar_set(set, map( (s,l) -> resize(s, l), elements(set), n))

set_promote_domaintype(set::CompositeSet, ::Type{S}) where {S} =
    similar_set(set, map(s->set_promote_domaintype(s, S), elements(set)))

zeros(::Type{T}, set::CompositeSet) where {T} = MultiArray(map(s->zeros(T,s),elements(set)))

for op in (:isreal, )
    @eval $op(set::CompositeSet) = reduce($op, elements(set))
end

for op in (:has_derivative, :has_antiderivative, :has_extension)
    @eval $op(set::CompositeSet) = reduce(&, map($op, elements(set)))
end

checkbounds(set::CompositeSet, idx::Tuple{Int,Any}) = checkbounds(element(set,idx[1]), idx[2])

getindex(set::CompositeSet, i::Int) = getindex(set, multilinear_index(set, i))

# For getindex: return indexed basis function of the underlying set
getindex(set::CompositeSet, idx::Tuple{Int,Any}) = getindex(set, idx[1], idx[2])

getindex(set::CompositeSet, i, j) = set.sets[i][j]

MultiLinearIndex{N} = NTuple{N,Int}

function multilinear_index(set::CompositeSet, idx::Int)
    i = 0
    while idx > set.offsets[i+1]
        i += 1
    end
    (i,idx-set.offsets[i])
end

native_index(set::CompositeSet, idx::Int) =
    native_index(set, multilinear_index(set, idx))

# Conversion from multilinear index
function native_index(set::CompositeSet, idxm::MultiLinearIndex{2})
    i,j = idxm
    (i, native_index(element(set, i), j))
end

# Convert from a multilinear index
linear_index(set::CompositeSet, idxm::MultiLinearIndex{2}) = set.offsets[idxm[1]] + idxm[2]

# Convert from a native index (whose type is anything but a tuple of 2 Int's)
function linear_index(set::CompositeSet, idxn)
    # We convert the native index in idxn[2] to a linear index
    i = idxn[1]
    j = linear_index(element(set, i), idxn[2])
    # Now we have a multilinear index and we can use the routine above
    linear_index(set, (i,j))
end


function checkbounds(set::CompositeSet, idx::NTuple{2,Int})
    checkbounds(element(set,idx[1]), idx[2])
end

# Translate a linear index into a multilinear index
in_support(set::CompositeSet, idx, x) = _in_support(set, elements(set), idx, x)

_in_support(set::CompositeSet, sets, idx::Int, x) = in_support(set, multilinear_index(set, idx), x)

_in_support(set::CompositeSet, sets, idx, x) = in_support(sets[idx[1]], idx[2], x)

eachindex(set::CompositeSet) = CompositeIndexIterator(map(length, elements(set)))

## Extension and restriction

extension_size(set::CompositeSet) = map(extension_size, elements(set))

for op in [:extension_operator, :restriction_operator]
    @eval $op(s1::CompositeSetSpan, s2::CompositeSetSpan; options...) =
        BlockDiagonalOperator( AbstractOperator{coeftype(s2)}[$op(element(s1,i),element(s2,i); options...) for i in 1:nb_elements(s1)], s1, s2)
end

# Calling and evaluation
eval_element(set::CompositeSet, idx::Int, x) = eval_element(set, multilinear_index(set,idx), x)

function eval_element(set::CompositeSet{S,T}, idx::Tuple{Int,Any}, x) where {S,T}
    convert(S, eval_element( element(set, idx[1]), idx[2], x))
end


## Differentiation

derivative_space(s::CompositeSetSpan, order; options...) =
    similar_compositespan(s, map(u->derivative_space(u, order; options...), elements(s)))

antiderivative_space(s::CompositeSetSpan, order; options...) =
    similar_compositespan(s, map(u-> antiderivative_space(u, order; options...), elements(s)))
