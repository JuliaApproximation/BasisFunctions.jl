# composite_dict.jl

"""
`CompositeDict` is the abstract supertype of a dictionary that consists of
several subdictionaries. The `CompositeDict` type defines common routines for
indexing and iteration.

The representation of a `CompositeDict` is a `MultiArray`. The outer array of this
`MultiArray` adopts the structure of the elements of the `CompositeDict`: if the elements
are stored in a tuple, the outer array will be a tuple. If the elements are
stored in an array, the outer array will be an array as well.

CompositeDict's that store elements in an array can have a large number of elements.
However, for good efficiency they should have the same type. Tuples are suitable
to hold mixed-type elements with good efficiency, however in this case there can't
be too many.

The concrete subtypes differ in what evaluation means. Examples include:
- MultiDict, where evaluation is the sum of the evaluation of the subsets
- PiecewiseDict, where evaluation depends on the location of the evaluation point
"""
abstract type CompositeDict{S,T} <: Dictionary{S,T}
end

# We assume that every subset has an indexable field called dicts
is_composite(set::CompositeDict) = true
elements(set::CompositeDict) = set.dicts
element(set::CompositeDict, j) = set.dicts[j]
numelements(set::CompositeDict) = length(elements(set))

# For a generic implementation of range indexing, we need a 'similar_dictionary' function
# to create a new set of the same type as the given set.
element(set::CompositeDict, range::AbstractRange) = similar_dictionary(set, set.dicts[range])

tail(set::CompositeDict) = numelements(set) == 2 ? element(set, 2) : element(set, 2:numelements(set))

# We compute offsets of the individual dicts using a cumulative sum
compute_offsets(dicts::Array) = [0; cumsum(map(length, dicts))]

# Convert a tuple to an array in order to use cumsum like above
compute_offsets(dicts::NTuple{N,Any}) where {N} = compute_offsets([dict for dict in dicts])

"""
Pass the internal `offsets` vector of the composite dict. This is not safe because
the vector is not copied, hence its elements could be changed. That would affect
the original composite dictionary. Use with care.
"""
unsafe_offsets(dict::CompositeDict) = dict.offsets

# Implement equality in terms of equality of the elements.
==(s1::CompositeDict, s2::CompositeDict) = (elements(s1) == elements(s2))

length(s::CompositeDict) = s.offsets[end]

length(s::CompositeDict, i::Int) = length(element(s,i))

## Concrete subtypes should override similar_dictionary and call their own constructor

# Using map in the definitions below ensures that a tuple is created when
# elements(set) is a tuple, and an array when elements(set) is an array.
resize(set::CompositeDict, n) =
    similar_dictionary(set, map( (s,l) -> resize(s, l), elements(set), n))

dict_promote_domaintype(set::CompositeDict, ::Type{S}) where {S} =
    similar_dictionary(set, map(s->dict_promote_domaintype(s, S), elements(set)))

zeros(::Type{T}, set::CompositeDict) where {T} = MultiArray(map(s->zeros(T,s),elements(set)))

for op in (:isreal, )
    @eval $op(set::CompositeDict) = reduce($op, elements(set))
end

for op in (:has_derivative, :has_antiderivative, :has_extension)
    @eval $op(set::CompositeDict) = reduce(&, map($op, elements(set)))
end

##################
# Indexing
##################

const MultilinearIndices = Union{MultilinearIndex,Tuple{Int,Any}}

native_index(dict::CompositeDict, idx::MultilinearIndex) = idx

multilinear_index(dict::CompositeDict, idx::LinearIndex) =
    offsets_multilinear_index(unsafe_offsets(dict), idx)

linear_index(dict::CompositeDict, idx::MultilinearIndex) =
    offsets_linear_index(unsafe_offsets(dict), idx)


# The native index is a MultilinearIndex, defined in util/indexing.jl
ordering(d::CompositeDict) = MultilinearIndexList(unsafe_offsets(d))

# We have to amend the boundscheck ecosystem to catch some cases:
# - This line will catch indexing with tuples of integers, and we assume
#   the user wanted to use a CartesianIndex
checkbounds(::Type{Bool}, d::CompositeDict, idx::MultilinearIndices) =
    checkbounds(Bool, d, (idx[1], linear_index(element(d,idx[1]), idx[2])))
# - and this line to avoid an ambiguity
checkbounds(::Type{Bool}, d::CompositeDict, idx::Tuple{Int,Int}) =
    checkbounds(Bool, d, linear_index(d, idx))

# For getindex: return indexed basis function of the underlying set
getindex(set::CompositeDict, idx::LinearIndex) = getindex(set, native_index(set, idx))
getindex(set::CompositeDict, idx::MultilinearIndices) = getindex(set, idx[1], idx[2])
getindex(set::CompositeDict, i, j) = set.dicts[i][j]


dict_in_support(set::CompositeDict, idx, x) = _dict_in_support(set, elements(set), idx, x)
_dict_in_support(set::CompositeDict, dicts, idx, x) = in_support(dicts[idx[1]], idx[2], x)

eachindex(set::CompositeDict) = MultilinearIndexIterator(map(length, elements(set)))

## Extension and restriction

extension_size(set::CompositeDict) = map(extension_size, elements(set))

for op in [:extension_operator, :restriction_operator]
    @eval $op(s1::CompositeDict, s2::CompositeDict; options...) =
        BlockDiagonalOperator( DictionaryOperator{coeftype(s2)}[$op(element(s1,i),element(s2,i); options...) for i in 1:numelements(s1)], s1, s2)
end

# Calling and evaluation
unsafe_eval_element(set::CompositeDict, idx::Int, x) = unsafe_eval_element(set, multilinear_index(set,idx), x)

function unsafe_eval_element(set::CompositeDict{S,T}, idx::Tuple{Int,Any}, x) where {S,T}
    convert(T, unsafe_eval_element( element(set, idx[1]), idx[2], x))
end

unsafe_eval_element_derivative(set::CompositeDict, idx::Int, x) = unsafe_eval_element_derivative(set, multilinear_index(set,idx), x)

function unsafe_eval_element_derivative(set::CompositeDict{S,T}, idx::Tuple{Int,Any}, x) where {S,T}
    convert(T, unsafe_eval_element_derivative( element(set, idx[1]), idx[2], x))
end


derivative_dict(s::CompositeDict, order; options...) =
    similar_dictionary(s,map(u->derivative_dict(u, order; options...), elements(s)))

antiderivative_dict(s::CompositeDict, order; options...) =
    similar_dictionary(s,map(u->antiderivative_dict(u, order; options...), elements(s)))
