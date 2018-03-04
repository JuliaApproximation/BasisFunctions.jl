# composite_dict.jl

"""
`CompositeDict` is the abstract supertype of a dictionary that consists of
several subdictionaries. The `CompositeDict` type defines common routines for
indexing and iteration.

The representation of a CompositeDict is a MultiArray. The outer array of this
MultiArray adopts the structure of the elements of the CompositeDict: if the elements
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

const CompositeDictSpan{A,S,T,D <: CompositeDict} = Span{A,S,T,D}

# We assume that every subset has an indexable field called dicts
is_composite(set::CompositeDict) = true
elements(set::CompositeDict) = set.dicts
element(set::CompositeDict, j::Int) = set.dicts[j]

similar_compositespan(s::CompositeDictSpan, spans) =
    Span(similar_dictionary(dictionary(s), map(dictionary, spans)), coeftype(s))

# For a generic implementation of range indexing, we need a 'similar_dictionary' function
# to create a new set of the same type as the given set.
element(set::CompositeDict, range::Range) = similar_dictionary(set, set.dicts[range])

tail(set::CompositeDict) = nb_elements(set) == 2 ? element(set, 2) : element(set, 2:nb_elements(set))

# We compute offsets of the individual dicts using a cumulative sum
compute_offsets(dicts::Array) = [0; cumsum(map(length, dicts))]

# Convert a tuple to an array in order to use cumsum like above
compute_offsets(dicts::NTuple{N,Any}) where {N} = compute_offsets([dict for dict in dicts])

# Implement equality in terms of equality of the elements.
==(s1::CompositeDict, s2::CompositeDict) = (elements(s1) == elements(s2))

length(s::CompositeDict) = s.offsets[end]

length(s::CompositeDict, i::Int) = length(element(s,i))

# Concrete subtypes should override similar_dictionary and call their own constructor

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

checkbounds(set::CompositeDict, idx::Tuple{Int,Any}) = checkbounds(element(set,idx[1]), idx[2])

getindex(set::CompositeDict, i::Int) = getindex(set, multilinear_index(set, i))

# For getindex: return indexed basis function of the underlying set
getindex(set::CompositeDict, idx::Tuple{Int,Any}) = getindex(set, idx[1], idx[2])

getindex(set::CompositeDict, i, j) = set.dicts[i][j]

MultiLinearIndex{N} = NTuple{N,Int}

function multilinear_index(set::CompositeDict, idx::Int)
    i = 0
    while idx > set.offsets[i+1]
        i += 1
    end
    (i,idx-set.offsets[i])
end

native_index(set::CompositeDict, idx::Int) =
    native_index(set, multilinear_index(set, idx))

# Conversion from multilinear index
function native_index(set::CompositeDict, idxm::MultiLinearIndex{2})
    i,j = idxm
    (i, native_index(element(set, i), j))
end

# Convert from a multilinear index
linear_index(set::CompositeDict, idxm::MultiLinearIndex{2}) = set.offsets[idxm[1]] + idxm[2]

# Convert from a native index (whose type is anything but a tuple of 2 Int's)
function linear_index(set::CompositeDict, idxn)
    # We convert the native index in idxn[2] to a linear index
    i = idxn[1]
    j = linear_index(element(set, i), idxn[2])
    # Now we have a multilinear index and we can use the routine above
    linear_index(set, (i,j))
end


function checkbounds(set::CompositeDict, idx::NTuple{2,Int})
    checkbounds(element(set,idx[1]), idx[2])
end

# Translate a linear index into a multilinear index
in_support(set::CompositeDict, idx, x) = _in_support(set, elements(set), idx, x)

_in_support(set::CompositeDict, dicts, idx::Int, x) = in_support(set, multilinear_index(set, idx), x)

_in_support(set::CompositeDict, dicts, idx, x) = in_support(dicts[idx[1]], idx[2], x)

eachindex(set::CompositeDict) = CompositeIndexIterator(map(length, elements(set)))

## Extension and restriction

extension_size(set::CompositeDict) = map(extension_size, elements(set))

for op in [:extension_operator, :restriction_operator]
    @eval $op(s1::CompositeDictSpan, s2::CompositeDictSpan; options...) =
        BlockDiagonalOperator( AbstractOperator{coeftype(s2)}[$op(element(s1,i),element(s2,i); options...) for i in 1:nb_elements(s1)], s1, s2)
end

# Calling and evaluation
eval_element(set::CompositeDict, idx::Int, x) = eval_element(set, multilinear_index(set,idx), x)

function eval_element(set::CompositeDict{S,T}, idx::Tuple{Int,Any}, x) where {S,T}
    convert(T, eval_element( element(set, idx[1]), idx[2], x))
end

eval_element_derivative(set::CompositeDict, idx::Int, x) = eval_element_derivative(set, multilinear_index(set,idx), x)

function eval_element_derivative(set::CompositeDict{S,T}, idx::Tuple{Int,Any}, x) where {S,T}
    convert(T, eval_element_derivative( element(set, idx[1]), idx[2], x))
end

## Differentiation

derivative_space(s::CompositeDictSpan, order; options...) =
    similar_compositespan(s, map(u->derivative_space(u, order; options...), elements(s)))

antiderivative_space(s::CompositeDictSpan, order; options...) =
    similar_compositespan(s, map(u-> antiderivative_space(u, order; options...), elements(s)))
