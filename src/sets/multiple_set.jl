# multiple_set.jl

"""
A MultiSet is the concatenation of several function sets. The function sets
may be the same (but scaled to different intervals, say) or they can be different.
The number of subsets may be large.

The native representation of a MultiSet is a MultiArray, of which each element
is the native representation of the corresponding element of the multiset.
"""
immutable MultiSet{S <: FunctionSet,N,T} <: FunctionSet{N,T}
    sets        ::  Array{S,1}
    # The cumulative sum of the lengths of the subsets. Used to compute indices.
    offsets     ::  Array{Int,1}

    function MultiSet(sets)
        # Disallow a multiset with just one set
        @assert length(sets) > 1
        # We compute offsets of the individual sets using a cumulative sum
        new(sets, [0; cumsum(map(length, sets))])
    end
end

function MultiSet{S <: FunctionSet}(sets::Array{S,1})
    T = reduce(promote_type, map(eltype, sets))
    N = ndims(sets[1])
    MultiSet{S,N,T}([promote_eltype(set, T) for set in sets])
end

==(s1::MultiSet, s2::MultiSet) = (s1.sets == s2.sets) && (s1.offsets == s2.offsets)

multiset(set::FunctionSet) = set

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

vcat(s1::FunctionSet, s2::FunctionSet) = multiset(s1,s2)

⊕(s1::FunctionSet, s2::FunctionSet) = multiset(s1, s2)

name(s::MultiSet) = "A set consisting of $(composite_length(s)) sets"

elements(s::MultiSet) = s.sets
element(s::MultiSet, j::Int) = s.sets[j]
element(s::MultiSet, range::Range) = multiset(s.sets[range])
composite_length(s::MultiSet) = length(s.sets)

length(s::MultiSet) = s.offsets[end]

length(s::MultiSet, i::Int) = length(element(s,i))

resize{S,N,T}(s::MultiSet{S,N,T}, n) =
    MultiSet( [resize(element(s,i), n[i]) for i in 1:composite_length(s)] )

promote_eltype{S,N,T,T2}(s::MultiSet{S,N,T}, ::Type{T2}) =
    MultiSet([promote_eltype(el, T2) for el in elements(s)])

zeros(T::Type, s::MultiSet) = MultiArray([zeros(T,el) for el in elements(s)])

for op in (:isreal, )
    @eval $op(s::MultiSet) = reduce($op, elements(s))
end

for op in (:is_orthogonal, :is_biorthogonal, :is_basis, :is_frame)
    # Redirect the calls to multiple_is_basis with the elements as extra arguments,
    # and that method can decide whether the property holds for the multiset.
    fname = Symbol("multiple_$(op)")
    @eval $op(s::MultiSet) = ($fname)(s, elements(s)...)
    # By default, multisets do not have these properties:
    @eval ($fname)(s, elements...) = false
end

for op in (:has_derivative, :has_antiderivative, :has_extension)
    @eval $op(s::MultiSet) = reduce(&, map($op, elements(s)))
end

for op in (:has_grid, :has_transform)
    fname = Symbol("multiple_$(op)")
    @eval $op(s::MultiSet) = ($fname)(s, elements(s)...)
    @eval ($fname)(s, elements...) = false
end

getindex(s::MultiSet, i::Int) = getindex(s, multilinear_index(s, i))

# For getindex: return indexed basis function of the underlying set
getindex(s::MultiSet, idx::Tuple{Int,Any}) = getindex(s, idx[1], idx[2])

getindex(s::MultiSet, i, j) = s.sets[i][j]

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


function multilinear_index(s::MultiSet, idx::Int)
    i = 0
    while idx > s.offsets[i+1]
        i += 1
    end
    (i,idx-s.offsets[i])
end

function native_index(s::MultiSet, idx::Int)
    (i,j) = multilinear_index(s, idx)
    (i,native_index(element(s,i), j))
end

# Convert from a multilinear index
linear_index(s::MultiSet, idx_ml::NTuple{2,Int}) = s.offsets[idx_ml[1]] + idx_ml[2]

# Convert from a native index (whose type is anything but a tuple of 2 Int's)
function linear_index(s::MultiSet, idxn)
    # We convert the native index in idxn[2] to a linear index
    i = idxn[1]
    j = linear_index(element(s, i), idxn[2])
    # Now we have a multilinear index and we can use the routine above
    linear_index(s, (i,j))
end


function checkbounds(s::MultiSet, idx::NTuple{2,Int})
    checkbounds(element(s,idx[1]), idx[2])
end

call_element(s::MultiSet, idx::Int, x) = call_element(s, multilinear_index(s,idx), x)

function call_element(s::MultiSet, idx::Tuple{Int,Any}, x)
    call_element( element(s, idx[1]), idx[2], x)
end

for op in [:left, :right, :moment, :norm]
    @eval function $op(s::MultiSet, idx)
        (i,j) = native_index(s, idx)
        $op(s.sets[i], j)
    end
end

left(set::MultiSet) = minimum(map(left, elements(set)))
right(set::MultiSet) = maximum(map(right, elements(set)))


## Iteration

immutable MultiSetIndexIterator{S,N,T}
    set ::  MultiSet{S,N,T}
end

eachindex(s::MultiSet) = MultiSetIndexIterator(s)

start(it::MultiSetIndexIterator) = (1,1)

function next(it::MultiSetIndexIterator, state)
    i = state[1]
    j = state[2]
    if j == length(it.set, i)
        nextstate = (i+1,1)
    else
        nextstate = (i,j+1)
    end
    (state, nextstate)
end

done(it::MultiSetIndexIterator, state) = state[1] > composite_length(it.set)

length(it::MultiSetIndexIterator) = length(it.set)

## Differentiation

derivative_set(s::MultiSet, order; options...) =
    MultiSet(map(b-> derivative_set(b, order; options...), elements(s)))

antiderivative_set(s::MultiSet, order; options...) =
    MultiSet(map(b-> antiderivative_set(b, order; options...), elements(s)))

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
            block_diagonal_operator(ops)
        end
    end
end

evaluation_operator(set::MultiSet, dgs::DiscreteGridSpace; options...) =
    block_row_operator( AbstractOperator{eltype(set)}[evaluation_operator(el,dgs; options...) for el in elements(set)])

## Extension and restriction

extension_size(s::MultiSet) = map(extension_size, elements(s))

for op in [:extension_operator, :restriction_operator]
    @eval $op(s1::MultiSet, s2::MultiSet; options...) =
        BlockDiagonalOperator( AbstractOperator{eltype(s1)}[$op(element(s1,i),element(s2,i); options...) for i in 1:composite_length(s1)], s1, s2)
end

## Rescaling

rescale(s::MultiSet, a, b) = multiset(map( t-> rescale(t, a, b), elements(s)))
