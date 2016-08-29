# multiple_set.jl

"""
A MultiSet is the concatenation of several different function sets.
"""
immutable MultiSet{N,T} <: FunctionSet{N,T}
    sets        ::  Array{FunctionSet{N,T},1}
    lengths     ::  Array{Int,1}

    function MultiSet(sets)
        @assert length(sets) > 1

        lengths = map(length, sets)
        new(sets, lengths)
    end
end

function MultiSet(sets)
    T = reduce(promote_type, map(eltype, sets))
    N = ndims(sets[1])
    MultiSet{N,T}([promote_eltype(set, T) for set in sets])
end

==(s1::MultiSet, s2::MultiSet) = (s1.sets == s2.sets) && (s1.lengths == s2.lengths)

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

⊕(s1::FunctionSet, s2::FunctionSet) = multiset(s1, s2)

name(s::MultiSet) = "A set consisting of $(composite_length(s)) sets"

elements(s::MultiSet) = s.sets
element(s::MultiSet, j::Int) = s.sets[j]
element(s::MultiSet, range::Range) = MultiSet(s.sets[range])
composite_length(s::MultiSet) = length(s.sets)

length(s::MultiSet) = sum(s.lengths)

resize{N,T}(s::MultiSet{N,T}, n::Array{Int,1}) =
    MultiSet( FunctionSet{N,T}[resize(element(s,i), n[i]) for i in 1:composite_length(s)] )

promote_eltype{N,T,S}(s::MultiSet{N,T}, ::Type{S}) =
    MultiSet([promote_eltype(el, S) for el in elements(s)])

zeros(T::Type, s::MultiSet) = MultiArray(Array{T,ndims(s)}[zeros(T,el) for el in elements(s)])

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

getindex(s::MultiSet, idx::NTuple{2,Int}) = getindex(s, idx[1], idx[2])

getindex(s::MultiSet, i::Int, j::Int) = s.sets[i][j]

getindex(s::MultiSet, i::Int, j::Int...) = s.sets[i][j...]

function native_index(s::MultiSet, idx::Int)
    i = 1
    while idx > s.lengths[i]
        idx -= s.lengths[i]
        i += 1
    end
    (i,idx)
end

function linear_index(s::MultiSet, idxn)
    idx = idxn[2]
    for i = 1:idxn[1]-1
        idx += s.lengths[i]
    end
    idx
end

getindex(s::MultiSet, i::Int) = getindex(s, native_index(s, i))

function checkbounds(s::MultiSet, idx::NTuple{2,Int})
    checkbounds(element(s,idx[1]), idx[2])
end

function call_element_native(s::MultiSet, idx, x...)
    call_element( element(s, idx[1]), idx[2], x...)
end

for op in [:left, :right, :moment, :norm]
    @eval function $op(s::MultiSet, idx)
        (i,j) = native_index(s, idx)
        $op(s.sets[i], j)
    end
end

left(set::MultiSet) = minimum(map(left, elements(set)))
right(set::MultiSet) = maximum(map(right, elements(set)))

function linearize_coefficients!(coef_linear, set::MultiSet, coef_native)
    for (i,j) in enumerate(eachindex(set))
        coef_linear[i] = coef_native[j[1]][j[2]]
    end
    coef_linear
end

function delinearize_coefficients!(coef_native, set::MultiSet, coef_linear)
    for (i,j) in enumerate(eachindex(set))
        coef_native[j[1]][j[2]] = coef_linear[i]
    end
    coef_native
end

## Iteration

immutable MultiSetIndexIterator{N,T}
    set ::  MultiSet{N,T}
end

eachindex(s::MultiSet) = MultiSetIndexIterator(s)

start(it::MultiSetIndexIterator) = (1,1)

function next(it::MultiSetIndexIterator, state)
    i = state[1]
    j = state[2]
    if j == it.set.lengths[i]
        nextstate = (i+1,1)
    else
        nextstate = (i,j+1)
    end
    (state, nextstate)
end

done(it::MultiSetIndexIterator, state) = state[1] > composite_length(it.set)

length(it::MultiSetIndexIterator) = length(it.set)

## Differentiation

# Below, we deliberately use MultiSet instead of multiset, because it is difficult
# to construct the right differentiation operator if the sets are flattened into
# a single multiset. Note that muliset flattens while MultiSet doesn't.
derivative_set(s::MultiSet, order::Int; options...) =
    multiset(map(b-> derivative_set(b, order; options...), elements(s)))

for op in [:differentiation_operator, :antidifferentiation_operator]
    @eval function $op(s1::MultiSet, s2::MultiSet, order::Int; options...)
        if composite_length(s1) == composite_length(s2)
            BlockDiagonalOperator(AbstractOperator{eltype(s1)}[$op(element(s1,i), element(s2, i); options...) for i in 1:composite_length(s1)], s1, s2)
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

evaluation_operator{N,T}(set::MultiSet{N,T}, dgs::DiscreteGridSpace; options...) =
    block_row_operator( AbstractOperator{T}[evaluation_operator(el,dgs; options...) for el in elements(set)])

## Extension and restriction

extension_size(s::MultiSet) = map(extension_size, elements(s))

for op in [:extension_operator, :restriction_operator]
    @eval $op{N,T}(s1::MultiSet{N,T}, s2::MultiSet{N,T}; options...) =
        BlockDiagonalOperator( AbstractOperator{T}[$op(element(s1,i),element(s2,i); options...) for i in 1:composite_length(s1)], s1, s2)
end
