# multiple_set.jl

"""
A MultiSet is the concatenation of several different function sets.
"""
immutable MultiSet{N,T} <: FunctionSet{N,T}
    sets        ::  Array{FunctionSet{N,T},1}
    lengths     ::  Array{Int,1}

    function MultiSet(sets)
        lengths = map(length, sets)
        new(sets, lengths)
    end
end

function MultiSet(sets)
    T = eltype(sets[1])
    N = ndims(sets[1])
    MultiSet{N,T}(sets)
end

name(s::MultiSet) = "A set consisting of multiple sets"

elements(s::MultiSet) = s.sets
element(s::MultiSet, j::Int) = s.sets[j]
element(s::MultiSet, range::Range) = MultiSet(s.sets[range])
composite_length(s::MultiSet) = length(s.sets)

length(s::MultiSet) = sum(s.lengths)

resize{N,T}(s::MultiSet{N,T}, n::Array{Int,1}) =
    MultiSet( FunctionSet{N,T}[resize(element(s,i), n[i]) for i in 1:composite_length(s)] )

promote_eltype{N,T,S}(s::MultiSet{N,T}, ::Type{S}) =
    MultiSet([promote_eltype(element(s,i), S) for i in 1:composite_length(s)])

zeros(T::Type, s::MultiSet) = [zeros(element(s,i)) for i in 1:composite_length(s)]

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
        i += 1
        idx -= s.lengths[i]
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

for op in [:left, :right, :moment, :norm]
    @eval function $op(s::MultiSet, idx)
        (i,j) = native_index(s, idx)
        $op(s.sets[i], j)
    end
end

left(set::MultiSet) = left(set.sets[1])
right(set::MultiSet) = right(set.sets[end])

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

immutable MultiSetIterator{N,T}
    set ::  MultiSet{N,T}
end

eachindex(s::MultiSet) = MultiSetIterator(s)

start(it::MultiSetIterator) = (1,1)

function next(it::MultiSetIterator, state)
    i = state[1]
    j = state[2]
    if j == it.set.lengths[i]
        nextstate = (i+1,1)
    else
        nextstate = (i,j+1)
    end
    (state, nextstate)
end

done(it::MultiSetIterator, state) = state[1] > composite_length(it.set)

## Differentiation





## Extension and restriction

extension_size(s::MultiSet) = map(extension_size, elements(s))

extension_operator{N,T}(s1::MultiSet{N,T}, s2::MultiSet{N,T}; options...) =
    BlockDiagonalOperator( AbstractOperator{T}[extension_operator(element(s1,i),element(s2,i); options...) for i in 1:composite_length(s1)], s1, s2)

restriction_operator{N,T}(s1::MultiSet{N,T}, s2::MultiSet{N,T}; options...) =
    BlockDiagonalOperator( AbstractOperator{T}[restriction_operator(element(s1,i),element(s2,i); options...) for i in 1:composite_length(s1)], s1, s2)
