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
    N = dim(sets[1])
    MultiSet{N,T}(sets)
end

name(s::MultiSet) = "A set consisting of multiple sets"

elements(s::MultiSet) = s.sets
element(s::MultiSet, j::Int) = s.sets[j]
element(s::MultiSet, range::Range) = MultiSet(s.sets[range])
composite_length(s::MultiSet) = length(s.sets)

promote_eltype{NT,T,S}(s::MultiSet{NT,T}, ::Type{S}) =
    MultiSet([promote_eltype(element(s,i), S) for i in 1:composite_length(s)])

length(s::MultiSet) = sum(s.lengths)

resize{N,T}(s::MultiSet{N,T}, n::Array{Int,1}) =
    MultiSet( FunctionSet{N,T}[resize(element(s,i), n[i]) for i in 1:composite_length(s)] )

for op in (:isreal, )
    @eval $op(s::MultiSet) = reduce($op, elements(s))
end

for op in (:is_orthogonal, :is_biorthogonal, :is_basis, :is_frame)
    @eval $op(s::MultiSet) = multiple_$op(s, elements(s))
end

for op in (:has_derivative, :has_antiderivative, :has_extension)
    @eval $op(s::MultiSet) = reduce($op, elements(s))
end

for op in (:has_grid, :has_transform)
    @eval $op(s::MultiSet) = multiple_$op(s, elements(s))
end

getindex(s::MultiSet, idx::NTuple{2,Int}) = getindex(s, idx[1], idx[2])

getindex(s::MultiSet, i::Int, j::Int) = s.sets[i][j]

getindex(s::MultiSet, i::Int, j::Int...) = s.sets[i][j...]

function getsubindex(s::MultiSet, j::Int)
    i = 1
    while j > s.lengths[i]
        i += 1
        j -= s.lengths[i]
    end
    (i,j)
end

getindex(s::MultiSet, i::Int) = getindex(s, getsubindex(s, i))

for op in [:left, :right, :moment, :norm]
    @eval function $op(s::MultiSet, idx)
        (i,j) = getsubindex(s, idx)
        $op(s.sets[i], j)
    end
end

left(set::MultiSet) = left(set.sets[1])
right(set::MultiSet) = right(set.sets[end])


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
    (it.set[state[1]], nextstate)
end

done(it::MultiSetIterator, state) = state[1] > composite_length(it.set)
