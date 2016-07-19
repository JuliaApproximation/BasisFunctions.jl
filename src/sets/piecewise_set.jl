# piecewise_set.jl

immutable PiecewiseSet{NT,T} <: FunctionSet{1,T}
    sets        ::  Array{FunctionSet{1,T},1}
    breakpts    ::  Array{NT,1}
    lengths     ::  Array{Int,1}

    function PiecewiseSet(sets)
        breakpts = NT[map(left, sets)..., right(sets[end])]
        lengths = map(length, sets)
        new(sets, breakpts, lengths)
    end
end

function PiecewiseSet(sets)
    T = eltype(sets[1])
    NT = numtype(sets[1])
    PiecewiseSet{NT,T}(sets)
end

left(set::PiecewiseSet) = set.breakpts[1]
right(set::PiecewiseSet) = set.breakpts[end]

name(s::PiecewiseSet) = "Piecewise function set"

elements(s::PiecewiseSet) = s.sets
element(s::PiecewiseSet, j::Int) = s.sets[j]
element(s::PiecewiseSet, range::Range) = PiecewiseSet(s.sets[range])
composite_length(s::PiecewiseSet) = length(s.sets)

promote_eltype{NT,T,S}(s::PiecewiseSet{NT,T}, ::Type{S}) =
    PiecewiseSet([promote_eltype(element(s,i), S) for i in 1:composite_length(s)])

length(s::PiecewiseSet) = sum(s.lengths)


for op in (:isreal, :is_orthogonal, :is_biorthogonal, :is_basis, :is_frame)
    @eval $op(s::PiecewiseSet) = reduce($op, elements(s))
end

for op in (:has_derivative, :has_antiderivative, :has_extension, :has_grid, :has_transform)
    @eval $op(s::PiecewiseSet) = reduce($op, elements(s))
end


getindex(s::PiecewiseSet, idx::NTuple{2,Int}) = getindex(s, idx[1], idx[2])

getindex(s::PiecewiseSet, i::Int, j::Int) = s.sets[i][j]

immutable PiecewiseSetIterator{NT,T}
    set ::  PiecewiseSet{NT,T}
end

eachindex(s::PiecewiseSet) = PiecewiseSetIterator(s)

start(it::PiecewiseSetIterator) = (1,1)

function next(it::PiecewiseSetIterator, state)
    i = state[1]
    j = state[2]
    if j == it.set.lengths[i]
        nextstate = (i+1,1)
    else
        (i,j+1)
    end
    (it.set[state[1]], nextstate)
end

done(it::PiecewiseSetIterator, state) = state[1] > composite_length(it.set)

function getsubindex(s::PiecewiseSet, j::Int)
    i = 1
    while j > s.lengths[i]
        i += 1
        j -= s.lengths[i]
    end
    (i,j)
end

getindex(s::PiecewiseSet, i::Int) = getindex(s, getsubindex(s, i))

for op in [:left, :right, :moment, :norm]
    @eval function $op(s::PiecewiseSet, idx)
        (i,j) = getsubindex(s, idx)
        $op(s.sets[i], j)
    end
end
