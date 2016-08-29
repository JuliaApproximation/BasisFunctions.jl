# multiarray.jl

immutable MultiArray{T,N}
    # The underlying data is an array of arrays. Thus, we assume that the elements
    # of the multi array are regular arrays.
    arrays  ::  Array{Array{T,N},1}
    lengths ::  Array{Int,1}

    function MultiArray(arrays)
        lengths = map(length, arrays)
        new(arrays, lengths)
    end
end

MultiArray{T,N}(a::Array{Array{T,N}}) = MultiArray{T,N}(a)

eltype{T,N}(::Type{MultiArray{T,N}}) = T

element(a::MultiArray, i) = a.arrays[i]
elements(a::MultiArray) = a.arrays
composite_length(a::MultiArray) = length(a.arrays)

length(a::MultiArray) = sum(a.lengths)

length(a::MultiArray, i::Int) = a.lengths[i]

show(io::IO, a::MultiArray) = show(io, a.arrays)

##########################
## Iteration and indexing
##########################


immutable MultiArrayIndexIterator{T,N}
    array   ::  MultiArray{T,N}
end

eachindex(s::MultiArray) = MultiArrayIndexIterator(s)

start(it::MultiArrayIndexIterator) = (1,1)

function next(it::MultiArrayIndexIterator, state)
    i = state[1]
    j = state[2]
    if j == it.array.lengths[i]
        nextstate = (i+1,1)
    else
        nextstate = (i,j+1)
    end
    (state, nextstate)
end

done(it::MultiArrayIndexIterator, state) = state[1] > composite_length(it.array)

length(it::MultiArrayIndexIterator) = length(it.array)

getindex(s::MultiArray, idx::NTuple{2,Int}) = getindex(s, idx[1], idx[2])

getindex(s::MultiArray, i::Int, j::Int) = s.arrays[i][j]

getindex(s::MultiArray, i::Int, j::Int...) = s.arrays[i][j...]

setindex!(s::MultiArray, v, idx::NTuple{2,Int}) = setindex!(s, v, idx[1], idx[2])

setindex!(s::MultiArray, v, i::Int, j::Int) = s.arrays[i][j] = v

function native_index(s::MultiArray, idx::Int)
    i = 1
    while idx > s.lengths[i]
        idx -= s.lengths[i]
        i += 1
    end
    (i,idx)
end

function linear_index(s::MultiArray, idxn)
    idx = idxn[2]
    for i = 1:idxn[1]-1
        idx += s.lengths[i]
    end
    idx
end

################
# Arithmetic
################

function fill!(a::MultiArray, val)
    for array in elements(a)
        fill!(array, val)
    end
end

for op in (:+, :*, :-, :/, :.*, :.+, :.-, :./, :.\, :.^, :.÷)
    @eval $op(a::MultiArray, b::MultiArray) = MultiArray([$op(a.arrays[i], b.arrays[i]) for i in 1:length(a.arrays)])
    @eval $op(a::Number, b::MultiArray) = MultiArray([$op(a, array) for array in b.arrays])
    @eval $op(a::MultiArray, b::Number) = MultiArray([$op(array, b) for array in a.arrays])
end
