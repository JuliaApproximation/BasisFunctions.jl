# multiarray.jl

"""
A MultiArray in its simplest form is an array of arrays. It is meant to store
coefficient sets of different expansions together, i.e. each element of the outer
array stores the coefficients of an expansion. This is useful as a representation
for MultiSet's.

A MultiArray has one outer array, of which each element is an inner array. The
inner arrays can represent the coefficients of an expansion. In case the
outer array is actually of type Array, the number of inner arrays may be large.
However, for good efficiency, all inner arrays should have the same type. The outer
array can also be a tuple, in which case the inner arrays can have mixed types,
but there should not be too many.

The MultiArray can be indexed with a linear index, which iterates over all the
elements of the subarray's linearly. This index would correspond to the linear
index of a MultiSet. Alternatively, the MultiArray can be index with a native index,
which is a tuple consisting of the outer array index and the native index of the
coefficient set it points to.

Usually, the individual expansions (i.e. the inner arrays) have type Array{ELT,N},
but that need not be the case. A MultiArray can contain other kinds of function
representations. However, they should all have the same eltype.

The differences between this type and an Array of Array's are:
- this type supports linear indexing
- the eltype of a MultiArray is the eltype of the inner arrays, not the type of
an inner array.
These differences have motivated this special purpose type.
"""
immutable MultiArray{A,ELT}
    # The underlying data is usually an array of arrays, but may be more general.
    # At the least, it is an indexable set that supports a linear index.
    arrays  ::  A
    # The cumulative sum of the lengths of the subarrays. Used to compute indices.
    # The linear index for the i-th subarray starts at offsets[i]+1.
    offsets ::  Array{Int,1}

    function MultiArray(arrays)
        new(arrays, compute_offsets(arrays))
    end
end

function MultiArray(arrays, ELT = eltype(arrays[1]))
    for array in arrays
        @assert eltype(array) == ELT
    end
    MultiArray{typeof(arrays),ELT}(arrays)
end

eltype{A,ELT}(::Type{MultiArray{A,ELT}}) = ELT

element(a::MultiArray, i) = a.arrays[i]
elements(a::MultiArray) = a.arrays
composite_length(a::MultiArray) = length(a.arrays)

length(a::MultiArray) = a.offsets[end]

# This definition is up for debate: what is the size of a MultiArray?
# By defining it as (length(a),), we can assert elsewhere that the size of a
# multiset equals the size of its multiarray representation.
size(a::MultiArray) = (length(a),)

length(a::MultiArray, i::Int) = length(a.arrays[i])

show(io::IO, a::MultiArray) = show(io, a.arrays)

function linearize_coefficients(a::MultiArray)
    b = zeros(eltype(a), length(a))
    linearize_coefficients!(b, a)
end

function linearize_coefficients!{T}(b::Array{T,1}, a::MultiArray)
    @assert length(a) == length(b)
    for (i,j) in enumerate(eachindex(a))
        b[i] = a[j]
    end
    b
end

function delinearize_coefficients!{T}(a::MultiArray, b::Array{T,1})
    @assert length(a) == length(b)
    for (i,j) in enumerate(eachindex(a))
        a[j] = b[i]
    end
    b
end

"Set the coefficients of element idx of the multiarray to the given values."
function coefficients!(a::MultiArray, idx, values)
    z = element(a, idx)
    for k in eachindex(z, values)
        z[k] = values[k]
    end
    a
end

##########################
## Iteration and indexing
##########################

# We introduce this type in order to support eachindex for MultiArray's below
immutable MultiArrayIndexIterator{A,ELT}
    array   ::  MultiArray{A,ELT}
end

eachindex(s::MultiArray) = MultiArrayIndexIterator(s)

# Or should we just do:
# eachindex(s::MultiArray) = CompositeIndexIterator(map(length, elements(s)))
# and do away with MultiArrayIndexIterator?


function eachindex(s1::MultiArray, s2::MultiArray)
    @assert composite_length(s1) == composite_length(s2)
    @assert length(s1) == length(s2)
    eachindex(s1)
end

function eachindex(s1::MultiArray, s2::AbstractArray)
    @assert length(s1) == length(s2)
    1:length(s1)
end

function eachindex(s1::AbstractArray, s2::MultiArray)
    @assert length(s1) == length(s2)
    1:length(s1)
end


function start(it::MultiArrayIndexIterator)
    I = eachindex(element(it.array, 1))
    (1, I, start(I))
end

function next(it::MultiArrayIndexIterator, state)
    i, sub_iterator, sub_iterator_state = state
    sub_index, sub_nextstate = next(sub_iterator, sub_iterator_state)
    if done(sub_iterator, sub_nextstate) && (i < composite_length(it.array))
        next_sub_iterator = eachindex(element(it.array, i+1))
        next_state = (i+1, next_sub_iterator, start(next_sub_iterator))
    else
        next_state = (i, sub_iterator, sub_nextstate)
    end
    ((i,sub_index), next_state)
end

done(it::MultiArrayIndexIterator, state) = (state[1]==composite_length(it.array)) && done(state[2], state[3])

length(it::MultiArrayIndexIterator) = length(it.array)

typealias ArrayOfArray{T} Array{Array{T,1},1}

# Our iterator can be simpler when the element sets are vectors in an array
# Strictly speaking we can do even better, since we don't need the full array in the
# field of the iterator, only its dimensions
start{T}(it::MultiArrayIndexIterator{ArrayOfArray{T}}) = (1,1)

function next{T}(it::MultiArrayIndexIterator{ArrayOfArray{T}}, state)
    i = state[1]
    j = state[2]
    if j == length(it.array, i)
        nextstate = (i+1,1)
    else
        nextstate = (i,j+1)
    end
    (state, nextstate)
end

done{T}(it::MultiArrayIndexIterator{ArrayOfArray{T}}, state) = state[1] > composite_length(it.array)

# Support linear indexing too
getindex(a::MultiArray, idx::Int) = getindex(a, multilinear_index(a, idx))

# Indexing with two arguments: the first is the number of the subset, the
# second can be any index of the subset.
getindex(a::MultiArray, i::Int, j) = a.arrays[i][j]

# Indexing with tuples allows for recursive indices
getindex{I}(a::MultiArray, idx::Tuple{Int,I}) = getindex(a, idx[1], idx[2])

getindex(a::MultiArray, ::Colon) = linearize_coefficients(a)


# Support linear indexing too
setindex!(a::MultiArray, v, idx::Int) = setindex!(a, v, multilinear_index(a, idx))

setindex!(a::MultiArray, v, i::Int, j) = a.arrays[i][j] = v

setindex!(a::MultiArray, v, idx::Tuple{Int,Any}) = setindex!(a, v, idx[1], idx[2])

# Assignment from linear representation
setindex!{T}(a::MultiArray, b::Array{T,1}, ::Colon) =
    delinearize_coefficients!(a, b)

# Assignment from other MultiArray: assign recursively
function setindex!(a::MultiArray, b::MultiArray, ::Colon)
    for (i,el) in enumerate(elements(a))
        el[:] = element(b,i)
    end
    b
end

# Convert a linear index into the MultiArray into the index of a subarray
# and a linear index into that subarray.
function multilinear_index(a::MultiArray, idx::Int)
    i = 0
    while idx > a.offsets[i+1]
        i += 1
    end
    (i,idx-a.offsets[i])
end

linear_index(a::MultiArray, idxn::NTuple{2,Int}) = a.offsets[idxn[1]] + idxn[2]

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

for op in (:≈,)
    @eval $op(a::MultiArray, b::MultiArray) = reduce(&, map($op, elements(a), elements(b)))
end
