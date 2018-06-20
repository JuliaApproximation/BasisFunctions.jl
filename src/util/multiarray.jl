# multiarray.jl

"""
A `MultiArray` in its simplest form is an array of arrays. It is meant to store
coefficient sets of different expansions together, i.e. each element of the outer
array stores the coefficients of an expansion. This is useful as a representation
for MultiDict's.

A MultiArray has one outer array, of which each element is an inner array. The
inner arrays can represent the coefficients of an expansion. In case the
outer array is actually of type Array, the number of inner arrays may be large.
However, for good efficiency, all inner arrays should have the same type. The outer
array can also be a tuple, in which case the inner arrays can have mixed types,
but there should not be too many.

The MultiArray can be indexed with a linear index, which iterates over all the
elements of the subarray's linearly. This index would correspond to the linear
index of a MultiDict. Alternatively, the MultiArray can be index with a native index,
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
struct MultiArray{A,T}
    # The underlying data is usually an array of arrays, but may be more general.
    # At the least, it is an indexable set that supports a linear index.
    arrays  ::  A
    # The cumulative sum of the lengths of the subarrays. Used to compute indices.
    # The linear index for the i-th subarray starts at offsets[i]+1.
    offsets ::  Vector{Int}

    MultiArray{A,T}(arrays) where {A,T} = new(arrays, compute_offsets(arrays))

    function MultiArray{A,T}(offsets::Vector{Int}) where {A,T}
        @assert eltype(eltype(A)) == T
        new(zeros_oftype(A, offsets), offsets)
    end
end

"Generate zeros of type `A`, suitable for a `MultiArray{A,T}`."
function zeros_oftype(::Type{A}, offsets::Vector{Int}) where {A}
end

# - the outer array is a Vector
zeros_oftype(::Type{Vector{A}}, offsets::Vector{Int}) where {A} =
    [zeros_oftype(A, offsets[i+1]-offsets[i]) for i in 1:length(offsets)-1]
# - the outer array is an NTuple
function zeros_oftype(::Type{NTuple{N,A}}, offsets::Vector{Int}) where {N,A}
    @assert length(offsets) == N+1
    ntuple(i->zeros_oftype(A, offsets[i+1]-offsets[i]), Val{N})
end

zeros_oftype(::Type{Vector{T}}, len::Int) where {T} = zeros(T, len)
function zeros_oftype(::Type{NTuple{N,T}}, len::Int) where {N,T}
    @assert len == N
    ntuple(i->zero(T), Val{N})
end

function MultiArray(arrays, T = eltype(arrays[1]))
    for array in arrays
        @assert eltype(array) == T
    end
    MultiArray{typeof(arrays),T}(arrays)
end

eltype(::Type{MultiArray{A,T}}) where {A,T} = T

element(a::MultiArray, i) = a.arrays[i]
elements(a::MultiArray) = a.arrays

similar_multiarray(a::MultiArray) = MultiArray(elements(a))

length(a::MultiArray) = a.offsets[end]

unsafe_offsets(a::MultiArray) = a.offsets

# This definition is up for debate: what is the size of a MultiArray?
# By defining it as (length(a),), we can assert elsewhere that the size of a
# multidict equals the size of its multiarray representation.
size(a::MultiArray) = (length(a),)

length(a::MultiArray, i::Int) = length(a.arrays[i])

show(io::IO, a::MultiArray) = show(io, a.arrays)

function linearize_coefficients(a::MultiArray)
    b = zeros(eltype(a), length(a))
    linearize_coefficients!(b, a)
end

function linearize_coefficients!(b::Vector, a::MultiArray)
    @assert length(a) == length(b)
    for (i,j) in enumerate(eachindex(a))
        b[i] = a[j]
    end
    b
end

function delinearize_coefficients(a::MultiArray, b::Vector)
    multiarray = similar_multiarray(a)
    delinearize_coefficients!(multiarray, b)
    multiarray
end

function delinearize_coefficients!(a::MultiArray, b::Vector)
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

# Convert a linear index into the MultiArray into the index of a subarray
# and a linear index into that subarray.
multilinear_index(a::MultiArray, idx::Int) = offsets_multilinear_index(unsafe_offsets(a), idx)

linear_index(a::MultiArray, idx::MultilinearIndex) = offsets_linear_index(unsafe_offets(a), idx)

eachindex(a::MultiArray) = MultilinearIndexIterator(map(length, elements(a)))

function eachindex(s1::MultiArray, s2::MultiArray)
    @assert length(s1) == length(s2)
    if s1.offsets == s2.offsets
        eachindex(s1)
    else
        Base.OneTo(length(s1))
    end
end

function eachindex(s1::MultiArray, s2::AbstractArray)
    @assert length(s1) == length(s2)
    Base.OneTo(length(s1))
end

function eachindex(s1::AbstractArray, s2::MultiArray)
    @assert length(s1) == length(s2)
    Base.OneTo(length(s1))
end


# Support linear indexing
getindex(a::MultiArray, idx::Int) = getindex(a, multilinear_index(a, idx))

# Indexing with two arguments: the first is the number of the subset, the
# second can be any index of the subset.
getindex(a::MultiArray, i::Int, j) = a.arrays[i][j]

# Indexing with tuples allows for recursive indices
getindex(a::MultiArray, idx::Tuple{Int,I}) where {I} = getindex(a, idx[1], idx[2])

getindex(a::MultiArray, ::Colon) = linearize_coefficients(a)


# Similar routines as for getindex above
setindex!(a::MultiArray, v, idx::Int) = setindex!(a, v, multilinear_index(a, idx))
setindex!(a::MultiArray, v, i::Int, j) = a.arrays[i][j] = v
setindex!(a::MultiArray, v, idx::Tuple{Int,Any}) = setindex!(a, v, idx[1], idx[2])

# Assignment from linear representation
setindex!(a::MultiArray, b::Vector, ::Colon) =
    delinearize_coefficients!(a, b)

# Assignment from other MultiArray: assign recursively
function setindex!(a::MultiArray, b::MultiArray, ::Colon)
    for (i,el) in enumerate(elements(a))
        el[:] = element(b,i)
    end
    b
end



################
# Arithmetic
################

function fill!(a::MultiArray, val)
    for array in elements(a)
        fill!(array, val)
    end
end

# for op in (:+, :*, :-, :/, :.*, :.+, :.-, :./, :.\, :.^, :.÷)
#     @eval $op(a::MultiArray, b::MultiArray) = MultiArray([$op(a.arrays[i], b.arrays[i]) for i in 1:length(a.arrays)])
#     @eval $op(a::Number, b::MultiArray) = MultiArray([$op(a, array) for array in b.arrays])
#     @eval $op(a::MultiArray, b::Number) = MultiArray([$op(array, b) for array in a.arrays])
# end
for op in (:+, :*, :-, :/)
    @eval $op(a::MultiArray, b::MultiArray) = MultiArray([$op(a.arrays[i], b.arrays[i]) for i in 1:length(a.arrays)])
    @eval $op(a::Number, b::MultiArray) = MultiArray([$op(a, array) for array in b.arrays])
    @eval $op(a::MultiArray, b::Number) = MultiArray([$op(array, b) for array in a.arrays])
end

for op in (:*, :+, :-, :/, :\, :^, :÷)
    @eval broadcast(::typeof($op), a::MultiArray, b::MultiArray) = MultiArray([broadcast($op, a.arrays[i], b.arrays[i]) for i in 1:length(a.arrays)])
    @eval broadcast(::typeof($op), a::Number, b::MultiArray) = MultiArray([broadcast($op, a, array) for array in b.arrays])
    @eval broadcast(::typeof($op), a::MultiArray, b::Number) = MultiArray([broadcast($op, array, b) for array in a.arrays])
end

for op in (:≈,)
    @eval $op(a::MultiArray, b::MultiArray) = reduce(&, map($op, elements(a), elements(b)))
end
