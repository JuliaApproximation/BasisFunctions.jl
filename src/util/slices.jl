# slices.jl

module Slices

# In this module we implement `eachslice`, a way of iterating over all slices along one
# dimension in a multidimensional array.

export eachslice, joint, view

abstract type SliceIterator{N} end

abstract type SliceIndex end

struct SliceIteratorCartesian{N} <: SliceIterator{N}
    range   ::  CartesianIndices{N}
    dim     ::  Int
    len     ::  Int
end

struct SliceIteratorLinear{N} <: SliceIterator{N}
    range   ::  CartesianIndices{N}
    dim     ::  Int
    strides ::  NTuple{N,Int}
    stride  ::  Int
    len     ::  Int
end

struct SliceIndexCartesian{N} <: SliceIndex
    cartidx ::  CartesianIndex{N}
    dim     ::  Int
    len     ::  Int
end

struct SliceIndexLinear{N} <: SliceIndex
    cartidx ::  CartesianIndex{N}
    offset  ::  Int
    stride  ::  Int
    len     ::  Int
end


# TODO: more efficient implementation for all N
function remaining_size(siz::NTuple{N,Int}, dim) where {N}
    if dim == 1
        tuple(siz[2:N]...)
    elseif 1 < dim < N
        tuple(siz[1:dim-1]..., siz[dim+1:N]...)
    elseif dim == N
        tuple(siz[1:N-1]...)
    else
        throw(BoundsError())
    end
end

function remaining_size(siz::NTuple{2,Int}, dim)
    if dim == 1
        tuple(siz[2])
    elseif dim ==2
        tuple(siz[1])
    else
        throw(BoundsError())
    end
end

function remaining_size(siz::NTuple{3,Int}, dim)
    if dim == 1
        tuple(siz[2], siz[3])
    elseif dim ==2
        tuple(siz[1], siz[3])
    elseif dim == 3
        tuple(siz[1], siz[2])
    else
        throw(BoundsError())
    end
end


# TODO: more efficient implementation for all N
onetuple(::Type{Val{1}}) = (1,)
onetuple(::Type{Val{2}}) = (1, 1)
onetuple(::Type{Val{3}}) = (1, 1, 1)
onetuple(::Type{Val{4}}) = (1, 1, 1, 1)
# onetuple{N}(::Type{Val{N}}) = (1, onetuple(Val{N-1})...)


function stride(siz, dim)
    s = 1
    for i = 1:dim-1
        s *= siz[i]
    end
    s
end

strides(siz::NTuple{1,Int}) = (1,)
strides(siz::NTuple{2,Int}) = (1, siz[1])
strides(siz::NTuple{3,Int}) = (1, siz[1], siz[1]*siz[2])
strides(siz::NTuple{4,Int}) = (1, siz[1], siz[1]*siz[2], siz[1]*siz[2]*siz[3])

function substrides(siz::NTuple{2,Int}, dim)
    if dim == 1
        (siz[1],)
    elseif dim == 2
        (1,)
    else
        throw(BoundsError())
    end
end

function substrides(siz::NTuple{3,Int}, dim)
    if dim == 1
        (siz[1], siz[1]*siz[2])
    elseif dim == 2
        (1, siz[1]*siz[2])
    else # dim == 3
        (1, siz[1])
    end
end


# Todo: efficient implementation for general N
strides(siz::NTuple{N,Int}) where {N} = (strides(siz[1:N-1])..., stride(siz,N))


eachslice(a::AbstractArray, dim) = eachslice(Base.IndexStyle(a), a, dim)

# Type inference does not work with the N-1 argument below:
#eachslice{T,N}(a::AbstractArray{T,N}, dim) =
#    SliceIterator(CartesianIndices(CartesianIndex(onetuple(Val{N-1})), CartesianIndex(remaining_size(size(a),dim))), dim, size(a, dim))

# So we do a generated function for the time being:
eachslice(::Base.IndexLinear, a::AbstractArray, dim) =
    SliceIteratorLinear(
        CartesianIndices(remaining_size(size(a), dim)),
                        dim, substrides(size(a), dim), stride(size(a), dim), size(a, dim) )

function Base.iterate(it::SliceIteratorCartesian)
    first_tuple = iterate(it.range)
    if first_tuple != nothing
        first_item, first_state = first_tuple
        SliceIndexCartesian(first_item, it.dim, it.len), SliceIndexCartesian(first_item, it.dim, it.len)
    end
end

function Base.iterate(it::SliceIteratorCartesian, state)
    next_tuple = iterate(it.range, state.cartidx)
    if next_tuple != nothing
        next_item, next_state = next_tuple
        SliceIndexCartesian(next_item, it.dim, it.len), SliceIndexCartesian(next_item, it.dim, it.len)
    end
end

Base.eltype(::Type{SliceIteratorCartesian}) = SliceIndexCartesian


function Base.getindex(a::AbstractArray{T,2}, sidx::SliceIndexCartesian{1}, i::Int) where {T}
    if sidx.dim == 1
        a[i, sidx.cartidx[1]]
    else
        a[sidx.cartidx[1], i]
    end
end


offset(strides::NTuple{1,Int}, idx::CartesianIndex{1}) = 1 + (idx[1]-1) * strides[1]

offset(strides::NTuple{2,Int}, idx::CartesianIndex{2}) =
    1 + (idx[1]-1) * strides[1] + (idx[2]-1) * strides[2]

offset(strides::NTuple{3,Int}, idx::CartesianIndex{3}) =
    1 + (idx[1]-1) * strides[1] + (idx[2]-1) * strides[2] + (idx[3]-1) * strides[3]

function Base.iterate(it::SliceIteratorLinear)
    first_tuple = iterate(it.range)
    if first_tuple != nothing
        first_item, first_state = first_tuple

        index = SliceIndexLinear(first_item, 1, it.stride, it.len)
        index, index
    end
end

function Base.iterate(it::SliceIteratorLinear, state)
    idx = state.cartidx
    next_tuple = iterate(it.range, idx)
    if next_tuple != nothing
        next_item, next_state = next_tuple

        index = SliceIndexLinear(next_item, offset(it.strides, next_item), it.stride, it.len)
        index, index
    end
end

Base.eltype(::Type{SliceIteratorLinear}) = SliceIndexLinear


Base.getindex(a::AbstractArray, sidx::SliceIndexLinear, i::Int) = a[sidx.offset + (i-1)*sidx.stride]

Base.setindex!(a::AbstractArray, v, sidx::SliceIndexLinear, i::Int) = a[sidx.offset + (i-1)*sidx.stride] = v

function Base.setindex!(a::AbstractArray, v::AbstractVector, sidx::SliceIndexLinear)
    ofs = sidx.offset
    for i in eachindex(v)
        a[ofs] = v[i]
        ofs += sidx.stride
    end
    v
end



struct JointIterator{I1,I2}
    iter1   ::  I1
    iter2   ::  I2
end

joint(iter1, iter2) = JointIterator(iter1, iter2)



function Base.iterate(it::JointIterator)
    first_tuple1 = iterate(it.iter1)
    first_tuple2 = iterate(it.iter2)
    if (first_tuple1 != nothing) && (first_tuple2 != nothing)
        first_item1, first_state1 = first_tuple1
        first_item2, first_state2 = first_tuple2

        (first_item1, first_item2), (first_state1, first_state2)
    end
end

function Base.iterate(it::JointIterator, state)
    state1, state2 = state
    next_tuple1 = iterate(it.iter1, state1)
    next_tuple2 = iterate(it.iter2, state2)

    if (next_tuple1 != nothing) && (next_tuple2 != nothing)
        next_item1, next_state1 = next_tuple1
        next_item2, next_state2 = next_tuple2
        (next_item1, next_item2), (next_state1, next_state2)
    end
end

Base.eltype(::Type{JointIterator}) = Tuple{SliceIndex,SliceIndex}

Base.view(a::AbstractArray, sidx::SliceIndexLinear) =
    view(a, sidx.offset:sidx.stride:sidx.offset+(sidx.len-1)*sidx.stride)

end # module
