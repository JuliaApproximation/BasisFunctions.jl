# slices.jl

module Slices

# In this module we implement `eachslice`, a way of iterating over all slices along one
# dimension in a multidimensional array.

export eachslice, joint, view

abstract SliceIterator{N}

immutable SliceIteratorCartesian{N} <: SliceIterator{N}
    range   ::  CartesianRange{CartesianIndex{N}}
    dim     ::  Int
    len     ::  Int
end


immutable SliceIteratorLinear{N} <: SliceIterator{N}
    range   ::  CartesianRange{CartesianIndex{N}}
    dim     ::  Int
    strides ::  NTuple{N,Int}
    stride  ::  Int
    len     ::  Int
end


abstract SliceIndex

immutable SliceIndexCartesian{N} <: SliceIndex
    cartidx ::  CartesianIndex{N}
    dim     ::  Int
    len     ::  Int
end

immutable SliceIndexLinear{N} <: SliceIndex
    cartidx ::  CartesianIndex{N}
    offset  ::  Int
    stride  ::  Int
    len     ::  Int
end



# TODO: more efficient implementation for all N
function remaining_size{N}(siz::NTuple{N,Int}, dim)
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
onetuple{N}(::Type{Val{N}}) = (1, onetuple(Val{N-1})...)


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
strides{N}(siz::NTuple{N,Int}) = (strides(siz[1:N-1])..., stride(siz,N))


eachslice(a::AbstractArray, dim) = eachslice(Base.linearindexing(a), a, dim)

# Type inference does not work with the N-1 argument below:
#eachslice{T,N}(a::AbstractArray{T,N}, dim) =
#    SliceIterator(CartesianRange(CartesianIndex(onetuple(Val{N-1})), CartesianIndex(remaining_size(size(a),dim))), dim, size(a, dim))

# So we do a generated function for the time being:
@generated function eachslice{T,N}(::Base.LinearSlow, a::AbstractArray{T,N}, dim)
    one_tuple = onetuple(Val{N-1})
    quote
        SliceIteratorCartesian(
            CartesianRange( CartesianIndex($one_tuple),
                            CartesianIndex(remaining_size(size(a), dim))),
                            dim, size(a, dim) )
    end
end

@generated function eachslice{T,N}(::Base.LinearFast, a::AbstractArray{T,N}, dim)
    one_tuple = onetuple(Val{N-1})
    quote
        SliceIteratorLinear(
            CartesianRange( CartesianIndex($one_tuple),
                            CartesianIndex(remaining_size(size(a), dim))),
                            dim, substrides(size(a), dim), stride(size(a), dim), size(a, dim) )
    end
end

Base.start(it::SliceIteratorCartesian) = SliceIndexCartesian(start(it.range), it.dim, it.len)

Base.next(it::SliceIteratorCartesian, sidx::SliceIndex) =
    (sidx, SliceIndexCartesian(next(it.range, sidx.cartidx)[2], it.dim, it.len))

Base.done(it::SliceIteratorCartesian, sidx::SliceIndex) = done(it.range, sidx.cartidx)

function Base.getindex{T}(a::AbstractArray{T,2}, sidx::SliceIndexCartesian{1}, i::Int)
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


Base.start(it::SliceIteratorLinear) = SliceIndexLinear(start(it.range), 1, it.stride, it.len)

function Base.next(it::SliceIteratorLinear, sidx::SliceIndex)
    nextidx = next(it.range, sidx.cartidx)[2]
    (sidx, SliceIndexLinear(nextidx, offset(it.strides, nextidx), it.stride, it.len))
end

Base.done(it::SliceIteratorLinear, sidx::SliceIndex) = done(it.range, sidx.cartidx)


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



immutable JointIterator{I1,I2}
    iter1   ::  I1
    iter2   ::  I2
end

joint(iter1, iter2) = JointIterator(iter1, iter2)

Base.start(it::JointIterator) = (start(it.iter1), start(it.iter2))

function Base.next(it::JointIterator, state)
    el1, state1 = next(it.iter1, state[1])
    el2, state2 = next(it.iter2, state[2])
    ((el1,el2), (state1, state2))
end

Base.done(it::JointIterator, state) = done(it.iter1, state[1]) || done(it.iter2, state[2])

Base.sub(a::AbstractArray, sidx::SliceIndexLinear) =
    sub(a, sidx.offset:sidx.stride:sidx.offset+(sidx.len-1)*sidx.stride)


import ArrayViews: view

view(a::AbstractArray, sidx::SliceIndexLinear) =
    view(a, sidx.offset:sidx.stride:sidx.offset+(sidx.len-1)*sidx.stride)

end # module
