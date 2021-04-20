# using LazyArrays
# const AbstractOuterProductArray{T,N} = BroadcastArray{T,N,typeof(*),I} where {T,N,I}

# OuterProductArray(arrays::Vararg{<:AbstractVector{T},N}) where {T,N} =
#                  BroadcastArray(*,ntuple(k->reshape(arrays[k],ntuple(k1-> k==k1 ? length(arrays[k]) : 1,Val(N)))  ,Val(N))...)

tensorproduct(v::AbstractVector...)  =
    OuterProductArray(v...)
tensorproduct(v::GridArrays.AbstractIntervalGrid...)  =
    ProductGrid(v...)
abstract type AbstractOuterProductArray{T,N} <: AbstractArray{T,N} end

components(opa::AbstractOuterProductArray) = opa.vectors
size(A::AbstractOuterProductArray) = map(length, components(A))
axes(A::AbstractOuterProductArray) = map(x->axes(x,1), components(A))

@propagate_inbounds function Base.getindex(F::AbstractOuterProductArray, k::Integer)
    @boundscheck checkbounds(F, k)
    unsafe_array_getindex(F, (k,))
end

@propagate_inbounds function Base.getindex(F::AbstractOuterProductArray{T, N}, kj::Vararg{<:Integer, N}) where {T, N}
    @boundscheck checkbounds(F, kj...)
    unsafe_array_getindex(F, kj)
end

unsafe_array_getindex(F::AbstractOuterProductArray{T,N}, kj::NTuple{N,Int}) where {T,N} =
    prod(map(getindex, components(F), kj))


struct OuterProductArray{T,N,V} <: AbstractOuterProductArray{T,N}
    vectors   ::  V
    OuterProductArray(v::AbstractVector...) =
        new{promote_type(map(eltype,v)...),length(v),typeof(v)}(v)
end




struct OPArrayStyle{N} <: Broadcast.AbstractArrayStyle{N} end
Broadcast.BroadcastStyle(::Type{<:OuterProductArray{T,N}}) where {T,N} = OPArrayStyle{N}()
Base.copyto!(dest::OuterProductArray, bc::Broadcast.Broadcasted{<:OPArrayStyle}) =
    OuterProductArray(map(argsi->broadcast(bc.f, argsi...), zip(map(components, bc.args)...))...)
Base.copy(a::OuterProductArray) = OuterProductArray(map(copy, components(a))...)
Base.similar(a::OuterProductArray) = OuterProductArray(map(similar, components(a))...)
Base.similar(::Type{OuterProductArray{T}}, shape::Tuple{Union{Integer, Base.OneTo},Vararg{Union{Integer,Base.OneTo},N} where N}) where T =
    OuterProductArray(map(x->Base.similar(Vector{T}, x), shape)...)
Base.similar(bc::Broadcast.Broadcasted{OPArrayStyle{N}},::Type{ElType}) where {N,ElType} =
    Base.similar(OuterProductArray{ElType}, axes(bc))
function Base.copyto!(dest::OuterProductArray, src::OuterProductArray)
    for (deste,srce) in zip(components(dest),components(src))
        copy!(deste,srce)
    end
end
