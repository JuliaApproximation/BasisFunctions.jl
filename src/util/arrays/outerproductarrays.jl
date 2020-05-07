abstract type AbstractOuterProductArray{T,N} <: AbstractArray{T,N} end

elements(opa::AbstractOuterProductArray) = opa.vectors
element(opa::AbstractOuterProductArray, i) = opa.vectors[i]
size(A::AbstractOuterProductArray) = map(length, elements(A))
axes(A::AbstractOuterProductArray) = map(x->axes(x,1), elements(A))

@inline function Base.getindex(F::AbstractOuterProductArray, k::Integer)
    @boundscheck checkbounds(F, k)
    unsafe_getindex(F, (k,))
end

@inline function Base.getindex(F::AbstractOuterProductArray{T, N}, kj::Vararg{<:Integer, N}) where {T, N}
    @boundscheck checkbounds(F, kj...)
    unsafe_getindex(F, kj)
end

@inline Base.unsafe_getindex(F::AbstractOuterProductArray{T,N}, kj::NTuple{N,Int}) where {T,N} =
    prod(map(getindex, elements(F), kj))

export OuterProductArray
struct OuterProductArray{T,N,V} <: AbstractOuterProductArray{T,N}
    vectors   ::  V
    OuterProductArray(v::AbstractVector...) =
        new{promote_type(map(eltype,v)...),length(v),typeof(v)}(v)
end


tensorproduct(v::AbstractVector...)  =
    OuterProductArray(v...)
tensorproduct(v::GridArrays.AbstractIntervalGrid...)  =
    ProductGrid(v...)

struct OPArrayStyle{N} <: Broadcast.AbstractArrayStyle{N} end
Broadcast.BroadcastStyle(::Type{<:OuterProductArray{T,N}}) where {T,N} = OPArrayStyle{N}()
Base.copyto!(dest::OuterProductArray, bc::Broadcast.Broadcasted{<:OPArrayStyle}) =
    OuterProductArray(map(argsi->broadcast(bc.f, argsi...), zip(map(elements, bc.args)...))...)
Base.copy(a::OuterProductArray) = OuterProductArray(map(copy, elements(a))...)
Base.similar(a::OuterProductArray) = OuterProductArray(map(similar, elements(a))...)
Base.similar(::Type{OuterProductArray{T}}, shape::Tuple{Union{Integer, Base.OneTo},Vararg{Union{Integer,Base.OneTo},N} where N}) where T =
    OuterProductArray(map(x->Base.similar(Vector{T}, x), shape)...)
Base.similar(bc::Broadcast.Broadcasted{OPArrayStyle{N}},::Type{ElType}) where {N,ElType} =
    Base.similar(OuterProductArray{ElType}, axes(bc))
function Base.copyto!(dest::OuterProductArray, src::OuterProductArray)
    for (deste,srce) in zip(elements(dest),elements(src))
        copy!(deste,srce)
    end
end
