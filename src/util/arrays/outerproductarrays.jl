abstract type AbstractOuterProductArray{T,N} <: AbstractArray{T,N} end

iscomposite(::AbstractOuterProductArray) = true
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

struct OuterProductArray{T,N,V} <: AbstractOuterProductArray{T,N}
    vectors   ::  V
    OuterProductArray(v::AbstractVector...) =
        new{promote_type(map(eltype,v)...),length(v),typeof(v)}(v)
end


tensorproduct(v::AbstractVector...)  =
    OuterProductArray(v...)
