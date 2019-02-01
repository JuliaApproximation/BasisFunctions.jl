struct OuterProductArray{T,N} <: AbstractArray{T,N}
    vectors   ::  NTuple{N,Vector{T}}
end

tensorproduct(v::Vector{T}...) where {T} =
    OuterProductArray(v)

size(A::OuterProductArray) = map(size(A.vectors))
