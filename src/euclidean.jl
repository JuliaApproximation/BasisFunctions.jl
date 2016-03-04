# euclidean.jl

abstract DiscreteVectorSpace{T} <: FunctionSet{1,T}

length(b::DiscreteVectorSpace) = b.n


immutable Rn{T} <: DiscreteVectorSpace{T}
    n   ::  Int
end


immutable Cn{T} <: DiscreteVectorSpace{T}
    n   ::  Int
end

isreal{B <: Cn}(::Type{B}) = False

promote_eltype{T,S}(b::Rn{T}, ::Type{S}) = Rn{promote_type(T,S)}(b.n)
promote_eltype{T,S}(b::Cn{T}, ::Type{S}) = Cn{promote_type(T,S)}(b.n)

resize{T}(b::Rn{T}, n) = Rn{T}(n)
resize{T}(b::Cn{T}, n) = Cn{T}(n)
