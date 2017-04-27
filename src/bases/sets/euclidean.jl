# euclidean.jl

abstract type DiscreteVectorSpace{T} <: FunctionSet{1,T} end

length(b::DiscreteVectorSpace) = b.n

immutable DiscreteSet{N,T} <: FunctionSet{N,T}
    n   ::  Int
end

length(s::DiscreteSet) = s.n

isreal{N,T}(s::DiscreteSet{N,T}) = isreal(T)

set_promote_eltype{N,T,S}(set::DiscreteSet{N,T}, ::Type{S}) = DiscreteSet{N,S}(length(set))

resize{N,T}(s::DiscreteSet{N,T}, n) = DiscreteSet{N,T}(n)

struct Rn{T} <: DiscreteVectorSpace{T}
    n   ::  Int
end


struct Cn{T} <: DiscreteVectorSpace{T}
    n   ::  Int
end

isreal(c::Cn) = false

promote_eltype{T,S}(b::Rn{T}, ::Type{S}) = Rn{promote_type(T,S)}(b.n)
promote_eltype{T,S}(b::Cn{T}, ::Type{S}) = Cn{promote_type(T,S)}(b.n)

resize{T}(b::Rn{T}, n) = Rn{T}(n)
resize{T}(b::Cn{T}, n) = Cn{T}(n)
