# euclidean.jl

abstract type DiscreteVectorSpace{T} <: FunctionSet{T}
end

length(b::DiscreteVectorSpace) = b.n

immutable DiscreteSet{T} <: FunctionSet{T}
    n   ::  Int
end

length(s::DiscreteSet) = s.n

set_promote_domaintype{T,S}(set::DiscreteSet{T}, ::Type{S}) = DiscreteSet{S}(length(set))

resize(s::DiscreteSet{T}, n) where {T} = DiscreteSet{T}(n)

struct Rn{T} <: DiscreteVectorSpace{T}
    n   ::  Int
end


struct Cn{T} <: DiscreteVectorSpace{T}
    n   ::  Int
end

isreal(c::Cn) = false

set_promote_domaintype(b::Rn{T}, ::Type{S}) where {T,S} = Rn{S}(b.n)
set_promote_domaintype(b::Cn{T}, ::Type{S}) where {T,S} = Cn{S}(b.n)

resize(b::Rn{T}, n) where {T} = Rn{T}(n)
resize(b::Cn{T}, n) where {T} = Cn{T}(n)
