# euclidean.jl

"""
A `DiscreteVectorSpace{S,T}` is a dictionary where `S` is a discrete type, e.g.,
the integers. Expansions in this discrete set typically constitute a vector space.
"""
abstract type DiscreteVectorSpace{S,T} <: Dictionary{S,T}
end

coefficient_type(::Type{DiscreteVectorSpace{S,T}}) where {S,T} = T

eval_element(set::DiscreteVectorSpace{S,T}, idx, x) where {S,T} =
    idx == x ? one(T) : zero(T)

is_discrete(dict::Dictionary) = false
is_discrete(dict::DiscreteVectorSpace) = true



immutable DiscreteSet{T} <: DiscreteVectorSpace{Int,T}
    n   ::  Int
end

length(s::DiscreteSet) = s.n

dict_promote_domaintype(set::DiscreteSet, ::Type{S}) where {S} = DiscreteSet{S}(length(set))

resize(s::DiscreteSet, n) = DiscreteSet(n)
