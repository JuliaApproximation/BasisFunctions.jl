# euclidean.jl

"""
A `DiscreteVectorSpace{S,T}` is a dictionary where `S` is a discrete type, e.g.,
the integers. Expansions in this discrete set typically constitute a vector space.
"""
abstract type DiscreteVectorSpace{S,T} <: Dictionary{S,T}
end

coefficient_type(::Type{DiscreteVectorSpace{S,T}}) where {S,T} = T

unsafe_eval_element(set::DiscreteVectorSpace{S,T}, idx, x::S) where {S,T} =
    idx == x ? one(T) : zero(T)

# If x is not of type S, we return zero. Perhaps this should be an error?
unsafe_eval_element(set::DiscreteVectorSpace{S,T}, idx, x) where {S,T} = zero(T)

is_discrete(dict::Dictionary) = false
is_discrete(dict::DiscreteVectorSpace) = true



immutable DiscreteSet{T} <: DiscreteVectorSpace{Int,T}
    n   ::  Int
end

# We set a default codomain type Float64
DiscreteSet(n::Int) = DiscreteSet{Float64}(n)

length(d::DiscreteSet) = d.n

dict_promote_domaintype(d::DiscreteSet, S) = error("The domain type of a discrete set is fixed.")

resize(d::DiscreteSet{T}, n) where {T} = DiscreteSet{T}(n)

domain(d::DiscreteSet) = ClosedInterval{Int}(1, length(d))

in_support(d::DiscreteSet, idx, x::Int) = 1 <= x <= length(d)
in_support(d::DiscreteSet, idx, x) = false
