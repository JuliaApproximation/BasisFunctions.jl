# euclidean.jl

abstract type DiscreteVectorSpace{T} <: FunctionSet{T}
end

length(b::DiscreteVectorSpace) = b.n

coefficient_type(::Type{DiscreteVectorSpace{T}}) where {T} = Int

function in_support(set::DiscreteVectorSpace, i, x)
    checkbounds(set, x)
    true
end

eval_element(set::DiscreteVectorSpace, idx, x) = idx == x ? 1 : 0

is_discrete(set::FunctionSet) = false
is_discrete(set::DiscreteVectorSpace) = true



immutable DiscreteSet <: DiscreteVectorSpace{Int}
    n   ::  Int
end

length(s::DiscreteSet) = s.n

set_promote_domaintype(set::DiscreteSet, ::Type{S}) where {S} = DiscreteSet{S}(length(set))

resize(s::DiscreteSet, n) = DiscreteSet(n)
