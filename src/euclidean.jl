# euclidean.jl

abstract DiscreteVectorSpace{T} <: FunctionSet{1,T}

length(b::DiscreteVectorSpace) = b.length



immutable Rn{T} <: DiscreteVectorSpace{T}
    length  ::  Int
end


immutable Cn{T} <: DiscreteVectorSpace{T}
    length  ::  Int
end

isreal{B <: Cn}(::Type{B}) = False

similar{T}(b::Rn, ::Type{T}, n) = Rn{T}(n)
similar{T}(b::Cn, ::Type{T}, n) = Cn{T}(n)
