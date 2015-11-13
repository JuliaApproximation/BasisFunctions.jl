# euclidean.jl

abstract DiscreteVectorSpace{T} <: FunctionSet{1,T}

length(b::DiscreteVectorSpace) = b.length



immutable Rn{T} <: DiscreteVectorSpace{T}
    length  ::  Int
end


immutable Cn{T} <: DiscreteVectorSpace{T}
    length  ::  Int
end

isreal(::Cn) = False()
isreal{T}(::Type{Cn{T}}) = False



