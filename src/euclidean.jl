# euclidean.jl

abstract EuclideanSpace{T}

length(b::EuclideanSpace) = b.length

dim(b::EuclideanSpace) = 1

size(b::EuclideanSpace) = (length(b),)

immutable Rn{T} <: EuclideanSpace{T}
    length  ::  Int
end

eltype{T}(b::Rn{T}) = T

immutable Cn{T} <: EuclideanSpace{T}
    length  ::  Int
end

eltype{T}(b::Cn{T}) = Complex{T}

isreal(::Cn) = False()
isreal{T}(::Type{Cn{T}}) = False


