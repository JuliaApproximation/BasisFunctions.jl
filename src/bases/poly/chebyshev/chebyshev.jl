include("ChebyshevT.jl")
include("ChebyshevU.jl")
include("diff.jl")
include("orth.jl")

using DomainSets: WrappedMap

const CosineMap{T} = WrappedMap{T,typeof(cos)}
const SineMap{T} = WrappedMap{T,typeof(sin)}
const ArcCosineMap{T} = WrappedMap{T,typeof(acos)}
const ArcSineMap{T} = WrappedMap{T,typeof(asin)}

cosinemap(::Type{T} = Float64) where {T} = convert(Map{T}, cos)
sinemap(::Type{T} = Float64) where {T} = convert(Map{T}, sin)
arccosinemap(::Type{T} = Float64) where {T} = convert(Map{T}, acos)
arcsinemap(::Type{T} = Float64) where {T} = convert(Map{T}, asin)

inv(::CosineMap{T}) where {T} = arccosinemap(T)
inv(::ArcCosineMap{T}) where {T} = cosinemap(T)
inv(::SineMap{T}) where {T} = arcsinemap(T)
inv(::ArcSineMap{T}) where {T} = sinemap(T)

jacobian(::CosineMap, x) = -sin(x)
jacobian(::SineMap{T}) where {T} = cosinemap(T)
