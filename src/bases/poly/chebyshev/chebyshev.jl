include("ChebyshevT.jl")
include("ChebyshevU.jl")
include("diff.jl")
include("orth.jl")

using FunctionMaps: WrappedMap

const CosineMap{T} = WrappedMap{T,typeof(cos)}
const SineMap{T} = WrappedMap{T,typeof(sin)}
const ArcCosineMap{T} = WrappedMap{T,typeof(acos)}
const ArcSineMap{T} = WrappedMap{T,typeof(asin)}

cosinemap(::Type{T} = Float64) where {T} = convert(Map{T}, cos)
sinemap(::Type{T} = Float64) where {T} = convert(Map{T}, sin)
arccosinemap(::Type{T} = Float64) where {T} = convert(Map{T}, acos)
arcsinemap(::Type{T} = Float64) where {T} = convert(Map{T}, asin)

FunctionMaps.inverse(::CosineMap{T}) where {T} = arccosinemap(T)
FunctionMaps.inverse(::ArcCosineMap{T}) where {T} = cosinemap(T)
FunctionMaps.inverse(::SineMap{T}) where {T} = arcsinemap(T)
FunctionMaps.inverse(::ArcSineMap{T}) where {T} = sinemap(T)

FunctionMaps.jacobian(::CosineMap, x) = -sin(x)
FunctionMaps.jacobian(::SineMap{T}) where {T} = cosinemap(T)
