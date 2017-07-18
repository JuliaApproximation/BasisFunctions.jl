# domain_extensions.jl

is_compatible(map1::AbstractMap, map2::AbstractMap) = map1==map2

"Assign a floating point type to a domain element type T."
float_type(::Type{T}) where {T <: Real} = T
float_type(::Type{Complex{T}}) where {T <: Real} = Complex{T}
float_type(::Type{SVector{N,T}}) where {N,T} = T
float_type(::Type{NTuple{N,T}}) where {N,T} = T

# Fallback: we return T itself
float_type(::Type{T}) where {T} = Float64
