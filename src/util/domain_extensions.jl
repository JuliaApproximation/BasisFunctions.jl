# domain_extensions.jl

is_compatible(map1::AbstractMap, map2::AbstractMap) = map1==map2

"Assign a floating point type to a domain element type T."
float_type(::Type{T}) where {T <: Real} = T
float_type(::Type{Complex{T}}) where {T <: Real} = Complex{T}
float_type(::Type{SVector{N,T}}) where {N,T} = T
float_type(::Type{NTuple{N,T}}) where {N,T} = T

# Fallback: we return T itself
float_type(::Type{T}) where {T} = Float64

dimension(::Type{T}) where {T <: Number} = 1
dimension(::Type{SVector{N,T}}) where {N,T <: Number} = N
dimension(::Type{NTuple{N,T}}) where {N,T <: Number} = N
dimension(::Type{Tuple{A}}) where {A} = 1
dimension(::Type{Tuple{A,B}}) where {A,B} = 2
dimension(::Type{Tuple{A,B,C}}) where {A,B,C} = 3
dimension(::Type{Tuple{A,B,C,D}}) where {A,B,C,D} = 4
