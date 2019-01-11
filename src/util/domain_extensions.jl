

iscompatible(map1::AbstractMap, map2::AbstractMap) = map1==map2

"Assign a floating point type to a domain element type T."
float_type(::Type{T}) where {T <: Real} = T
float_type(::Type{Complex{T}}) where {T <: Real} = Complex{T}
float_type(::Type{SVector{N,T}}) where {N,T} = T
float_type(::Type{NTuple{N,T}}) where {N,T} = T

# Fallback: we return Float64
float_type(::Type{T}) where {T} = Float64

dimension(::Type{T}) where {T <: Number} = 1
dimension(::Type{SVector{N,T}}) where {N,T <: Number} = N
dimension(::Type{NTuple{N,T}}) where {N,T <: Number} = N
dimension(::Type{Tuple{A}}) where {A} = 1
dimension(::Type{Tuple{A,B}}) where {A,B} = 2
dimension(::Type{Tuple{A,B,C}}) where {A,B,C} = 3
dimension(::Type{Tuple{A,B,C,D}}) where {A,B,C,D} = 4
dimension(::Type{CartesianIndex{N}}) where {N} = N

# Convenience functions to make some simple domains
interval() = UnitInterval{Float64}()

circle(::Type{T} = Float64) where {T} = UnitCircle{T}()
circle(radius::Number) = radius * circle(float(typeof(radius)))
circle(radius::Number, center::AbstractVector) = circle(radius) + center

sphere(::Type{T} = Float64) where {T} = UnitSphere{T}()
sphere(radius::Number) = radius * sphere(float(typeof(radius)))
sphere(radius::Number, center::AbstractVector) = sphere(radius) + center

disk(::Type{T} = Float64) where {T} = UnitDisk{T}()
disk(radius::Number) = radius * disk(typeof(radius))
disk(radius::Number, center::AbstractVector) = disk(radius) + center

ball(::Type{T} = Float64) where {T} = UnitBall{T}()
ball(radius::Number) = radius * ball(typeof(radius))
ball(radius::Number, center::AbstractVector) = ball(radius) + center

simplex(::Type{Val{N}}, ::Type{T} = Float64) where {T,N} = UnitSimplex{N,T}()

cube(::Type{Val{N}}, ::Type{T} = Float64) where {N,T} = cartesianproduct(UnitInterval{T}(), Val{N})
cube() = cube(Val{3})
rectangle(a, b, c, d) = (a..b) × (c..d)
cube(a, b, c, d, e, f) = (a..b) × (c..d) × (e..f)
# This one is not type-stable
cube(a::NTuple{N,T}, b::NTuple{N,T}) where {N,T} = ProductDomain(map((ai,bi)->ClosedInterval{T}(ai,bi), a, b)...)
# This one isn't either
cube(a::AbstractVector{T}, b::AbstractVector{T}) where {T} = cube(tuple(a...), tuple(b...))

convert(::Type{Domain{T}}, ::UnitInterval{S}) where{T,S} = UnitInterval{T}()
