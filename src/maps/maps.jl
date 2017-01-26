# maps.jl

"""
A map is any transformation of the form y = f(x).
"""
abstract AbstractMap

inverse_map(map::AbstractMap, y) = forward_map(inv(map), y)

(*)(map::AbstractMap, x) = forward_map(map, x)

(\)(map::AbstractMap, y) = inverse_map(map, y)

is_linear(map::AbstractMap) = false

isreal(map::AbstractMap) = true

linearize(map::AbstractMap, x) = (jacobian(map, x), translation_vector(map, x))

"""
Return the matrix and vector of a linear map, with elements of the given type
(which defaults to eltype, if applicable).
"""
function matrix_vector(map::AbstractMap, T = eltype(map))
    is_linear(map) || throw(ExceptionError())
    N = ndims(map)
    I = eye(SMatrix{N,N,T})
    B = map * zeros(SVector{N,T})
    mA = zeros(T,N,N)
    for i in 1:N
        v = I[:,i]
        mA[:,i] = map * v
    end
    A = SMatrix{N,N}(mA)
    A,B
end

is_compatible(m1::AbstractMap, m2::AbstractMap) = m1==m2


"""
The identity map.
"""
immutable IdentityMap <: AbstractMap
end

(m::IdentityMap)(x) = forward_map(m, x)

forward_map(map::IdentityMap, x) = x

inverse_map(map::IdentityMap, y) = y

inv(map::IdentityMap) = map

jacobian{N,T}(map::IdentityMap, x::SVector{N,T}) = eye(SMatrix{N,N,T})

jacobian{T}(map::IdentityMap, x::Vector{T}) = eye(T, length(x), length(x))

is_linear(map::IdentityMap) = true

translation_vector{N,T}(map::IdentityMap, x::SVector{N,T}) = @SVector zeros(T,N)

translation_vector{T}(map::IdentityMap, x::Vector{T}) = zeros(T,length(x))

# dest_type{T}(map::IdentityMap, ::Type{T}) = T


"""
A Cartesion to Polar map. First dimension is interpreted as angle in radians, second as radial distance
A square [-1,1]x[-1,1] is mapped to the unit circle

"""
immutable CartToPolarMap <: AbstractMap
end

(m::CartToPolarMap)(x) = forward_map(m, x)

forward_map{T}(map::CartToPolarMap, x::SVector{2,T}) = SVector{2,T}(((x[2]+1)/2)*cos(pi*x[1]), ((x[2]+1)/2)*sin(pi*x[1]))

inv(map::CartToPolarMap) = PolarToCartMap()

is_linear(map::CartToPolarMap) = false

 
"""
A Polar to Cartesian map. The angle is mapped to the first dimension, radius to the second.
The unit circle is mapped to a square [-1,1]x[-1,1]
"""
immutable PolarToCartMap <: AbstractMap
end

(m::PolarToCartMap)(x) = forward_map(m, x)

forward_map{T}(map::PolarToCartMap, x::SVector{2,T}) = SVector{2,T}(sqrt(x[1]^2+x[2]^2)*2-1, atan2(x[2],x[1])/pi)

inv(map::PolarToCartMap) = CartToPolarMap()

is_linear(map::PolarToCartMap) = false

 

include("affine_map.jl")
include("diagonal_map.jl")


"""
The composition of several maps.
"""
immutable CompositeMap{MAPS} <: AbstractMap
    maps    ::  MAPS
end

(m::CompositeMap)(x) = forward_map(m, x)

elements(map::CompositeMap) = map.maps
element(map::CompositeMap, i::Int) = map.maps[i]
element(map::CompositeMap, range::Range) = CompositeMap(map.maps[range])
composite_length(map::CompositeMap) = length(elements(map))

# dest_type{T}(map::CompositeMap, ::Type{T}) = promote_type(map(m->dest_type(m,T), elements(map))...)

forward_map(map::CompositeMap, x) = forward_map_rec(x, map.maps...)

forward_map_rec(x) = x
forward_map_rec(x, map1::AbstractMap, maps::AbstractMap...) = forward_map_rec(map1*x, maps...)

inv(cmap::CompositeMap) = CompositeMap(reverse(map(inv, cmap.maps)))

for op in (:is_linear, :isreal)
    @eval $op(cmap::CompositeMap) = reduce(&, map($op, elements(cmap)))
end

# TODO: implement jacobian
# jacobian(map::CompositeMap, x) =

(*)(map1::AbstractMap, map2::AbstractMap) = CompositeMap((map2,map1))
(*)(map1::CompositeMap, map2::AbstractMap) = CompositeMap((map2, elements(map1)...))
(*)(map1::AbstractMap, map2::CompositeMap) = CompositeMap((elements(map2)..., map1))
(*)(map1::CompositeMap, map2::CompositeMap) = CompositeMap((elements(map2)...,elements(map1)...))


########################
# Special maps
########################

translation{N,T}(x::SVector{N,T}) = AffineMap(eye(SMatrix{N,N,T}), x)


########################
# Arithmetic
########################

(*)(map1::AffineMap, map2::AffineMap) = affine_composition(map2, map1)

(*)(map1::AbstractMap, map2::AffineMap) = composite_map(map1, map2, is_linear(map1))

composite_map(map1::AbstractMap, map2::AffineMap, ::False) =
    CompositeMap((map2,map1))

function composite_map(map1::AbstractMap, map2::AffineMap, ::True)
    A,B = matrix_vector(map1)
    AffineMap(A,B) * map2
end

(*)(a::Number, m::AbstractMap) = scaling_map(a) * m

(+)(m::AffineMap, x) = AffineMap(m.a, m.b+x)

(+)(m1::AffineMap, m2::AffineMap) = AffineMap(m1.a+m2.a, m1.b+m2.b)
