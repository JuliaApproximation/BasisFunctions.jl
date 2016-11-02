# maps.jl

"""
A map is any transformation of the form y = f(x).
"""
abstract AbstractMap

inverse_map(map::AbstractMap, y) = forward_map(y, inv(map))

(*)(map::AbstractMap, x) = forward_map(map, x)

(\)(map::AbstractMap, y) = inverse_map(map, y)

is_linear(map::AbstractMap) = False()

isreal(map::AbstractMap) = true

"""
Return the matrix and vector of a linear map.
"""
matrix_vector(map::AbstractMap) = matrix_vector(is_linear(map), map)

function matrix_vector(::False, map::AbstractMap)
    println("In matrix_vector: map ", typeof(map), " is not a linear map.")
	throw(ExceptionError())
end

function matrix_vector(::True, map::AbstractMap)
    N = ndims(map)
    T = eltype(map)
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


"""
The identity map.
"""
immutable IdentityMap <: AbstractMap
end

(m::IdentityMap)(x) = forward_map(m, x)

forward_map(map::IdentityMap, x) = x

inverse_map(map::IdentityMap, y) = y

inv(map::IdentityMap) = map

jacobian(map::IdentityMap, x) = x

is_linear(map::IdentityMap) = True()

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

forward_map(map::CompositeMap, x) = forward_map_rec(x, map.maps...)

forward_map_rec(x) = x
forward_map_rec(x, map1::AbstractMap, maps::AbstractMap...) = forward_map_rec(map1*x, maps...)

inv(cmap::CompositeMap) = CompositeMap(reverse(map(inv, cmap.maps)))

isreal(cmap::CompositeMap) = reduce(&, map(isreal, elements(cmap)))

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
