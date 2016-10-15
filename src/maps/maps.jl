# maps.jl

"""
A map is any transformation of the form y = f(x).
"""
abstract AbstractMap

inverse_map(map::AbstractMap, y) = forward_map(y, inv(map))

(*)(map::AbstractMap, x) = forward_map(map, x)

(\)(map::AbstractMap, y) = inverse_map(map, y)


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


"""
An affine map has the form y = A*x + B.
The fields A and B can be anything that can multiply a vector (A) and be added
to a vector (B). In higher dimensions, one would expect A and B to be static
arrays. In that case, the application of the map does not allocate memory.
"""
immutable AffineMap{TA,TB} <: AbstractMap
    A   ::  TA
    B   ::  TB
end

(m::AffineMap)(x) = forward_map(m, x)

forward_map(map::AffineMap, x) = map.A * x + map.B

inverse_map(map::AffineMap, y) = map.A \ (y-map.B)

inv(map::AffineMap) = AffineMap(inv(map.A), -map.A\map.B)

jacobian(map::AffineMap, x) = map.A * x


# Some useful functions
linearmap(a, b) = AffineMap(a, b)

"Map the interval [a,b] to the interval [c,d]."
interval_map(a, b, c, d) = linearmap((d-c)/(b-a), c - a*(d-c)/(b-a))

scaling_map(a) = AffineMap(a, 0)
scaling_map(a, b...) = DiagonalMap(map(scaling_map, (a, b...)))


"""
A diagonal map acts on each of the components of x separately:
y = f(x) becomes y_i = f_i(x_i)
"""
immutable DiagonalMap{N,MAPS} <: AbstractMap
    # maps has an indexable and iterable type, for example a tuple of maps
    maps    ::  MAPS
end

DiagonalMap(maps) = DiagonalMap{length(maps),typeof(maps)}(maps)

(m::DiagonalMap)(x) = forward_map(m, x)

elements(map::DiagonalMap) = map.maps
element(map::DiagonalMap, i::Int) = map.maps[i]
element(map::DiagonalMap, range::Range) = DiagonalMap(map.maps[range])
composite_length(map::DiagonalMap) = length(elements(map))

⊗(map1::AbstractMap, map2::AbstractMap) = DiagonalMap((map1,map2))
⊗(map1::DiagonalMap, map2::AbstractMap) = DiagonalMap((elements(map1)...,map2))
⊗(map1::AbstractMap, map2::DiagonalMap) = DiagonalMap((map1,elements(map2)...))
⊗(map1::DiagonalMap, map2::DiagonalMap) = DiagonalMap((elements(map1)...,elements(map2)...))

# TODO: provide dimension-independent implementation
forward_map(dmap::DiagonalMap{1}, x) = SVector(dmap.maps[1]*x[1])
forward_map(dmap::DiagonalMap{2}, x) = SVector(dmap.maps[1]*x[1], dmap.maps[2]*x[2])
forward_map(dmap::DiagonalMap{3}, x) = SVector(dmap.maps[1]*x[1], dmap.maps[2]*x[2], dmap.maps[3]*x[3])
forward_map(dmap::DiagonalMap{4}, x) = SVector(dmap.maps[1]*x[1], dmap.maps[2]*x[2], dmap.maps[3]*x[3], dmap.maps[4]*x[4])

inv(dmap::DiagonalMap) = DiagonalMap(map(inv, dmap.maps))

# TODO: implement jacobian
# jacobian(map::DiagonalMap, x) =

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
forward_map_rec(x, map1::AbstractMap, maps::AbstractMap...) = forward_map_rec(map1*x, maps)

inv(cmap::CompositeMap) = CompositeMap(reverse(map(inv, cmap.maps)))

# TODO: implement jacobian
# jacobian(map::CompositeMap, x) =

(*)(map1::AbstractMap, map2::AbstractMap) = CompositeMap((map1,map2))
(*)(map1::CompositeMap, map2::AbstractMap) = CompositeMap((elements(map1)...,map2))
(*)(map1::AbstractMap, map2::CompositeMap) = CompositeMap((map1,elements(map2)...))
(*)(map1::CompositeMap, map2::CompositeMap) = CompositeMap((elements(map1)...,elements(map2)...))


########################
# Arithemetic
########################

(*)(map1::AffineMap, map2::AffineMap) = AffineMap(map1.A*map2.A, map1.A*map2.B+map1.B)

(*)(a::Number, m::AbstractMap) = scaling_map(a) * m

(+)(m::AffineMap, x) = AffineMap(m.a, m.b+x)

(+)(m1::AffineMap, m2::AffineMap) = AffineMap(m1.a+m2.a, m1.b+m2.b)
