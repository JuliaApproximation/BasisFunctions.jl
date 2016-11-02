# diagonal_map.jl

"""
A diagonal map acts on each of the components of x separately:
y = f(x) becomes y_i = f_i(x_i)
"""
immutable DiagonalMap{N,MAPS} <: AbstractMap
    # maps has an indexable and iterable type, for example a tuple of maps
    maps    ::  MAPS
end

DiagonalMap(maps) = DiagonalMap{length(maps),typeof(maps)}(maps)

eltype(dmap::DiagonalMap) = promote_type(map(eltype, dmap.maps)...)

ndims{N}(::DiagonalMap{N}) = N

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

is_linear(map::DiagonalMap{1}) = is_linear(map.maps[1])
is_linear(map::DiagonalMap{2}) = is_linear(map.maps[1]) & is_linear(map.maps[2])
is_linear(map::DiagonalMap{3}) = is_linear(map.maps[1]) & is_linear(map.maps[2]) & is_linear(map.maps[3])
is_linear(map::DiagonalMap{4}) = is_linear(map.maps[1]) & is_linear(map.maps[2]) & is_linear(map.maps[3]) & is_linear(map.maps[4])
is_linear{N}(map::DiagonalMap{N}) = reduce(&, map(is_linear, map.maps))

inv(dmap::DiagonalMap) = DiagonalMap(map(inv, dmap.maps))

isreal(dmap::DiagonalMap) = reduce(&, map(isreal, elements(dmap)))


# TODO: implement jacobian
# jacobian(map::DiagonalMap, x) =
