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

forward_map{N}(dmap::DiagonalMap{N}, x) = SVector{N}(map(forward_map, dmap.maps, x))
inverse_map{N}(dmap::DiagonalMap{N}, x) = SVector{N}(map(inverse_map, dmap.maps, x))

inv(dmap::DiagonalMap) = DiagonalMap(map(inv, dmap.maps))

for op in (:is_linear, :isreal)
    @eval $op(dmap::DiagonalMap) = reduce(&, map($op, elements(dmap)))
end


# TODO: implement jacobian
# jacobian(map::DiagonalMap, x) =
