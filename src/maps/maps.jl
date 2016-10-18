# maps.jl

"""
A map is any transformation of the form y = f(x).
"""
abstract AbstractMap

inverse_map(map::AbstractMap, y) = forward_map(y, inv(map))

(*)(map::AbstractMap, x) = forward_map(map, x)

(\)(map::AbstractMap, y) = inverse_map(map, y)

is_linear(map::AbstractMap) = False()

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

"""
An affine map has the form y = A*x + B.
The fields A and B can be anything that can multiply a vector (A) and be added
to a vector (B). In higher dimensions, one would expect A and B to be static
arrays. In that case, the application of the map does not allocate memory.

The field B can be a number, as it will be broadcasted to a vector in
computations. Similarly, the field A can be a scalar, in which case it scales
all dimensions by the same factor.
"""
immutable AffineMap{TA,TB} <: AbstractMap
    A   ::  TA
    B   ::  TB
end

(m::AffineMap)(x) = forward_map(m, x)

eltype(map::AffineMap) = promote_type(eltype(map.A),eltype(map.B))

forward_map(map::AffineMap, x) = map.A * x + map.B

inverse_map(map::AffineMap, y) = map.A \ (y-map.B)

inv(map::AffineMap) = affine_inv(map.A, map.B)

affine_inv(a, b) = AffineMap(inv(a), -a\b)

# The special case where b = 0.
function affine_inv(a::AbstractMatrix, b::Number)
    @assert b == 0
    AffineMap(inv(a), 0)
end

# The special case where a = 0.
function affine_inv(a::Number, b::AbstractVector)
    @assert a == 0
    AffineMap(0, -b)
end

jacobian(map::AffineMap, x) = map.A * x

is_linear(map::AffineMap) = True()


# Some useful functions
linearmap(a, b) = AffineMap(a, b)

"Map the interval [a,b] to the interval [c,d]."
interval_map(a, b, c, d) = linearmap((d-c)/(b-a), c - a*(d-c)/(b-a))

scaling_map(a) = AffineMap(a, 0)
scaling_map(a, b) = AffineMap(SMatrix{2,2}(a,0,0,b), 0)
scaling_map(a, b, c) = AffineMap(SMatrix{3,3}(a,0,0, 0,b,0, 0,0,c), 0)
scaling_map(a, b, c, d) = AffineMap(SMatrix{4,4}(a,0,0,0, 0,b,0,0, 0,0,c,0, 0,0,0,d), 0)

"""
Compute the affine map that represents map2*map1, that is:
y = a2*(a1*x+b1)+b2 = a2*a1*x + a2*b1 + b2.
"""
affine_composition(map1::AffineMap, map2::AffineMap) = affine_composition(map1.A, map1.B, map2.A, map2.B)

# We have to compute a matrix a2*a1 and a vector a2*b1+b2.
# We have to be careful to treat the cases where A and/or B are scalars properly,
# since A*x+B sometimes relies on broadcasting, but the dimensions of A*B can be
# unexpected.
# It turns out the only problematic case is A*B when A is a matrix and B is a
# scalar. In that case A*B is again a matrix, but we want it to be a vector.
affine_composition(a1, b1, a2, b2) = AffineMap(a2*a1, affine_composition_vector(a2, b1, b2))

# Now the vector
# - The general expression is fine in most cases
affine_composition_vector(a2, b1, b2) = a2*b1 + b2

# - be careful when a2 is a matrix and b1 a scalar
function affine_composition_vector{M,N}(a2::SMatrix{M,N}, b1::Number, b2)
    a2*(b1*ones(SVector{N,typeof(b1)})) + b2
end

function affine_composition_vector{T}(a2::AbstractArray{T,2}, b1::Number, b2)
    a2*(b1*ones(typeof(b1), size(a2,2))) + b2
end



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
forward_map_rec(x, map1::AbstractMap, maps::AbstractMap...) = forward_map_rec(map1*x, maps...)

inv(cmap::CompositeMap) = CompositeMap(reverse(map(inv, cmap.maps)))

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
