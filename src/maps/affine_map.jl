# affine_map.jl

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

isreal(map::AffineMap) = isreal(map.A) && isreal(map.B)

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
