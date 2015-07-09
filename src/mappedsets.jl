# mappedsets.jl

# An AbstractMappedSet collects all sets that are defined in terms of another set through a mapping.
abstract AbstractMappedSet{S,N,T} <: AbstractFunctionSet{N,T}

set(s::AbstractMappedSet) = s.set

# Delegate methods to the underlying set
for op in (:is_basis, :isreal, :length, :eltype, :index_dim, :is_orthogonal)
    @eval $op(s::AbstractMappedSet) = $op(set(s))
end

# Delegate methods invoked with a type to the underlying set
for op in (:is_basis, :isreal, :eltype, :index_dim, :is_orthogonal)
    @eval $op{S,N,T}(::Type{AbstractMappedSet{S,N,T}}) = $op(S)
    @eval $op{S <: AbstractMappedSet}(::Type{S}) = $op(super(S))
end


# A set defined via a linear map.
# The underlying set S should support left(s) and right(s).
immutable LinearMappedSet{S <: AbstractFunctionSet1d,T} <: AbstractMappedSet{S,1,T}
    set     ::  S
    a       ::  T
    b       ::  T

    LinearMappedSet(set::AbstractFunctionSet1d{T}, a::T, b::T) = new(set, a, b)
end

LinearMappedSet{T}(s::AbstractFunctionSet1d{T}, a, b) = LinearMappedSet{typeof(s),T}(s, T(a), T(b))

left(s::LinearMappedSet) = s.a
right(s::LinearMappedSet) = s.b

name(s::LinearMappedSet) = name(set(s)) * ", mapped to [" * left(s) * "," * right(s) * "]"

# Map the point x in [c,d] to a point in [a,b]
map_linear(x, a, b, c, d) = a + (x-c)/(d-c) * (b-a)

# Map the point y in [a,b] to a point in [c,d]
imap_linear(y, a, b, c, d) = map_linear(y, c, d, a, b)

mapx(s::LinearMappedSet, x) = map_linear(x, s.a, s.b, left(set(s)), right(set(s)))

imapx(s::LinearMappedSet, y) = imap_linear(y, s.a, s.b, left(set(s)), right(set(s)))

call(s::LinearMappedSet, idx::Int, y) = call(set(s), idx, imapx(s,y))

grid(s::LinearMappedSet) = LinearMappedGrid(grid(set(s)), left(s), right(s))


rescale(s::AbstractFunctionSet1d, a, b) = LinearMappedSet(s, a, b)

# avoid multiple linear mappings
rescale(s::LinearMappedSet, a, b) = LinearMappedSet(set(s), a, b)

# Preserve tensor product structure
function rescale{TS,SN,LEN}(s::TensorProductSet{TS,SN,LEN}, a::AbstractArray, b::AbstractArray)
    scaled_sets = [ rescale(set(s,i), a[i], b[i]) for i in 1:LEN]
    TensorProductSet(scaled_sets...)
end


(*){T <: Number}(s::AbstractFunctionSet1d, a::T) = rescale(s, a*left(s), a*right(s))
(*){T <: Number}(a::T, s::AbstractFunctionSet1d) = s*a

(+){T <: Number}(s::AbstractFunctionSet1d, a::T) = rescale(s, a+left(s), a+right(s))
(+){T <: Number}(a::T, s::AbstractFunctionSet1d) = s+a





