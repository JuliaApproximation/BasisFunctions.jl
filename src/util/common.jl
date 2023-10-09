
macro add_properties(T, props...)
    e = quote end
    for prop in props
        f = quote $(esc(prop))(a::$(esc(T))) = true end
        append!(e.args, f.args)
    end
    e
end

tolerance(x) = tolerance(typeof(x))
tolerance(::Type{T}) where {T} = tolerance(prectype(T))
tolerance(T::Type{<:AbstractFloat}) = sqrt(eps(T))

default_threshold(x) = default_threshold(typeof(x))
default_threshold(T::Type{<:AbstractFloat}) = 100eps(T)
default_threshold(::Type{T}) where {T} = default_threshold(prectype(T))

iswidertype(::Type{T}, ::Type{T}) where {T} = true
iswidertype(::Type{Float32}, ::Type{Float64}) = true
iswidertype(::Type{Float32}, ::Type{BigFloat}) = true
iswidertype(::Type{Float64}, ::Type{BigFloat}) = true
iswidertype(::Type{S}, ::Type{Complex{T}}) where {S<:AbstractFloat,T<:AbstractFloat} =
    iswidertype(S,T)


linspace(a, b, n=100) = range(a, stop=b, length=n)


# delinearize_coefficients!(dest::BlockVector, src::AbstractVector) =
#     dest[:] .= src[:]
#
# linearize_coefficients!(dest::AbstractVector, src::BlockVector) =
#     dest[:] .= src[:]

delinearize_coefficients!(dest::AbstractArray{T,N}, src::AbstractVector{S}) where {S,T,N} =
    dest[:] .= src[:]

linearize_coefficients!(dest::AbstractVector{T}, src::AbstractArray{S,N}) where {S,T,N} =
    dest[:] .= src[:]

components(bv::BlockVector) = [view(bv, Block(i)) for i in 1:blocklength(bv)]

function BlockArrays.BlockVector(arrays::AbstractVector{T}...) where {T}
    A = BlockArray{T}(undef_blocks, [length(array) for array in arrays])
    for (i,a) in enumerate(arrays)
        A[Block(i)] = a
        # setblock!(A, a, i)
    end
    A
end

function BlockArrays.BlockMatrix(arrays::AbstractMatrix{T}...) where {T}
    A = BlockArray{T}(undef_blocks, [length(array) for array in arrays])
    for (i,a) in enumerate(arrays)
        A[Block(i)] = a
        # setblock!(A, a, i)
    end
    A
end

"Return true if the set is indexable and has elements whose type is a subtype of T."
indexable_list(set, ::Type{T}) where {T} = typeof(set[1]) <: T

indexable_list(set::Array{S}, ::Type{T}) where {S,T} = S <: T
indexable_list(set::NTuple{N,S}, ::Type{T}) where {N,S,T}= S <: T


#An efficient way to access elements of a Tuple type using index j
@generated function tuple_index(::Type{T}, j) where {T <: Tuple}
    :($T.parameters[j])
end

@generated function tuple_length(::Type{T}) where {T <: Tuple}
    :($length(T.parameters))
end

function insert_at(a::Vector, idx, b)
    if idx > 1
        [a[1:idx-1]..., b..., a[idx:end]...]
    else
        [b..., a...]
    end
end

isdyadic(n::Int) = n == 1<<round(Int, log2(n))


# This is a candidate for a better implementation. How does one generate a
# unit vector in a tuple?
dimension_tuple(::Val{N}, dim::Int) where N = ntuple(k -> ((k==dim) ? 1 : 0), Val(N))
dimension_tuple(N::Int, dim::Int) = ntuple(k -> ((k==dim) ? 1 : 0), N)

"Obtain a dense matrix copy of `A` using matrix-vector products."
function matrix_by_mul(A)
    T = eltype(A)
    Z = zeros(T, size(A))
    matrix_by_mul!(Z,A)
    Z
end

function matrix_by_mul!(Z::Matrix, A)
    e = zeros(eltype(Z),size(Z,2))
    for i in 1:size(Z,2)
        e[i] = 1
        mul!(view(Z, :, i), A, e)
        e[i] = 0
    end
end


# TODO: move these definitions to DomainSets
###############
# Subeltype
###############

"Return the type of the elements of `x`."
subeltype(x) = subeltype(typeof(x))
subeltype(::Type{T}) where {T} = eltype(eltype(T))

###############
# Dimension
###############

DomainSets.euclideandimension(::Type{CartesianIndex{N}}) where {N} = N


const Domain1d{T <: Number} = Domain{T}
const Domain2d{T} = EuclideanDomain{2,T}
const Domain3d{T} = EuclideanDomain{3,T}
const Domain4d{T} = EuclideanDomain{4,T}
export Domain1d, Domain2d, Domain3d, Domain4d

iscompatible(map1::AbstractMap, map2::AbstractMap) = map1==map2
iscompatible(map1::AffineMap, map2::AffineMap) = (map1.A ≈ map2.A) && (map1.b ≈ map2.b)
iscompatible(domain1::Domain, domain2::Domain) = domain1 == domain2
