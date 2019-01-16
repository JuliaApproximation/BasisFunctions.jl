
macro add_properties(T, props...)
    e = quote end
    for prop in props
        f = quote $(esc(prop))(a::$(esc(T))) = true end
        append!(e.args, f.args)
    end
    e
end

tolerance(::Type{T}) where {T} = sqrt(eps(T))
tolerance(::Type{Complex{T}}) where {T} = tolerance(T)

linspace(a,b,c) = range(a, stop=b, length=c)


# Convenience definitions for the implementation of traits
const True = Val{true}
const False = Val{false}

# delinearize_coefficients!(dest::BlockVector, src::AbstractVector) =
#     dest[:] .= src[:]
#
# linearize_coefficients!(dest::AbstractVector, src::BlockVector) =
#     dest[:] .= src[:]

delinearize_coefficients!(dest::AbstractArray{T,N}, src::AbstractVector{T}) where {T,N} =
    dest[:] .= src[:]

linearize_coefficients!(dest::AbstractVector{T}, src::AbstractArray{T,N}) where {T,N} =
    dest[:] .= src[:]

elements(bv::BlockVector) = [getblock(bv, i) for i in 1:nblocks(bv,1)]

function BlockArrays.BlockVector(arrays::AbstractVector{T}...) where {T}
    A = BlockArray{T}(undef_blocks, [length(array) for array in arrays])
    for (i,a) in enumerate(arrays)
        setblock!(A, a, i)
    end
    A
end

function BlockArrays.BlockMatrix(arrays::AbstractMatrix{T}...) where {T}
    A = BlockArray{T}(undef_blocks, [length(array) for array in arrays])
    for (i,a) in enumerate(arrays)
        setblock!(A, a, i)
    end
    A
end

mul!

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


default_threshold(y) = default_threshold(typeof(y))
default_threshold(::Type{T}) where {T <: AbstractFloat} = 100eps(T)
default_threshold(::Type{Complex{T}}) where {T <: AbstractFloat} = 100eps(T)
default_threshold(::AbstractArray{T}) where {T} = default_threshold(T)

# This is a candidate for a better implementation. How does one generate a
# unit vector in a tuple?
dimension_tuple(::Val{N}, dim::Int) where N = ntuple(k -> ((k==dim) ? 1 : 0), Val(N))

subeltype(x) = subeltype(eltype(x))
subeltype(::Type{T}) where {T <: Number} = T
subeltype(::Type{SVector{N,T}}) where {N,T} = T

function matrix_by_mul(A::AbstractMatrix{T}) where T
    Z = zeros(T, size(A))
    matrix_by_mul!(Z,A)
    Z
end

function matrix_by_mul!(Z::Matrix{T}, A::AbstractMatrix{T}) where T
    e = zeros(T,size(Z,2))
    for i in 1:size(Z,2)
        e[i] = 1
        mul!(view(Z, :, i), A, e)
        e[i] = 0
    end
end
