
"""
An `AbstractArrayOperator` combines an `AbstractArray` with a source and destination
dictionary.
"""
abstract type AbstractArrayOperator{T} <: DictionaryOperator{T} end

verify_size(src, dest, A) = (size(A,2) == length(src)) && (size(A,1)== length(dest))

A_src(A::AbstractMatrix{T}) where {T} = DiscreteVectorDictionary{T}(size(A,1))
A_dest(A::AbstractMatrix{T}) where {T} = DiscreteVectorDictionary{T}(size(A,2))

getindex(op::AbstractArrayOperator, i::Int, j::Int) = op.A[i,j]

size(op::AbstractArrayOperator) = size(op.A)

unsafe_wrap_operator(src, dest, op::AbstractArrayOperator) = similar_operator(op, src, dest)

inv(op::AbstractArrayOperator) = AbstractArrayOperator(inv(op.A), dest(op), src(op))

adjoint(op::AbstractArrayOperator) = AbstractArrayOperator(adjoint(op.A), dest(op), src(op))

similar_operator(op::AbstractArrayOperator, src::Dictionary, dest::Dictionary) =
    AbstractArrayOperator(op.A, src, dest)

apply_inplace!(op::AbstractArrayOperator, coef_srcdest) = _apply_inplace!(op, op.A, coef_srcdest)
_apply_inplace!(op::AbstractArrayOperator, A, x) = mul!(x, A, x)

apply!(op::AbstractArrayOperator, coef_dest, coef_src) = _apply!(op, op.A, coef_dest, coef_src)
_apply!(op::AbstractArrayOperator, A, coef_dest, coef_src) = mul!(coef_dest, A, coef_src)

diagonal(op::AbstractArrayOperator) = diag(op.A)

matrix(op::AbstractArrayOperator) = op.A



struct DiagonalOperator{T} <: AbstractArrayOperator{T}
    src     ::  Dictionary
    dest    ::  Dictionary
    A       ::  Diagonal{T,Array{T,1}}

    function DiagonalOperator{T}(src::Dictionary, dest::Dictionary, A) where {T}
        @assert verify_size(src, dest, A)
        new(src, dest, A)
    end
end

# Convert various arguments to a diagonal matrix with a concrete diagonal vector
# with elements of type T.
to_diagonal(A::AbstractArray{T}) where {T} = to_diagonal(T, A)
to_diagonal(::Type{T}, A::AbstractVector) where {T} = to_diagonal(T, collect(A))
to_diagonal(::Type{T}, A::Vector{T}) where {T} = Diagonal(A)
to_diagonal(::Type{T}, A::Vector{S}) where {S,T} = to_diagonal(T, convert(Vector{T}, A))
to_diagonal(::Type{T}, A::Diagonal{T,Array{T,1}})  where {T} = A
to_diagonal(::Type{T}, A::Diagonal{T}) where {T} = to_diagonal(T, diag(A))
to_diagonal(::Type{T}, A::Diagonal{S}) where {S,T} = to_diagonal(T, convert(Diagonal{T}, A))

DiagonalOperator(A::AbstractArray, args...) = DiagonalOperator(to_diagonal(A), args...)

DiagonalOperator(A::Diagonal{T,Array{T,1}}, src = A_src(A), dest = src) where {T} =
    DiagonalOperator{promote_type(T,op_eltype(src,dest))}(A, src, dest)

DiagonalOperator{T}(A::AbstractArray, args...) where {T} = DiagonalOperator{T}(to_diagonal(T,A), args...)

DiagonalOperator{T}(A::Diagonal{T,Array{T,1}}, src = A_src(A), dest = src) where {T} =
    DiagonalOperator{T}(src, dest, A)

# For backward compatibility
DiagonalOperator(src::Dictionary, dest::Dictionary, A::AbstractArray) = DiagonalOperator(A, src, dest)
DiagonalOperator(src::Dictionary, A::AbstractArray) = DiagonalOperator(A, src)
DiagonalOperator{T}(src::Dictionary, A::AbstractArray) where {T} = DiagonalOperator{T}(A, src)

is_diagonal(op::DiagonalOperator) = true
is_inplace(op::DiagonalOperator) = true


_apply_inplace!(op::AbstractArrayOperator, A::Diagonal, x::AbstractVector) = mul!(x, A, x)
function _apply_inplace!(op::AbstractArrayOperator, A::Diagonal, x)
    @assert length(x) == size(A,1)
    for i in 1:length(x)
        x[i] *= A[i,i]
    end
    x
end


_apply!(op::AbstractArrayOperator, A::Diagonal, coef_dest::AbstractVector, coef_src::AbstractVector) = mul!(coef_dest, A, coef_src)
function _apply!(op::AbstractArrayOperator, A::Diagonal, coef_dest, coef_src)
    @assert length(coef_dest) == length(coef_src) == size(A,1)
    for i in 1:length(coef_dest)
        coef_dest[i] = A[i,i] * coef_src[i]
    end
    coef_dest
end


struct UniformScalingOperator{T} <: AbstractArrayOperator{T}
    src     ::  Dictionary
    dest    ::  Dictionary
    I       ::  UniformScaling{T}
    size    ::  Tuple{Int,Int}

    UniformScalingOperator{T}(src, dest, lambda) where {T} =
        new(src, dest, lambda, (length(dest),length(src)))
end

UniformScalingOperator(I::UniformScaling{T}, src::Dictionary, dest::Dictionary = src) where {T} =
    UniformScalingOperator{promote_type(T,op_eltype(src,dest))}(I, src, dest)

UniformScalingOperator{T}(I::UniformScaling)

is_diagonal(op::UniformScalingOperator) = true
is_inplace(op::UniformScalingOperator) = true

_apply_inplace!(op::AbstractArrayOperator, A::UniformScaling, x::AbstractVector) = mul!(x, A, x)
function _apply_inplace!(op::AbstractArrayOperator, A::UniformScaling, x)
    for i in 1:length(x)
        x[i] *= A.λ
    end
    x
end

_apply!(op::AbstractArrayOperator, A::UniformScaling, coef_dest::AbstractVector, coef_src::AbstractVector) = mul!(coef_dest, A, coef_src)
function _apply!(op::AbstractArrayOperator, A::UniformScaling, coef_dest, coef_src)
    @assert length(coef_dest) == length(coef_src)
    for i in 1:length(coef_dest)
        coef_dest[i] = A.λ * coef_src[i]
    end
    coef_dest
end



AbstractArrayOperator(A::AbstractArray) =
    AbstractArrayOperator(A, A_src(A), A_dest(A))
AbstractArrayOperator(A::AbstractArray, src::Dictionary) =
    AbstractArrayOperator(A, src, src)
AbstractArrayOperator(A::UniformScaling, src::Dictionary) =
    AbstractArrayOperator(A, src, src)


AbstractArrayOperator(A::Diagonal, src::Dictionary, dest::Dictionary) =
    DiagonalOperator(A, src, dest)
AbstractArrayOperator(A::UniformScaling, src::Dictionary, dest::Dictionary) =
    UniformScalingOperator(A, src, dest)
