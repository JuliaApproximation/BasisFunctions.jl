
"""
An `ArrayOperator` combines an `AbstractArray` with a source and destination
dictionary.
"""
abstract type ArrayOperator{T} <: DictionaryOperator{T} end

verify_size(src, dest, A) = (size(A,2) == length(src)) && (size(A,1)== length(dest))

A_src(A::AbstractMatrix{T}) where {T} = DiscreteVectorDictionary{T}(size(A,1))
A_dest(A::AbstractMatrix{T}) where {T} = DiscreteVectorDictionary{T}(size(A,2))

getindex(op::ArrayOperator, i::Int, j::Int) = op.A[i,j]

size(op::ArrayOperator) = size(op.A)

unsafe_wrap_operator(src, dest, op::ArrayOperator) = similar_operator(op, src, dest)

inv(op::ArrayOperator) = ArrayOperator(inv(op.A), dest(op), src(op))
adjoint(op::ArrayOperator) = ArrayOperator(adjoint(op.A), dest(op), src(op))
conj(op::ArrayOperator) = ArrayOperator(conj(matrix(op)), src(op), dest(op))

similar_operator(op::ArrayOperator, src::Dictionary, dest::Dictionary) =
    ArrayOperator(op.A, src, dest)

apply_inplace!(op::ArrayOperator, coef_srcdest) = _apply_inplace!(op, op.A, coef_srcdest)
_apply_inplace!(op::ArrayOperator, A, x) = mul!(x, A, x)

apply!(op::ArrayOperator, coef_dest, coef_src) = _apply!(op, op.A, coef_dest, coef_src)
_apply!(op::ArrayOperator, A, coef_dest, coef_src) = mul!(coef_dest, A, coef_src)

diagonal(op::ArrayOperator) = diag(op.A)

matrix(op::ArrayOperator) = copy(op.A)


struct DiagonalOperator{T,D} <: ArrayOperator{T}
    src     ::  Dictionary
    dest    ::  Dictionary
    A       ::  Diagonal{T,D}

    function DiagonalOperator{T,D}(src::Dictionary, dest::Dictionary, A::Diagonal{T,D}) where {T,D}
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

DiagonalOperator(A::AbstractArray; kwargs...) = DiagonalOperator(to_diagonal(A); kwargs...)

DiagonalOperator(A::Diagonal{S,D}; src = A_src(A), dest = src, T=op_eltype(src,dest)) where {S,D} =
    DiagonalOperator{promote_type(S,T)}(A, src=src, dest=dest)

DiagonalOperator{T}(A::AbstractArray; kwargs...) where {T} = DiagonalOperator{T}(to_diagonal(T,A); kwargs...)

DiagonalOperator{T}(A::Diagonal{T,D}; src = A_src(A), dest = src) where {T,D} =
    DiagonalOperator{T,D}(src, dest, A)

# For backward compatibility
DiagonalOperator(src::Dictionary, dest::Dictionary, A::AbstractArray; options...) =
    DiagonalOperator(A; src=src, dest=dest, options...)
DiagonalOperator(src::Dictionary, A::AbstractArray; options...) = DiagonalOperator(A; src=src, options...)
DiagonalOperator{T}(src::Dictionary, A::AbstractArray) where {T} = DiagonalOperator{T}(A; src=src)
DiagonalOperator{T}(src::Dictionary, dest::Dictionary, A::AbstractArray) where {T} = DiagonalOperator{T}(A; src=src, dest=dest)

isdiagonal(op::DiagonalOperator) = true
isinplace(op::DiagonalOperator) = true

isefficient(op::DiagonalOperator) = true

_apply_inplace!(op::ArrayOperator, A::Diagonal, x::AbstractVector) = mul!(x, A, x)
function _apply_inplace!(op::ArrayOperator, A::Diagonal, x)
    @assert length(x) == size(A,1)
    for i in 1:length(x)
        x[i] *= A[i,i]
    end
    x
end


_apply!(op::ArrayOperator, A::Diagonal, coef_dest::AbstractVector, coef_src::AbstractVector) = mul!(coef_dest, A, coef_src)
function _apply!(op::ArrayOperator, A::Diagonal, coef_dest, coef_src)
    @assert length(coef_dest) == length(coef_src) == size(A,1)
    for i in 1:length(coef_dest)
        coef_dest[i] = A[i,i] * coef_src[i]
    end
    coef_dest
end


struct ScalingOperator{T} <: ArrayOperator{T}
    src     ::  Dictionary
    dest    ::  Dictionary
    A       ::  UniformScaling{T}
    size    ::  Tuple{Int,Int}

    ScalingOperator{T}(src::Dictionary, dest::Dictionary, A::UniformScaling) where {T} =
        new(src, dest, A, (length(dest),length(src)))
end

to_scaling(scalar::Number) = to_scaling(typeof(scalar), scalar)
to_scaling(A::UniformScaling{T}) where {T} = to_scaling(T, A)
to_scaling(::Type{T}, scalar) where {T} = UniformScaling{T}(scalar)
to_scaling(::Type{T}, A::UniformScaling{T}) where {T} = A
to_scaling(::Type{T}, A::UniformScaling{S}) where {S,T} = UniformScaling{T}(A.λ)

const AnyScaling = Union{Number,UniformScaling}

ScalingOperator(src::Dictionary, scalar::Number; dest = src, T=coefficienttype(src)) =
    ScalingOperator(src, to_scaling(scalar); dest=dest,T=T)

ScalingOperator(src::Dictionary, A::UniformScaling{S}; dest = src, T=op_eltype(src,dest)) where {S} =
    ScalingOperator{promote_type(S,T)}(src, A, dest=dest)

ScalingOperator{T}(src::Dictionary, scalar::AnyScaling; dest = src) where {T} =
    ScalingOperator{T}(src, dest, to_scaling(T, scalar))

# For backwards compatibility
ScalingOperator(src::Dictionary, dest::Dictionary, scalar::AnyScaling; T=op_eltype(src,dest)) =
    ScalingOperator(src, scalar; dest=dest, T=T)
ScalingOperator{T}(src::Dictionary, dest::Dictionary, scalar::Number) where {T} =
    ScalingOperator{T}(src, scalar; dest=dest)


scalar(op::ScalingOperator) = op.A.λ

size(op::ScalingOperator) = op.size

isdiagonal(op::ScalingOperator) = true
isinplace(op::ScalingOperator) = true
isefficient(op::ScalingOperator) = true

apply_inplace!(op::ScalingOperator, x) = _apply_inplace!(op, op.A.λ, x)
function _apply_inplace!(op::ScalingOperator, λ, x)
    for i in 1:length(x)
        x[i] *= λ
    end
    x
end

apply!(op::ScalingOperator, y, x) = _apply!(op, op.A.λ, y, x)
function _apply!(op::ScalingOperator, λ, y, x)
    @assert length(y) == length(x)
    for i in 1:length(y)
        y[i] = λ * x[i]
    end
    y
end

diagonal(op::ScalingOperator{T}) where {T} = fill!(zeros(T, size(op,1)), scalar(op))

matrix(op::ScalingOperator{T}) where {T} = Matrix{T}(op.A, size(op))

*(scalar::Number, op::DictionaryOperator) = ScalingOperator(dest(op), scalar) * op

function string(op::ScalingOperator)
    if scalar(op) == 1
        "Identity Operator"
    else
        "Scaling by $(scalar(op))"
    end
end

function symbol(op::ScalingOperator)
    if scalar(op) == 1
        "I"
    else
        "α"
    end
end


const IdentityOperator{T} = DiagonalOperator{T,Ones{T}}

IdentityOperator(src::Dictionary, dest::Dictionary = src; T=op_eltype(src,dest)) =
    IdentityOperator{T}(src, dest)

function IdentityOperator{T}(src::Dictionary, dest = src) where {T}
    diag = Ones{T}(length(src))
    DiagonalOperator{T,typeof(diag)}(src, dest, Diagonal(diag))
end

strings(op::IdentityOperator) = ("Identity Operator of size $(size(op)) with element type $(eltype(op))",strings(src(op)))


struct DenseMatrixOperator{T} <: ArrayOperator{T}
    src     ::  Dictionary
    dest    ::  Dictionary
    A       ::  Matrix{T}

    function DenseMatrixOperator{T}(src::Dictionary, dest::Dictionary, A::Matrix) where {T}
        @assert verify_size(src, dest, A)
        new(src, dest, A)
    end
end

DenseMatrixOperator(A::Matrix; src = A_src(A), dest = A_dest(A), T=op_eltype(src,dest)) =
    DenseMatrixOperator{promote_eltype(eltype(A),T)}(src, dest, A)

DenseMatrixOperator{T}(A::Matrix; src = A_src(A)) where {T} =
    DenseMatrixOperator{T}(A, src, A_dest(A))


ArrayOperator(A::AbstractArray) =
    ArrayOperator(A, A_src(A), A_dest(A))
ArrayOperator(A::AbstractArray, src::Dictionary) =
    ArrayOperator(A, src, src)
ArrayOperator(A::UniformScaling, src::Dictionary) =
    ArrayOperator(A, src, src)


ArrayOperator(A::Diagonal, src::Dictionary, dest::Dictionary; T=op_eltype(src,dest)) =
    DiagonalOperator(A; src=src, dest=dest, T=T)
ArrayOperator(A::UniformScaling, src::Dictionary, dest::Dictionary; T=op_eltype(src,dest)) =
    ScalingOperator(src, A; dest=dest, T=T)
ArrayOperator(A::Matrix, src::Dictionary, dest::Dictionary; T=op_eltype(src,dest)) =
    DenseMatrixOperator(A; src=src, dest=dest, T=T)
