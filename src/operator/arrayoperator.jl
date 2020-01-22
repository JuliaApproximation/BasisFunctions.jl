
"""
An `ArrayOperator` combines an `AbstractArray` with a source and destination
dictionary.

From the 'AbstractArray' we expect that `mul!`, `size`, `copy`, `diag`, `isdiag`, `inv`, `getindex`,
`adjoint`, `conj` are implemented where possible.
"""
abstract type ArrayOperator{T} <: DictionaryOperator{T} end

verify_size(src, dest, A) =
    (length(src)==size(A,2)) && (length(dest)==size(A,1))

A_src(A::AbstractMatrix{T}) where {T} = DiscreteVectorDictionary{T}(size(A,2))
A_dest(A::AbstractMatrix{T}) where {T} = DiscreteVectorDictionary{T}(size(A,1))

getindex(op::ArrayOperator, i::Int, j::Int) = op.A[i,j]

# Delegate to object
for f in (:size, :isefficient)
    @eval $f(op::ArrayOperator) = $f(op.A)
end#object related features

for f in (:isdiag, :diag, :eigvals, :Matrix, :svdvals, :norm, :rank)
    @eval $f(op::ArrayOperator) = $f(unsafe_matrix(op))
end#matrix related features

for f in (:inv, :conj, :adjoint, :sqrt)
    @eval $f(op::ArrayOperator) = ArrayOperator($f(op.A), dest(op), src(op))
end#operator related features

unsafe_wrap_operator(src, dest, op::ArrayOperator) = similar_operator(op, src, dest)

similar_operator(op::ArrayOperator, src::Dictionary, dest::Dictionary) =
    ArrayOperator(op.A, src, dest)

apply_inplace!(op::ArrayOperator, coef_srcdest::AbstractVector) = _apply_inplace!(op, op.A, coef_srcdest)
_apply_inplace!(op::ArrayOperator, A::AbstractArray, x) = mul!(x, A, x)

apply!(op::ArrayOperator, coef_dest::AbstractVector, coef_src::AbstractVector) = _apply!(op, op.A, coef_dest, coef_src)
mul!(op::ArrayOperator, coef_dest::AbstractVector, coef_src::AbstractVector) = mul!(coef_dest, op.A, coef_src)

# Be forgiving for matrices: if the coefficients are multi-dimensional, reshape to a linear array first.
apply!(op::ArrayOperator{T}, coef_dest::AbstractArray{T,N1}, coef_src::AbstractArray{T,N2}) where {T,N1,N2} =
    apply!(op, reshape(coef_dest, length(coef_dest)), reshape(coef_src, length(coef_src)))

apply_inplace!(op::ArrayOperator{T}, coef_srcdest::AbstractArray{T,N}) where {T,N} =
    apply_inplace!(op, reshape(coef_srcdest, length(coef_srcdest)))

_apply!(op::ArrayOperator, A::AbstractArray, coef_dest, coef_src) = mul!(coef_dest, A, coef_src)


pinv(op::ArrayOperator, tolerance::Real = eps(real(eltype(op)))) = ArrayOperator(pinv(op.A, tolerance), src(op), dest(op))

matrix(op::ArrayOperator) = copy(op.A)
unsafe_matrix(op::ArrayOperator) = op.A

isefficient(::AbstractArray) = false
isefficient(::AbstractSparseArray) = true
isefficient(::UniformScaling) = true
isefficient(::ToeplitzMatrices.AbstractToeplitz) = true

string(op::ArrayOperator) = string(op, op.A)
string(op::ArrayOperator,array) = "Multiplication by "*string(typeof(op.A))

"""
A banded operator of which every row contains equal elements.

The top row starts at index offset, the second row at step+offset.
"""
struct HorizontalBandedOperator{T} <: ArrayOperator{T}
    A       ::  HorizontalBandedMatrix{T}
    src     ::  Dictionary
    dest    ::  Dictionary

end

HorizontalBandedOperator(src::Dictionary, dest::Dictionary, array::Vector{S}, step::Int=1, offset::Int=0; T=promote_type(S,op_eltype(src,dest))) where S =
    HorizontalBandedOperator{T}(HorizontalBandedMatrix(length(dest), length(src), T.(array), step, offset), src, dest)

ArrayOperator(A::HorizontalBandedMatrix{T}, src::Dictionary, dest::Dictionary) where T =
    HorizontalBandedOperator{T}(A, src, dest)


"""
A banded operator of which every column contains equal elements.

The top column starts at index offset, the second column at step+offset.
"""
struct VerticalBandedOperator{T} <: ArrayOperator{T}
    A       ::  VerticalBandedMatrix{T}
    src     ::  Dictionary
    dest    ::  Dictionary
end

VerticalBandedOperator(src::Dictionary, dest::Dictionary, array::Vector{S}, step::Int=1, offset::Int=0; T=promote_type(S,op_eltype(src,dest))) where S =
    VerticalBandedOperator{T}(VerticalBandedMatrix(length(dest), length(src), T.(array), step, offset), src, dest)

ArrayOperator(A::VerticalBandedMatrix{T}, src::Dictionary, dest::Dictionary) where T =
    VerticalBandedOperator{T}(A, src, dest)

"""
An IndexRestrictionOperator selects a subset of coefficients based on their indices.
"""
struct IndexRestrictionOperator{T,N,I} <: ArrayOperator{T}
    A           ::  RestrictionIndexMatrix{T,N,I}
    src         ::  Dictionary
    dest        ::  Dictionary

    IndexRestrictionOperator{T}(A::RestrictionIndexMatrix{T,N,I}, src::Dictionary, dest::Dictionary) where {T,N,I} =
        (@assert length(dest)<=length(src) && length(src)==size(A,2) && length(dest)==size(A,1); new{T,N,I}(A,src,dest))
end

IndexRestrictionOperator(src::Dictionary, subindices::AbstractVector; opts...) =
    IndexRestrictionOperator(src, src[subindices], subindices; opts...)

IndexRestrictionOperator(src::Dictionary, dest::Dictionary, subindices::AbstractVector; T = op_eltype(src, dest)) =
    (@assert length(subindices) == length(dest) && length(dest)<=length(src);
    IndexRestrictionOperator{T}(RestrictionIndexMatrix{T}(size(src), subindices), src, dest))
subindices(op::IndexRestrictionOperator) = subindices(op.A)

ArrayOperator(A::RestrictionIndexMatrix{T}, src::Dictionary, dest::Dictionary=src[subindices(A)]) where {T} =
    IndexRestrictionOperator{T}(A, src, dest)

hasstencil(op::IndexRestrictionOperator) = true
stencilarray(op::IndexRestrictionOperator) = [restrictionsymbol(op), "[", setsymbol(subindices(op.A)), " → 𝕀]"]

restrictionsymbol(op::IndexRestrictionOperator) = PrettyPrintSymbol{:R}()
name(::PrettyPrintSymbol{:R}) = "Restriction of coefficients to subset"


"""
An `IndexExtensionOperator` embeds coefficients in a larger set based on their indices.
"""
struct IndexExtensionOperator{T,N,I} <: ArrayOperator{T}
    A           ::  ExtensionIndexMatrix{T,N,I}
    src         ::  Dictionary
    dest        ::  Dictionary

    IndexExtensionOperator{T}(A::ExtensionIndexMatrix{T,N,I}, src::Dictionary, dest::Dictionary) where {T,N,I} =
        (@assert length(dest)>=length(src) && length(src)==size(A,2) && length(dest)==size(A,1); new{T,N,I}(A,src,dest))
end

IndexExtensionOperator(dest::Dictionary, subindices::AbstractVector; opts...) =
    IndexExtensionOperator(dest[subindices], dest, subindices; opts...)

IndexExtensionOperator(src::Dictionary, dest::Dictionary, subindices::AbstractVector; T = op_eltype(src, dest)) =
    (@assert length(src)==length(subindices) && length(dest)>=length(src);
    IndexExtensionOperator{T}(ExtensionIndexMatrix{T}(size(dest), subindices), src, dest))
subindices(op::IndexExtensionOperator) = subindices(op.A)

ArrayOperator(A::ExtensionIndexMatrix{T}, src::Dictionary, dest::Dictionary) where {T} =
    IndexExtensionOperator{T}(A, src, dest)

ArrayOperator(A::ExtensionIndexMatrix{T}, dest::Dictionary) where {T} =
    IndexRestrictionOperator{T}(A, dest[subindices(A)], dest)

string(op::IndexExtensionOperator) = "Zero padding, original elements in "*string(subindices(op.A))

hasstencil(op::IndexExtensionOperator) = true
stencilarray(op::IndexExtensionOperator) = [extensionsymbol(op), "[ 𝕀 → ", setsymbol(subindices(op.A)), "]"]

extensionsymbol(op::IndexExtensionOperator) = PrettyPrintSymbol{:E}()
name(::PrettyPrintSymbol{:E}) = "Extending coefficients by zero padding"


struct DiagonalOperator{T,D} <: ArrayOperator{T}
    src     ::  Dictionary
    dest    ::  Dictionary
    A       ::  Diagonal{T,D}

    function DiagonalOperator{T,D}(src::Dictionary, dest::Dictionary, A::Diagonal{T,D}) where {T,D}
        @assert verify_size(src, dest, A)
        new(src, dest, A)
    end
end

# Convert various arguments to a diag matrix with a concrete diag vector
# with elements of type T.
to_diag(A::AbstractArray{T}) where {T} = to_diag(T, A)
to_diag(::Type{T}, A::AbstractVector) where {T} = to_diag(T, collect(A))
to_diag(::Type{T}, A::Vector{T}) where {T} = Diagonal(A)
to_diag(::Type{T}, A::FillArrays.AbstractFill{T,1}) where {T} = Diagonal(A)
to_diag(::Type{T}, A::Vector{S}) where {S,T} = to_diag(T, convert(Vector{T}, A))
to_diag(::Type{T}, A::Diagonal{T,Array{T,1}})  where {T} = A
to_diag(::Type{T}, A::Diagonal{T}) where {T} = to_diag(T, diag(A))
to_diag(::Type{T}, A::Diagonal{S}) where {S,T} = to_diag(T, convert(Diagonal{T}, A))

DiagonalOperator(A::AbstractArray; kwargs...) = DiagonalOperator(to_diag(A); kwargs...)

DiagonalOperator(A::Diagonal{S,D}; src = A_src(A), dest = src, T=op_eltype(src,dest)) where {S,D} =
    DiagonalOperator{promote_type(S,T)}(A, src=src, dest=dest)

DiagonalOperator{T}(A::AbstractArray; kwargs...) where {T} = DiagonalOperator{T}(to_diag(T,A); kwargs...)

DiagonalOperator{T}(A::Diagonal{T,D}; src = A_src(A), dest = src) where {T,D} =
    DiagonalOperator{T,D}(src, dest, A)

# For backward compatibility
DiagonalOperator(src::Dictionary, dest::Dictionary, A::AbstractArray; options...) =
    DiagonalOperator(A; src=src, dest=dest, options...)
DiagonalOperator(src::Dictionary, A::AbstractArray; options...) = DiagonalOperator(A; src=src, options...)
DiagonalOperator{T}(src::Dictionary, A::AbstractArray) where {T} = DiagonalOperator{T}(A; src=src)
DiagonalOperator{T}(src::Dictionary, dest::Dictionary, A::AbstractArray) where {T} = DiagonalOperator{T}(A; src=src, dest=dest)
function DiagonalOperator(src::Dictionary, dest::Dictionary, A::OuterProductArray; options...)
    tensorproduct(map(DiagonalOperator, elements(src), elements(dest), elements(A))...)

end

isdiag(op::DiagonalOperator) = true
isinplace(op::DiagonalOperator) = true

isefficient(op::Diagonal) = true

_apply_inplace!(op::ArrayOperator, A::Diagonal, x) = mul!(x, A, x)

_apply!(op::ArrayOperator, A::Diagonal, coef_dest, coef_src) = mul!(coef_dest, A, coef_src)

symbol(op::DiagonalOperator) = "D"

symbol(op::DiagonalOperator{T,<:Ones}) where {T} = "I"


struct CirculantOperator{T} <: ArrayOperator{T}
    A       :: Circulant{T}
    src     :: Dictionary
    dest    :: Dictionary

    function CirculantOperator{T}(A::Circulant{T}, src::Dictionary, dest::Dictionary) where {T}
        verify_size(src, dest, A)
        new{T}(A, src, dest)
    end
end

CirculantOperator(vc::AbstractVector, src::Dictionary=DiscreteVectorDictionary{eltype(vc)}(length(vc)), dest::Dictionary=src;
            T=promote_type(eltype(vc),op_eltype(src,dest))) =
    CirculantOperator{T}(Circulant{T}(vc), src, dest)

ArrayOperator(A::Circulant{T}, src::Dictionary, dest::Dictionary) where {T} =
    CirculantOperator{T}(A, src, dest)

function CirculantOperator(op::DictionaryOperator)
    e = zeros(src(op))
    e[1] = 1
    C = CirculantOperator(op*e, src(op), dest(op), )
    e = rand(src(op))
    @assert C*e≈op*e
    C
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
@inline unsafe_matrix(A::ScalingOperator) = size(A,1) == size(A,2) ?
    Diagonal(Fill(scalar(A), size(A,1))) :
    FillArrays.RectDiagonal(Fill(scalar(A), min(size(A)...)), size(A)...)

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
isinplace(::ScalingOperator) = true
apply_inplace!(op::ScalingOperator, x::AbstractVector) = _apply_inplace!(op,scalar(op), x)
function _apply_inplace!(op::ScalingOperator, λ::Number, x)
    for i in 1:length(x)
        x[i] *= λ
    end
    x
end

apply!(op::ScalingOperator, y::AbstractVector, x::AbstractVector) = _apply!(op, scalar(op), y, x)
function _apply!(op::ScalingOperator, λ::Number, y, x)
    @assert length(y) == length(x)
    for i in 1:length(y)
        y[i] = λ * x[i]
    end
    y
end

diag(op::ScalingOperator{T}) where {T} = Fill{T}(scalar(op), size(op,1))
sqrt(op::ScalingOperator) = ScalingOperator(src(op), sqrt(scalar(op)); dest=dest(op))

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


const IdentityOperator{T} = DiagonalOperator{T,Ones{T,1,Tuple{Base.OneTo{Int64}}}} where T
conj(vc::Ones) = vc
broadcasted(::Base.Broadcast.DefaultArrayStyle{N}, ::Type{T}, a::Ones{S,N}) where {T<:Number,S,N} =
    Ones{T}(axes(a))
broadcasted(::Base.Broadcast.DefaultArrayStyle{N}, ::Type{T}, a::Zeros{S,N}) where {T<:Number,S,N} =
    Zeros{T}(axes(a))

IdentityOperator(src::Dictionary, dest::Dictionary = src; T=op_eltype(src,dest)) =
    IdentityOperator{T}(src, dest)

function IdentityOperator{T}(src::Dictionary, dest = src) where {T}
    diag = Ones{T}(length(src))
    DiagonalOperator{T,typeof(diag)}(src, dest, Diagonal(diag))
end

strings(op::IdentityOperator) = ("Identity Operator of size $(size(op)) with element type $(eltype(op))",strings(src(op)))
symbol(op::IdentityOperator) = "I"

struct DenseMatrixOperator{T} <: ArrayOperator{T}
    src     ::  Dictionary
    dest    ::  Dictionary
    A       ::  AbstractMatrix

    function DenseMatrixOperator{T}(src::Dictionary, dest::Dictionary, A::AbstractMatrix) where {T}
        @assert verify_size(src, dest, A)
        new{T}(src, dest, A)
    end
end

DenseMatrixOperator(A::AbstractMatrix; src = A_src(A), dest = A_dest(A), T=op_eltype(src,dest)) =
    DenseMatrixOperator{promote_eltype(eltype(A),T)}(src, dest, A)

DenseMatrixOperator{T}(A::AbstractMatrix; src = A_src(A)) where {T} =
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
ArrayOperator(A::AbstractMatrix, src::Dictionary, dest::Dictionary; T=op_eltype(src,dest)) =
    DenseMatrixOperator(A; src=src, dest=dest, T=T)

"The zero operator maps everything to zero."
struct ZeroOperator{T} <: ArrayOperator{T}
    A    ::     Zeros{T,2}
    src  ::     Dictionary
    dest ::     Dictionary

    ZeroOperator{T}(src::Dictionary, dest::Dictionary) where T =
        new(Zeros{T}(length(dest),length(src)), src, dest)
    function ZeroOperator{T}(Z::Zeros{T,2}, src::Dictionary, dest::Dictionary) where T
        @assert verify_size(src, dest, Z)
        new(Z, src, dest)
    end
end

ZeroOperator(src, dest=src; T = op_eltype(src, dest)) =
    ZeroOperator{T}(src, dest)

ArrayOperator(Z::Zeros{T,2}, src::Dictionary, dest::Dictionary) where T =
    ZeroOperator{T}(Z, src, dest)

isefficient(::Zeros) = true

mul!(dest::AbstractVector, op::Zeros, src::AbstractVector) =
    fill!(dest, zero(eltype(op)))
