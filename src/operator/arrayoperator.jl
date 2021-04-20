
"""
An `ArrayOperator` combines an `AbstractArray` with a source and destination
dictionary.

From the 'AbstractArray' we expect that `mul!`, `size`, `copy`, `diag`, `isdiag`, `inv`, `getindex`,
`adjoint`, `conj` are implemented where possible.
"""
abstract type ArrayOperator{T} <: DictionaryOperator{T} end

"Verify that size(A) == (length(dest),length(src)) ?"
verify_size(src, dest, A) = size(A) == (length(dest),length(src))

"Suggest a suitable src dictionary for a given matrix."
Asrc(A::AbstractMatrix{T}) where {T} = DiscreteVectorDictionary{T}(size(A,2))
"Suggest a suitable dest dictionary for a given matrix."
Adest(A::AbstractMatrix{T}) where {T} = DiscreteVectorDictionary{T}(size(A,1))

getindex(op::ArrayOperator, i::Int, j::Int) = op.A[i,j]

# Delegate to object
for f in (:size, :isefficient)
    @eval $f(op::ArrayOperator) = $f(op.A)
end#object related features

for f in (:isdiag, :diag, :eigvals, :Matrix, :svdvals, :norm, :rank, :sparse)
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
isefficient(::Diagonal) = true

string(op::ArrayOperator) = string(op, op.A)
string(op::ArrayOperator,array) = "Multiplication by "*string(typeof(op.A))



"A `MatrixOperator` is a wrapper around an abstract matrix type."
struct MatrixOperator{T,ARRAY} <: ArrayOperator{T}
    A       ::  ARRAY
    src     ::  Dictionary
    dest    ::  Dictionary

    function MatrixOperator{T,ARRAY}(A::ARRAY, src::Dictionary, dest::Dictionary) where {T,ARRAY}
        @assert verify_size(src, dest, A)
        new(A, src, dest)
    end
end

"Find a suitable element type for a dictionary operator with the given construct arguments."
deduce_eltype(A::AbstractArray{T}, args...) where {T} = T
deduce_eltype(A::AbstractArray, src::Dictionary, args...) = promote_type(eltype(A), operatoreltype(src))
deduce_eltype(A::AbstractArray, src::Dictionary, dest::Dictionary, args...) = promote_type(eltype(A), operatoreltype(src,dest))
deduce_eltype(src::Dictionary, args...) = operatoreltype(src)
deduce_eltype(src::Dictionary, A::AbstractArray, args...) = promote_type(eltype(A),operatoreltype(src))
deduce_eltype(src::Dictionary, z::Number, args...) = promote_type(typeoef(z),operatoreltype(src))
deduce_eltype(src::Dictionary, dest::Dictionary, args...) = operatoreltype(src, dest)
deduce_eltype(src::Dictionary, dest::Dictionary, A::AbstractArray, args...) = promote_type(eltype(A),operatoreltype(src, dest))
deduce_eltype(src::Dictionary, dest::Dictionary, z::Number, args...) = promote_type(typeoef(z),operatoreltype(src, dest))

MatrixOperator(args...) = MatrixOperator{deduce_eltype(args...)}(args...)

MatrixOperator{T}(A::AbstractArray) where {T} = MatrixOperator{T}(A, Asrc(A))
MatrixOperator{T}(A::AbstractArray, src::Dictionary) where {T} = MatrixOperator{T}(A, src, src)

MatrixOperator{T}(A::AbstractArray{T}, src::Dictionary, dest::Dictionary) where {T} =
    MatrixOperator{T,typeof(A)}(A, src, dest)
MatrixOperator{T}(A::AbstractArray{S}, src::Dictionary, dest::Dictionary) where {S,T} =
    MatrixOperator{T}(convert(AbstractArray{T}, A), src, dest)



## Banded operators

"""
A banded operator of which every row contains equal elements.

The top row starts at index offset, the second row at step+offset.
"""
const HorizontalBandedOperator{T} = MatrixOperator{T,HorizontalBandedMatrix{T}}

HorizontalBandedOperator(args...) = HorizontalBandedOperator{deduce_eltype(args...)}(args...)

HorizontalBandedOperator{T}(src::Dictionary, dest::Dictionary, array::AbstractVector, step::Int=1, offset::Int=0) where {T} =
    HorizontalBandedOperator{T}(HorizontalBandedMatrix(length(dest), length(src), T.(array), step, offset), src, dest)



"""
A banded operator of which every column contains equal elements.

The top column starts at index offset, the second column at step+offset.
"""
const VerticalBandedOperator{T} = MatrixOperator{T,VerticalBandedMatrix{T}}

VerticalBandedOperator(args...) = VerticalBandedOperator{deduce_eltype(args...)}(args...)

VerticalBandedOperator{T}(src::Dictionary, dest::Dictionary, array::AbstractVector, step::Int=1, offset::Int=0) where {T} =
    VerticalBandedOperator{T}(VerticalBandedMatrix(length(dest), length(src), T.(array), step, offset), src, dest)



## Index extension and restriction

"An IndexRestriction selects a subset of coefficients based on their indices."
const IndexRestriction{T,N,I} = MatrixOperator{T,RestrictionIndexMatrix{T,N,I}}

IndexRestriction(src::Dictionary, args...) = IndexRestriction{operatoreltype(src)}(src, args...)
IndexRestriction(src::Dictionary, dest::Dictionary, args...) = IndexRestriction{operatoreltype(src,dest)}(src, dest, args...)

IndexRestriction{T}(A::RestrictionIndexMatrix{T,N,I}, src::Dictionary, dest::Dictionary) where {T,N,I} =
    IndexRestriction{T,N,I}(A, src, dest)

IndexRestriction{T}(src::Dictionary, subindices::AbstractVector) where {T} =
    IndexRestriction{T}(src, src[subindices], subindices)

function IndexRestriction{T}(src::Dictionary, dest::Dictionary, subindices::AbstractVector) where {T}
    @assert length(dest)==length(subindices) && length(dest)<=length(src)
    IndexRestriction{T}(RestrictionIndexMatrix{T}(size(src), subindices), src, dest)
end

subindices(op::IndexRestriction) = subindices(op.A)

hasstencil(op::IndexRestriction) = true
stencilarray(op::IndexRestriction) = [restrictionsymbol(op), "[", setsymbol(subindices(op.A)), " → 𝕀]"]

restrictionsymbol(op::IndexRestriction) = PrettyPrintSymbol{:R}()
name(::PrettyPrintSymbol{:R}) = "Restriction of coefficients to subset"


"An `IndexExtension` embeds coefficients in a larger set based on their indices."
const IndexExtension{T,N,I} = MatrixOperator{T,ExtensionIndexMatrix{T,N,I}}

IndexExtension(src::Dictionary, args...) = IndexExtension{operatoreltype(src)}(src, args...)
IndexExtension(src::Dictionary, dest::Dictionary, args...) = IndexExtension{operatoreltype(src,dest)}(src, dest, args...)


IndexExtension{T}(A::ExtensionIndexMatrix{T,N,I}, src::Dictionary, dest::Dictionary) where {T,N,I} =
    IndexExtension{T,N,I}(A, src, dest)

IndexExtension{T}(dest::Dictionary, subindices::AbstractVector) where {T} =
    IndexExtension{T}(dest[subindices], dest, subindices)

function IndexExtension{T}(src::Dictionary, dest::Dictionary, subindices::AbstractVector) where {T}
    @assert length(src)==length(subindices) && length(dest)>=length(src)
    IndexExtension{T}(ExtensionIndexMatrix{T}(size(dest), subindices), src, dest)
end


subindices(op::IndexExtension) = subindices(op.A)

string(op::IndexExtension) = "Zero padding, original elements in "*string(subindices(op.A))

hasstencil(op::IndexExtension) = true
stencilarray(op::IndexExtension) = [extensionsymbol(op), "[ 𝕀 → ", setsymbol(subindices(op.A)), "]"]

extensionsymbol(op::IndexExtension) = PrettyPrintSymbol{:E}()
name(::PrettyPrintSymbol{:E}) = "Extending coefficients by zero padding"



# Diagonal operator and friends

"A diagonal operator."
const DiagonalOperator{T,D} = MatrixOperator{T,Diagonal{T,D}}

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

DiagonalOperator(args...) = DiagonalOperator{deduce_eltype(args...)}(args...)

DiagonalOperator{T}(A::AbstractMatrix) where {T} = DiagonalOperator{T}(A, Asrc(A), Adest(A))
DiagonalOperator{T}(A::AbstractMatrix, src::Dictionary) where {T} = DiagonalOperator{T}(A, src, src)

DiagonalOperator{T}(A::Diagonal{T,D}, src::Dictionary, dest::Dictionary) where {T,D} =
    DiagonalOperator{T,D}(A, src, dest)
DiagonalOperator{T}(A::AbstractArray, src::Dictionary, dest::Dictionary) where {T} =
    DiagonalOperator{T}(to_diag(T,A), src, dest)

# Support other order of arguments
DiagonalOperator{T}(src::Dictionary, A::AbstractArray) where {T} =
    DiagonalOperator{T}(A, src, src)
DiagonalOperator{T}(src::Dictionary, dest::Dictionary, A::AbstractArray) where {T} =
    DiagonalOperator{T}(A, src, dest)

DiagonalOperator(src::Dictionary, dest::Dictionary, A::AbstractOuterProductArray) =
    tensorproduct(map(DiagonalOperator, components(src), components(dest), components(A))...)


isinplace(op::DiagonalOperator) = true

_apply_inplace!(op::ArrayOperator, A::Diagonal, x) = mul!(x, A, x)
_apply!(op::ArrayOperator, A::Diagonal, coef_dest, coef_src) = mul!(coef_dest, A, coef_src)

strings(op::DiagonalOperator) = ("Diagonal operator with element type $(eltype(op))", strings(diag(op)))

symbol(op::DiagonalOperator) = "D"


"The identity operator"
const IdentityOperator{T} = DiagonalOperator{T,<:Ones{T}}

IdentityOperator(src::Dictionary) = IdentityOperator{operatoreltype(src)}(src)
IdentityOperator(src::Dictionary, dest::Dictionary) = IdentityOperator{operatoreltype(src,dest)}(src, dest)

function IdentityOperator{T}(src::Dictionary, dest = src) where {T}
    diag = Ones{T}(length(src))
    DiagonalOperator{T,typeof(diag)}(Diagonal(diag), src, dest)
end

isidentity(op::IdentityOperator) = true

broadcasted(::Base.Broadcast.DefaultArrayStyle{N}, ::Type{T}, a::Ones{S,N}) where {T<:Number,S,N} =
    Ones{T}(axes(a))
broadcasted(::Base.Broadcast.DefaultArrayStyle{N}, ::Type{T}, a::Zeros{S,N}) where {T<:Number,S,N} =
    Zeros{T}(axes(a))

strings(op::IdentityOperator) = ("Identity operator of size $(size(op)) (T=$(eltype(op)))", strings(src(op)))
symbol(op::IdentityOperator) = "I"


"Scaling by a scalar value"
const ScalingOperator{T} = DiagonalOperator{T,<:Fill{T}}

ScalingOperator(args...) = ScalingOperator{deduce_eltype(args...)}(args...)
ScalingOperator(z::Number, src::Dictionary, dest = src) =
    ScalingOperator{promote_type(typeof(z),operatoreltype(src,dest))}(z, src, dest)
ScalingOperator(z::UniformScaling{S}, src::Dictionary, dest = src) where {S} =
    ScalingOperator{promote_type(S,operatoreltype(src,dest))}(z, src, dest)

# Allow other order of arguments
ScalingOperator(src::Dictionary, a) = ScalingOperator(a, src)
ScalingOperator(src::Dictionary, dest::Dictionary, a) = ScalingOperator(a, src, dest)
ScalingOperator{T}(src::Dictionary, a) where {T} = ScalingOperator{T}(a, src)
ScalingOperator{T}(src::Dictionary, dest::Dictionary, a) where {T} = ScalingOperator{T}(a, src, dest)

function ScalingOperator{T}(z::Number, src::Dictionary, dest::Dictionary = src) where {T}
    diag = Fill(T(z), length(src))
    DiagonalOperator{T}(diag, src, dest)
end
function ScalingOperator{T}(z::UniformScaling, src::Dictionary, dest::Dictionary = src) where {T}
    diag = Fill(T(z.λ), length(src))
    DiagonalOperator{T}(diag, src, dest)
end

scalar(op::ScalingOperator) = op.A.diag.value
strings(op::ScalingOperator) = ("Scaling by $(scalar(op))",)
symbol(op::ScalingOperator) = "α"


"The zero operator maps everything to zero."
const ZeroOperator{T} = MatrixOperator{T,<:Zeros{T,2}}

ZeroOperator(src::Dictionary) = ZeroOperator{operatoreltype(src)}(src)
ZeroOperator(src::Dictionary, dest::Dictionary) = ZeroOperator{operatoreltype(src,dest)}(src,dest)

function ZeroOperator{T}(src::Dictionary, dest::Dictionary = src) where {T}
    A = Zeros{T}(length(dest), length(src))
    MatrixOperator{T}(A, src, dest)
end

strings(op::ZeroOperator) = ("Zero operator of size $(size(op)) (T=$(eltype(op)))", strings(src(op)))
symbol(op::ZeroOperator) = "0"




"A circulant operator."
const CirculantOperator{T,S} = MatrixOperator{T,Circulant{T,S}}

CirculantOperator(args...) = CirculantOperator{deduce_eltype(args...)}(args...)

to_circulant(v::AbstractVector) = to_circulant(eltype(v), v)
to_circulant(::Type{T}, v::AbstractVector{S}) where {S,T} = to_circulant(T, T.(v))
to_circulant(::Type{T}, v::AbstractVector{T}) where {T} = Circulant(v)
to_circulant(::Type{T}, A::Circulant{T}) where {T} = A

CirculantOperator{T}(v::AbstractVector) where {T} = CirculantOperator{T}(to_circulant(T, v))
CirculantOperator{T}(A::AbstractMatrix) where {T} = CirculantOperator{T}(A, Asrc(A), Adest(A))
CirculantOperator{T}(A::AbstractArray, src::Dictionary) where {T} =
    CirculantOperator{T}(A, src, src)
CirculantOperator{T}(A::AbstractArray, src::Dictionary, dest::Dictionary) where {T} =
    CirculantOperator{T}(to_circulant(T, A), src, dest)
CirculantOperator{T}(A::Circulant{T,S}, src::Dictionary, dest::Dictionary) where {T,S} =
    CirculantOperator{T,S}(A, src, dest)

# Temporary fix for ToeplitzMatrices issue #47
inv(C::CirculantOperator{T}) where {T} = CirculantOperator{T}(Circulant(ifft(1 ./ C.A.vcvr_dft)), dest(C), src(C))
inv(C::CirculantOperator{T}) where {T<:Real} = CirculantOperator{T}(Circulant(real(ifft(1 ./ C.A.vcvr_dft))), dest(C), src(C))

# Support other order of arguments
CirculantOperator{T}(src::Dictionary, A::AbstractArray) where {T} =
    CirculantOperator{T}(A, src)
CirculantOperator{T}(src::Dictionary, dest::Dictionary, A::AbstractArray) where {T} =
    CirculantOperator{T}(A, src, dest)

# Convert any dictionary operator to a circulant operator
function CirculantOperator(op::DictionaryOperator)
    e = zeros(src(op))
    e[1] = 1
    C = CirculantOperator(op*e, src(op), dest(op), )
    # Check accuracy of the conversion:
    e = rand(src(op))
    @assert C*e≈op*e
    C
end




"A dense matrix operator"
const DenseMatrixOperator{T} = MatrixOperator{T,Array{T,2}}

DenseMatrixOperator(args...) = DenseMatrixOperator{deduce_eltype(args...)}(args...)

DenseMatrixOperator{T}(A::AbstractMatrix) where {T} = DenseMatrixOperator{T}(A, Asrc(A), Adest(A))
DenseMatrixOperator{T}(A::AbstractMatrix, src::Dictionary) where {T} = DenseMatrixOperator{T}(A, src, src)


ArrayOperator(A::AbstractArray) =
    ArrayOperator(A, Asrc(A), Adest(A))
ArrayOperator(A::AbstractArray, src::Dictionary) =
    ArrayOperator(A, src, src)
ArrayOperator(A::UniformScaling, src::Dictionary) =
    ArrayOperator(A, src, src)

ArrayOperator(args...) = ArrayOperator{deduce_eltype(args...)}(args...)

ArrayOperator{T}(A::Matrix{T}, src::Dictionary, dest::Dictionary) where {T} =
    DenseMatrixOperator{T}(A, src, dest)
ArrayOperator{T}(A::Diagonal, src::Dictionary, dest::Dictionary) where {T} =
    DiagonalOperator{T}(A, src, dest)
ArrayOperator{T}(A::ExtensionIndexMatrix, src::Dictionary, dest::Dictionary) where {T} =
    IndexExtension{T}(A, src, dest)
ArrayOperator{T}(A::RestrictionIndexMatrix, src::Dictionary, dest::Dictionary) where {T} =
    IndexRestriction{T}(A, src, dest)
ArrayOperator{T}(A::HorizontalBandedMatrix{T}, src::Dictionary, dest::Dictionary) where {T} =
    HorizontalBandedOperator{T}(A, src, dest)
ArrayOperator{T}(A::VerticalBandedMatrix{T}, src::Dictionary, dest::Dictionary) where {T} =
    VerticalBandedOperator{T}(A, src, dest)
ArrayOperator{T}(A::Circulant, src::Dictionary, dest::Dictionary) where {T} =
    CirculantOperator{T}(A, src, dest)
ArrayOperator{T}(A::Zeros, src::Dictionary, dest::Dictionary) where {T} =
    ZeroOperator{T}(src, dest)

ArrayOperator{T}(A::AbstractMatrix, src::Dictionary, dest::Dictionary) where {T} =
    MatrixOperator{T}(A, src, dest)

export SparseMatrixOperator
"A sparse matrix operator"
const SparseMatrixOperator{T} = MatrixOperator{T,SparseMatrixCSC{T,Int}}
function SparseMatrixOperator(op; options...)
    A = sparse(op; options...)
    MatrixOperator(A, src(op), dest(op))
end

sparse(A::SparseMatrixOperator) = A.A
