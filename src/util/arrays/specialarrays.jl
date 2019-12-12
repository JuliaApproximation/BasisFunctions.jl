
"""
Shell for something that implements size, eltype and *
"""
struct ArrayShell{T,D} <: AbstractArray{T,D}
    A
    ArrayShell(A) = new{eltype(A),length(size(A))}(A)
end

Base.getindex(::ArrayShell) = error("AbstractArrayShell does not support getindex")

Base.size(A::ArrayShell) = size(A.A)

Base.eltype(A::ArrayShell) = eltype(A.A)

Base.print_array(io::IO,A::ArrayShell) = Base.print(io, A.A)

Base.show(io::IO,A::ArrayShell) = Base.show(io, A.A)

abstract type MyAbstractMatrix{T} <: AbstractArray{T,2} end

Base.Matrix(A::MyAbstractMatrix) = matrix_by_mul(A)

Base.print_array(io::IO,A::MyAbstractMatrix) = Base.print_array(io, Matrix(A))

Base.show(io::IO,A::MyAbstractMatrix) = Base.show(io, Matrix(A))

for (Tp,fun) in zip((:ProbabilityArray,), (:probabilityarray_fun,))
    @eval begin
        struct $Tp{T,N,Axes} <:FillArrays.AbstractFill{T,N,Axes}
            axes::Axes
            @inline $Tp{T, N}(sz::Axes) where Axes<:Tuple{Vararg{AbstractUnitRange,N}} where {T, N} =
                new{T,N,Axes}(sz)
            @inline $Tp{T,0,Tuple{}}(sz::Tuple{}) where T = new{T,0,Tuple{}}(sz)
        end


        @inline $Tp{T, 0}(sz::Tuple{}) where {T} = $Tp{T,0,Tuple{}}(sz)
        @inline $Tp{T, N}(sz::Tuple{Vararg{<:Integer, N}}) where {T, N} = $Tp{T,N}(Base.OneTo.(sz))
        @inline $Tp{T, N}(sz::Vararg{<:Integer, N}) where {T, N} = $Tp{T,N}(sz)
        @inline $Tp{T}(sz::Vararg{Integer,N}) where {T, N} = $Tp{T, N}(sz)
        @inline $Tp{T}(sz::SZ) where SZ<:Tuple{Vararg{Any,N}} where {T, N} = $Tp{T, N}(sz)
        @inline $Tp(sz::Vararg{Any,N}) where N = $Tp{Float64,N}(sz)
        @inline $Tp(sz::SZ) where SZ<:Tuple{Vararg{Any,N}} where N = $Tp{Float64,N}(sz)

        @inline $Tp{T,N}(A::AbstractArray{V,N}) where{T,V,N} = $Tp{T,N}(size(A))
        @inline $Tp{T}(A::AbstractArray) where{T} = $Tp{T}(size(A))
        @inline $Tp(A::AbstractArray) = $Tp(size(A))

        @inline axes(Z::$Tp) = Z.axes
        @inline size(Z::$Tp) = length.(Z.axes)
        @inline FillArrays.getindex_value(Z::$Tp) = $fun(Z)

        AbstractArray{T}(F::$Tp{T}) where T = F
        AbstractArray{T,N}(F::$Tp{T,N}) where {T,N} = F
        AbstractArray{T}(F::$Tp) where T = $Tp{T}(F.axes)
        AbstractArray{T,N}(F::$Tp{V,N}) where {T,V,N} = $Tp{T}(F.axes)
        convert(::Type{AbstractArray{T}}, F::$Tp{T}) where T = AbstractArray{T}(F)
        convert(::Type{AbstractArray{T,N}}, F::$Tp{T,N}) where {T,N} = AbstractArray{T,N}(F)
        convert(::Type{AbstractArray{T}}, F::$Tp) where T = AbstractArray{T}(F)
        convert(::Type{AbstractArray{T,N}}, F::$Tp) where {T,N} = AbstractArray{T,N}(F)

        getindex(F::$Tp{T,0}) where T = getindex_value(F)
        function getindex(F::$Tp{T}, kj::Vararg{AbstractVector{II},N}) where {T,II<:Integer,N}
            checkbounds(F, kj...)
            Fill{T}(FillArrays.getindex_value(F),length.(kj))
        end

        function getindex(A::$Tp{T}, kr::AbstractVector{Bool}) where T
            length(A) == length(kr) || throw(DimensionMismatch())
            Fill{T}(FillArrays.getindex_value(F),count(kr))
        end
        function getindex(A::$Tp{T}, kr::AbstractArray{Bool}) where T
            size(A) == size(kr) || throw(DimensionMismatch())
            Fill{T}(FillArrays.getindex_value(F),count(kr))
        end
    end
end

probabilityarray_fun(Z::ProbabilityArray{T}) where T = one(T) / convert(T,length(Z))


"A vector of the form `[1,-1,1,-1,...]`."
struct AlternatingSigns{T} <: AbstractArray{T,1}
    n   ::  Int
end

AlternatingSigns(n::Int) = AlternatingSigns{Float64}(n)

size(A::AlternatingSigns) = (A.n,)
getindex(A::AlternatingSigns{T}, i::Int) where {T} = iseven(i) ? -one(T) : one(T)

inv(D::Diagonal{T,AlternatingSigns{T}}) where {T} = D

adjoint(D::Diagonal{T,AlternatingSigns{T}}) where {T} = D


"A vector of the form `[1,1,...,1,V1,1,...1]`."
struct ScaledEntry{T} <: AbstractArray{T,1}
    n       ::  Int
    index   ::  Int
    scalar  ::  T
end

ScaledEntry(n::Int) = ScaledEntry{Float64}(n)

size(A::ScaledEntry) = (A.n,)
getindex(A::ScaledEntry{T}, i::Int) where {T} = i==A.index ? A.scalar : one(T)
conj(vc::ScaledEntry{T}) where T<:Real = vc
conj(vc::ScaledEntry{T}) where T = ScaledEntry{T}(vc.n, vc.index, conj(vc.scalar))

inv(D::Diagonal{T,ScaledEntry{T}}) where {T} = Diagonal(ScaledEntry{T}(D.diag.n, D.diag.index, inv(D.diag.scalar)))

adjoint(D::Diagonal{T,ScaledEntry{T}}) where {T} = Diagonal(ScaledEntry{T}(D.diag.n, D.diag.index, adjoint(D.diag.scalar)))



"A rectangular matrix"
struct ProjectionMatrix{I,T} <: AbstractArray{T,2}
    m       ::  Int
    n       ::  Int
    indices ::  I
end

size(A::ProjectionMatrix) = (A.m, A.n)






"A 2D array every row contains equal elements.

The top row starts at index offset, the second row at step+offset."
struct HorizontalBandedMatrix{T} <: MyAbstractMatrix{T}
    m       ::  Int
    n       ::  Int
    array   ::  Vector{T}
    step    ::  Int
    offset  ::  Int

    function HorizontalBandedMatrix{T}(m::Int,n::Int,array::Vector{T}, step::Int=1, offset::Int=0) where T
        @assert length(array) <= n
        @assert step <= n # apply! only works if step is smaller then L
        new{T}(m, n, array, step, offset)
    end
end

HorizontalBandedMatrix(m::Int,n::Int,array::Vector, step::Int=1, offset::Int=0) =
    HorizontalBandedMatrix{eltype(array)}(m, n, array, step, offset)

similar(A::HorizontalBandedMatrix{S}, ::Type{T}) where {S,T} =
    HorizontalBandedMatrix{T}(size(A)..., similar(A.array, T), A.step, A.offset)
# The method below is moved to SparseArrays for now
similar(A::HorizontalBandedMatrix, ::Type{T}, dims::Union{Dims{1},Dims{2}}) where {T} = spzeros(T, dims...)

Base.copy(A::HorizontalBandedMatrix{T}) where T =
    HorizontalBandedMatrix{T}(size(A,1), size(A,2), Base.copy(A.array), A.step, A.offset)

isefficient(::HorizontalBandedMatrix) = true

Base.size(A::HorizontalBandedMatrix) = (A.m,A.n)
Base.size(A::HorizontalBandedMatrix, i::Int) = (i==1) ? A.m : A.n

Base.getindex(op::HorizontalBandedMatrix, i::Int, j::Int) =
    _get_horizontal_banded_index(op.array, op.step, op.offset, size(op,1), size(op,2), i, j)

Base.inv(::HorizontalBandedMatrix) = error("Inverse not implemented.")

function _get_horizontal_banded_index(array::Vector{ELT}, step::Int, offset::Int, M::Int, N::Int, i::Int, j::Int) where {ELT}
    # first transform to an index of the first column
    index = mod(j-step*(i-1)-1-offset, N)+1
    if index <= length(array)
        array[index]
    else
        convert(ELT,0)
    end
end

Base.adjoint(M::HorizontalBandedMatrix) = VerticalBandedMatrix(size(M,2), size(M,1), conj(M.array), M.step, M.offset)

function mul!(dest::AbstractVector, op::HorizontalBandedMatrix, src::AbstractVector)
    dest[:] .= 0
    L = length(src)
    aL = length(op.array)
    dL = length(dest)
    @inbounds for a_i in 1:aL
        ind = mod(a_i+op.offset-1,L)+1
        for d_i in 1:dL
            dest[d_i] += op.array[a_i]*src[ind]
            ind += op.step
            if ind > L
                ind -= L
            end
        end
    end
    dest
end

"A 2D array every column contains equal elements.

The top column starts at index offset, the second column at step+offset."
struct VerticalBandedMatrix{T} <: MyAbstractMatrix{T}
    m       ::  Int
    n       ::  Int
    array   ::  Vector{T}
    step    ::  Int
    offset  ::  Int

    function VerticalBandedMatrix{T}(m::Int,n::Int,array::Vector{T}, step::Int=1, offset::Int=0) where T
        @assert length(array) <= m
        @assert step <= m # apply! only works if step is smaller then L
        new{T}(m, n, array, step, offset)
    end
end

VerticalBandedMatrix(m::Int,n::Int,array::Vector, step::Int=1, offset::Int=0) =
    VerticalBandedMatrix{eltype(array)}(m, n, array, step, offset)

Base.copy(A::VerticalBandedMatrix{T}) where T =
    VerticalBandedMatrix{T}(size(A,1), size(A,2), Base.copy(A.array), A.step, A.offset)
similar(A::VerticalBandedMatrix{S}, ::Type{T}) where {S,T} =
    VerticalBandedMatrix{T}(size(A)..., similar(A.array, T), A.step, A.offset)
# The method below is moved to SparseArrays for now
similar(A::VerticalBandedMatrix, ::Type{T}, dims::Union{Dims{1},Dims{2}}) where {T} = spzeros(T, dims...)

isefficient(::VerticalBandedMatrix) = true

Base.size(A::VerticalBandedMatrix) = (A.m,A.n)
Base.size(A::VerticalBandedMatrix, i::Int) = (i==1) ? A.m : A.n

Base.getindex(A::VerticalBandedMatrix, i::Int, j::Int) =
    _get_vertical_banded_index(A.array, A.step, A.offset, size(A,1), size(A,2), i, j)

Base.inv(::VerticalBandedMatrix) = error("Inverse not implemented.")

function _get_vertical_banded_index(array::Vector{ELT}, step::Int, offset::Int, M::Int, N::Int, i::Int, j::Int) where {ELT}
    # first transform to an index of the first column
    index = mod(i-step*(j-1)-1-offset, M)+1
    if index <= length(array)
        array[index]
    else
        ELT(0)
    end
end

Base.adjoint(M::VerticalBandedMatrix) = HorizontalBandedMatrix(size(M,2), size(M,1), conj(M.array), M.step, M.offset)

function mul!(dest::AbstractVector, op::VerticalBandedMatrix, src::AbstractVector)
    dest[:] .= 0
    # assumes step is smaller then L
    L = length(dest)
    aL = length(op.array)
    sL = length(src)
    @inbounds for a_i in 1:aL
        ind = mod(a_i+op.offset-1,L)+1
        for s_i in 1:sL
            dest[ind] += op.array[a_i]*src[s_i]
            ind += op.step
            if ind > L
                ind -= L
            end
        end
    end

    dest
end

"""
An IndexMatrix selects/restricts a subset of coefficients based on their indices.
"""
struct IndexMatrix{T,EXTENSION,N,I} <: MyAbstractMatrix{T}
    linear_size     ::Int
    original_size   ::NTuple{N,Int}
    subindices      ::I


    function IndexMatrix{T,EXTENSION,N,I}(original_size::NTuple{N,Int}, subindices::AbstractArray) where {N,T,I,EXTENSION}
        @assert (N==1) ? eltype(subindices) == Int : eltype(subindices) == CartesianIndex{N}
        n = length(subindices)
        m = prod(original_size)
        m == n && (@warn "IndexMatrix contains all elements, consider identity or (De)LinearizationOperator instead.")
        new{T,EXTENSION,N,I}(n, original_size, subindices)
    end
end

const ExtensionIndexMatrix{T,N,I} = IndexMatrix{T,true,N,I} where {T,N,I}
const RestrictionIndexMatrix{T,N,I} = IndexMatrix{T,false,N,I} where {T,N,I}

IndexMatrix{T,EXTENSION}(extended_size::NTuple{N,Int}, subindices::AbstractArray) where {T,EXTENSION,N} =
        IndexMatrix{T,EXTENSION,N,typeof(subindices)}(extended_size, subindices)

IndexMatrix(extended_size, subindices; T=Int, EXTENSION, options...) =
    IndexMatrix{T,EXTENSION}(extended_size, subindices)

_original_size(A::IndexMatrix) = A.original_size
_linear_size(A::IndexMatrix) = A.linear_size
subindices(A::IndexMatrix) = A.subindices
isextensionmatrix(A::IndexMatrix{T,EXTENSION,N,I}) where {T,EXTENSION,N,I} =
    EXTENSION

Base.copy(A::IndexMatrix{T,EXTENSION,N,I}) where {T,EXTENSION,N,I} =
    IndexMatrix{T,EXTENSION}(_original_size(A), Base.copy(subindices(A)))

Base.adjoint(A::IndexMatrix{T,EXTENSION,N,I}) where {T,EXTENSION,N,I} =
    IndexMatrix{T,!(EXTENSION),N,I}(_original_size(A), Base.copy(subindices(A)))

Base.size(A::IndexMatrix) = isextensionmatrix(A) ?
    (prod(_original_size(A)),_linear_size(A)) :
    (_linear_size(A),prod(_original_size(A)))

similar(A::IndexMatrix{S,EXTENSION}, ::Type{T}) where {S,T,EXTENSION} = IndexMatrix{T,EXTENSION}(_original_size(A), subindices(A))
# The method below is moved to SparseArrays for now
similar(A::IndexMatrix, ::Type{T}, dims::Union{Dims{1},Dims{2}}) where {T} = spzeros(T, dims...)

copyto!(D1::IndexMatrix, D2::AbstractArray) = error("Not possible")

isefficient(::IndexMatrix) = true

Base.getindex(A::ExtensionIndexMatrix{T,1}, i::Int, j::Int) where {T} =
    (@boundscheck checkbounds(A,i,j);
    Base.unsafe_getindex(A, i, j))

Base.getindex(A::ExtensionIndexMatrix{T,N}, i::Int, j::Int) where {T,N} =
    Base.getindex(A, CartesianIndices(CartesianIndex(_original_size(A)))[i], j)

function Base.getindex(A::ExtensionIndexMatrix{T,N}, i::CartesianIndex{N}, j::Int) where {T,N}
    @boundscheck checkbounds(subindices(A), j)
    @boundscheck i∈CartesianIndices(CartesianIndex(_original_size(A))) || throw(BoundsError())
    Base.unsafe_getindex(A, i, j)
end

Base.unsafe_getindex(A::ExtensionIndexMatrix{T,1}, i::Int, j::Int) where {T} =
    Base.unsafe_getindex(subindices(A),j) == i ? one(T) :  zero(T)

Base.unsafe_getindex(A::ExtensionIndexMatrix{T,N}, i::Int, j::Int) where {T,N} =
    Base.unsafe_getindex(A, CartesianIndices(CartesianIndex(_original_size(A)))[i], j)

Base.unsafe_getindex(A::ExtensionIndexMatrix{T,N}, i::CartesianIndex{N}, j::Int) where {T,N} =
    Base.unsafe_getindex(subindices(A),j)==i ? one(T) :  zero(T)

Base.getindex(A::RestrictionIndexMatrix{T,1}, i::Int, j::Int) where {T} =
    (@boundscheck checkbounds(A,i,j);
    Base.unsafe_getindex(A, i, j))

Base.getindex(A::RestrictionIndexMatrix{T,N}, i::Int, j::Int) where {T,N} =
    Base.getindex(A, i, CartesianIndices(CartesianIndex(_original_size(A)))[j])

function Base.getindex(A::RestrictionIndexMatrix{T,N}, i::Int, j::CartesianIndex{N}) where {T,N}
    @boundscheck checkbounds(subindices(A), i)
    @boundscheck j∈CartesianIndices(CartesianIndex(_original_size(A))) || throw(BoundsError())
    Base.unsafe_getindex(A, i, j)
end

Base.unsafe_getindex(A::RestrictionIndexMatrix{T,1}, i::Int, j::Int) where {T} =
    Base.unsafe_getindex(subindices(A),i)==j ? one(T) :  zero(T)

Base.unsafe_getindex(A::RestrictionIndexMatrix{T,N}, i::Int, j::Int) where {T,N} =
    Base.unsafe_getindex(A, i, CartesianIndices(CartesianIndex(_original_size(A)))[j])

Base.unsafe_getindex(A::RestrictionIndexMatrix{T,N}, i::Int, j::CartesianIndex{N}) where {T,I,N} =
    Base.unsafe_getindex(subindices(A),i)==j  ? one(T) :  zero(T)

Base.eltype(::IndexMatrix{T}) where T = T

function (*)(A::IndexMatrix, B::AbstractMatrix)
    TS = Base.promote_op(LinearAlgebra.matprod, eltype(A), eltype(B))
    C = similar(B, TS, (size(A,1),size(B,2)))

    if isextensionmatrix(A)
        fill!(C, zero(TS))
        for (i,j) in enumerate(LinearIndices(CartesianIndices(_original_size(A)))[subindices(A)])
            copyto!(C, CartesianIndices((j:j,1:size(C,2))), B, CartesianIndices(i:i,1:size(C,2)))
        end
    else
        for (i,j) in enumerate(LinearIndices(CartesianIndices(_original_size(A)))[subindices(A)])
            copyto!(C, CartesianIndices((i:i,1:size(C,2))), B, CartesianIndices((j:j,1:size(C,2))))
        end
    end
    C
end

(*)(A::IndexMatrix, B::Diagonal) = _sparse_res_mul(A, B)
(*)(A::IndexMatrix, B::VerticalBandedMatrix) = _sparse_res_mul(A, B)
(*)(A::IndexMatrix, B::HorizontalBandedMatrix) = _sparse_res_mul(A, B)

function _sparse_res_mul(A::IndexMatrix, B::Diagonal)
    @boundscheck (size(A,2)==size(B,1)) || throw(BoundsError("Sizes do not match"))
    TS = Base.promote_op(LinearAlgebra.matprod, eltype(A), eltype(B))
    I = LinearIndices(CartesianIndices(CartesianIndex(_original_size(A))))[subindices(A)]
    if isextensionmatrix(A)
        sparse(I,1:length(I),TS.(parent(B)[I]),size(A,1),size(B,2))
    else
        sparse(1:length(I),I,TS.(parent(B)[I]),size(A,1),size(B,2))
    end
end

function _sparse_res_mul(A::IndexMatrix, B::AbstractArray)
    TS = Base.promote_op(LinearAlgebra.matprod, eltype(A), eltype(B))
    C = similar(B, TS, (size(A,1),size(B,2)))
    @assert C isa AbstractSparseArray
    @warn "Creation of sparse array is not yet efficient."
    mul!(C,A,B)
end

mul!(dest, A::IndexMatrix, src) =
    mul!(dest, A, src, subindices(A))
mul!(dest::AbstractVector, A::ExtensionIndexMatrix, src::AbstractVector) =
    _tensor_mul!(reshape(dest, _original_size(A)), A, src, subindices(A))
mul!(dest::AbstractVector, A::RestrictionIndexMatrix, src::AbstractVector) =
    _tensor_mul!(dest, A, reshape(src,_original_size(A)), subindices(A))
function mul!(dest::AbstractVector, A::RestrictionIndexMatrix{S,N}, src::AbstractArray{T,N}) where {S,T,N}
    @boundscheck ( (size(src) == _original_size(A)) && (length(dest)==size(A,1)))|| throw(BoundsError("Sizes do not match"))
    @inbounds _tensor_mul!(dest, A, src, subindices(A))
end
function mul!(dest::AbstractMatrix, A::RestrictionIndexMatrix, src::AbstractMatrix)
    @boundscheck ( (size(src,1) == size(A,2)) && (size(src,2)==size(dest,2)) && (size(dest,1)==size(A,1)) ) || throw(BoundsError("Sizes do not match"))
    @inbounds for i in 1:size(src,2)
        mul!( view(dest, :, i), A, view(src, :, i)  )
    end
    dest
end



function _tensor_mul!(dest, A::RestrictionIndexMatrix, src, subindices)
    for (i,j) in enumerate(subindices)
        dest[i] = src[j]
    end
    dest
end

function _tensor_mul!(dest, A::ExtensionIndexMatrix, src, subindices)
    fill!(dest, zero(eltype(A)))
    for (i,j) in enumerate(subindices)
        dest[j] = src[i]
    end
    dest
end

# function Base.inv(C::Circulant{T}) where T<:Real
#     vdft = 1 ./ C.vcvr_dft
#     cvdft = copy(vdft)
#     return Circulant(real(C.dft \ vdft), cvdft, similar(vdft), C.dft)
# end
# function Base.inv(C::Circulant)C
#     vdft = 1 ./ C.vcvr_dft
#     cvdft = copy(vdft)
#     return Circulant(C.dft \ vdft, cvdft, similar(vdft), C.dft)
# end

function mul!(y::StridedVector{T}, A::Circulant{T}, x::StridedVector, α::T, β::T) where T
    fill!(A.tmp, 0)
    copyto!(A.tmp, in)
    mul!(A.tmp, A.dft, A.tmp)
    ldiv!(A.tmp, A.vcvr_dft)
    A.dft \ A.tmp
end

function mul!(out::StridedVector{T}, A::Circulant{T}, in::StridedVector{T}, _::T, _::T) where {T<:Real}
    fill!(A.tmp, 0)
    copyto!(A.tmp, in)
    mul!(A.tmp, A.dft, A.tmp)
    for i in 1:length(A.tmp)
        A.tmp[i] *= A.vcvr_dft[i]
    end
    A.dft \ A.tmp
    A.tmp .= real.(A.tmp)
    copyto!(out, A.tmp)
end

Base.adjoint(C::Circulant{T}) where T<:Real  =
    Circulant{T}(real(ifft(conj.(C.vcvr_dft))))
Base.adjoint(C::Circulant{T}) where T  =
    Circulant{T}(ifft(conj.(C.vcvr_dft)))
Base.conj(C::Circulant{T}) where {T<:Complex} =
    Circulant{T}(conj(ifft(C.vcvr_dft)))

function (*)(C1::Circulant{T1}, C2::Circulant{T2}) where {T1<:Real, T2<:Real}
    @boundscheck (size(C1)==size(C2)) || throw(BoundsError())
    Circulant(real.(ifft(C1.vcvr_dft.*C2.vcvr_dft)))
end
