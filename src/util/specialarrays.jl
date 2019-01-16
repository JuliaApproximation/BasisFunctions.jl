# These are a set of vectors with special values, for use in conjunction
# with `Diagonal` in order to create special diagonal operators.

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


"A vector of 1's."
struct Ones{T} <: AbstractArray{T,1}
    n   ::  Int
end

Ones(n::Int) = Ones{Float64}(n)

size(A::Ones) = (A.n,)
getindex(A::Ones{T}, i::Int) where {T} = one(T)

inv(D::Diagonal{T,Ones{T}}) where {T} = D

adjoint(D::Diagonal{T,Ones{T}}) where {T} = D


"A vector of zeros."
struct Zeros{T,N} <: AbstractArray{T,N}
    n   ::  NTuple{N,Int}

    Zeros{T}(n::Int...) where T =
        new{T,length(n)}(n)
end

Zeros(n::Int...) = Zeros{Float64}(n...)

size(A::Zeros) = A.n
getindex(A::Zeros{T}, i::Int...) where {T} = zero(T)

adjoint(D::Diagonal{T,Zeros{T,1}}) where {T} = D
adjoint(Z::Zeros{T,2}) where T = Zeros{T}(size(Z,2), size(Z,1))

isefficient(::Zeros) = true

mul!(dest::AbstractVector, op::Zeros, src::AbstractVector) =
    fill!(dest, zero(eltype(op)))

isdiag(::Zeros) = true

diag(op::Zeros) = Zeros{eltype(op)}(min(op.n...))

Base.copy(op::Zeros) = op

"A vector of constant values."
struct ConstantVector{T} <: AbstractArray{T,1}
    n   ::  Int
    val ::  T
end

size(A::ConstantVector) = (A.n,)
getindex(A::ConstantVector, i::Int) = A.val

for op in (:inv, :adjoint)
    @eval $op(D::Diagonal{T,ConstantVector{T}}) where {T} = Diagonal(ConstantVector(size(D,1), $op(D.diag.val)))
end


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
        @assert length(array) <= n
        @assert step <= n # apply! only works if step is smaller then L
        new{T}(m, n, array, step, offset)
    end
end

VerticalBandedMatrix(m::Int,n::Int,array::Vector, step::Int=1, offset::Int=0) =
    VerticalBandedMatrix{eltype(array)}(m, n, array, step, offset)

Base.copy(A::VerticalBandedMatrix{T}) where T =
    VerticalBandedMatrix{T}(size(A,1), size(A,2), Base.copy(A.array), A.step, A.offset)

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
struct IndexMatrix{T,I,SKINNY} <: MyAbstractMatrix{T}
    m::Int
    n::Int
    subindices  ::  I

    function IndexMatrix{T,I}(m::Int, n::Int, subindices::AbstractArray{Int}) where {T,I}
        skinny = n<m
        if Base.IteratorSize(subindices) != Base.SizeUnknown()
            @assert (length(subindices) == m) || (length(subindices) == n)
        end
        m == n && (@warn "IndexMatrix contains all elements, consider identity or (De)LinearizationOperator instead.")
        new{T,I,skinny}(m, n, subindices)
    end
end

IndexMatrix{T}(m::NTuple{1,Int}, n::NTuple{N,Int}, subindices::AbstractArray{CartesianIndex{N}}) where {T,N} =
    IndexMatrix{T}(m[1], prod(n), LinearIndices(CartesianIndices(CartesianIndex(n)))[subindices])

IndexMatrix{T}(m::NTuple{N,Int}, n::NTuple{1,Int}, subindices::AbstractArray{CartesianIndex{N}}) where {T,N} =
    IndexMatrix{T}(prod(m), n[1], LinearIndices(CartesianIndices(CartesianIndex(m)))[subindices])

IndexMatrix{T}(m::NTuple{1,Int}, n::NTuple{1,Int}, subindices::AbstractArray{Int}) where {T,N} =
    IndexMatrix{T}(m[1], n[1], subindices)

IndexMatrix{T}(m::Int, n::Int, subindices) where {T} =
    IndexMatrix{T,typeof(subindices)}(m, n, subindices)

IndexMatrix(m, n, subindices; T=Int) =
    IndexMatrix{T}(m, n, subindices)

Base.copy(A::IndexMatrix{T}) where T =
    IndexMatrix{T}(A.m, A.n, Base.copy(subindices(A)))

Base.adjoint(A::IndexMatrix{T}) where T = IndexMatrix{T}(A.n, A.m, subindices(A))

subindices(A::IndexMatrix) = A.subindices

Base.size(A::IndexMatrix) = (A.m,A.n)

isefficient(::IndexMatrix) = true

function Base.getindex(A::IndexMatrix{T,I,false}, i::Int, j::Int) where {T,I}
    @boundscheck checkbounds(A,i,j)
    subindices(A)[i]==j ? convert(T,1) : convert(T,0)
end

function Base.getindex(A::IndexMatrix{T,I,true}, i::Int, j::Int) where {T,I}
    @boundscheck checkbounds(A,i,j)
    subindices(A)[j]==i ? convert(T,1) : convert(T,0)
end

mul!(dest, A::IndexMatrix, src) =
    mul!(dest, A, src, subindices(A))
mul!(dest::AbstractVector, A::IndexMatrix, src::AbstractVector) =
    mul!(dest, A, src, subindices(A))

function mul!(dest, A::IndexMatrix{T,I,false}, src, subindices) where {T,I}
    for (i,j) in enumerate(subindices)
        dest[i] = src[j]
    end
    dest
end

function mul!(dest, A::IndexMatrix{T,I,true}, src, subindices) where {T,I}
    fill!(dest, zero(T))
    for (i,j) in enumerate(subindices)
        dest[j] = src[i]
    end
    dest
end
