# These are a set of vectors with special values, for use in conjunction
# with `Diagonal` in order to create special diagonal operators.

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
struct Zeros{T} <: AbstractArray{T,1}
    n   ::  Int
end

Zeros(n::Int) = Zeros{Float64}(n)

size(A::Zeros) = (A.n,)
getindex(A::Zeros{T}, i::Int) where {T} = zero(T)

adjoint(D::Diagonal{T,Zeros{T}}) where {T} = D


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
struct HorizontalBandedMatrix{T} <: AbstractArray{T,2}
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

Base.Matrix(A::HorizontalBandedMatrix) = matrix_by_mul(A)

Base.print_array(io::IO,A::HorizontalBandedMatrix) = Base.print_array(io, Matrix(A))

"A 2D array every column contains equal elements.

The top column starts at index offset, the second column at step+offset."
struct VerticalBandedMatrix{T} <: AbstractArray{T,2}
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

Base.Matrix(A::VerticalBandedMatrix) = matrix_by_mul(A)

Base.print_array(io::IO,A::VerticalBandedMatrix) = Base.print_array(io, Matrix(A))

"""
An IndexMatrix selects/restricts a subset of coefficients based on their indices.
"""
struct IndexMatrix{T,I <: AbstractArray,SKINNY,L<:LinearIndices} <: AbstractArray{T,2}
    m       ::  NTuple{N,Int} where N
    n       ::  NTuple{N,Int} where N
    subindices  ::  I
    linear  :: L

    function IndexMatrix{T,I}(ms, ns, subindices) where {T,I}
        m = prod(ms)
        n = prod(ns)
        skinny = n<m
        linear = LinearIndices(CartesianIndices(CartesianIndex(skinny ? ms : ns)))
        if Base.IteratorSize(subindices) != Base.SizeUnknown()
            @assert (length(subindices) == m) || (length(subindices) == n)
        end
        @assert m != n
        new{T,I,skinny,typeof(linear)}(tuple(ms...), tuple(ns...), subindices, linear)
    end
end

IndexMatrix{T}(m, n, subindices) where {T} =
    IndexMatrix{T,typeof(subindices)}(m, n, subindices)

IndexMatrix(m, n, subindices; T=Int) =
    IndexMatrix{T}(m, n, subindices)

Base.copy(A::IndexMatrix{T}) where T =
    IndexMatrix{T}(A.m, A.n, Base.copy(subindices(A)))

Base.adjoint(A::IndexMatrix{T}) where T = IndexMatrix{T}(A.n, A.m, subindices(A))

subindices(A::IndexMatrix) = A.subindices

Base.size(A::IndexMatrix) = (prod(A.m),prod(A.n))

isefficient(::IndexMatrix) = true

function Base.getindex(A::IndexMatrix{T,I,false}, i::Int, j::Int) where {T,I}
    @boundscheck checkbounds(A,i,j)
    A.linear[subindices(A)[i]]==j ? convert(T,1) : convert(T,0)
end

function Base.getindex(A::IndexMatrix{T,I,true}, i::Int, j::Int) where {T,I}
    @boundscheck checkbounds(A,i,j)
    A.linear[subindices(A)[j]]==i ? convert(T,1) : convert(T,0)
end

# Overwrite (*) function of LinearAlgebra to handle both tensor matrices and multiple tensor vectors.
function (*)(A::IndexMatrix, B::AbstractMatrix)
    TS = LinearAlgebra.promote_op(LinearAlgebra.matprod, eltype(A), eltype(B))
    @show TS
    if size(A,2) == size(B,1) # B is a a list of (tensor)vectors
        mul!(similar(B, TS, (size(A,1), size(B,2))), A, B)
    elseif size(A,2) == length(B) # B ia a tensormatrix
        mul!(similar(B, TS, (A.m,)), A, B)
    else
        error("Sizes not compatible")
    end
end

function (*)(A::IndexMatrix{T,I,true}, B::AbstractVector) where {T,I}
    TS = LinearAlgebra.promote_op(LinearAlgebra.matprod, eltype(A), eltype(B))
    mul!(similar(B, TS, A.m), A, B)
end

mul!(dest, A::IndexMatrix, src) =
    mul!(dest, A, src, subindices(A), eltype(subindices(A)))
mul!(dest::AbstractVector, A::IndexMatrix, src::AbstractVector) =
    mul!(dest, A, src, subindices(A), eltype(subindices(A)))
mul!(dest::AbstractMatrix, A::IndexMatrix, src::AbstractVector) =
    mul!(dest, A, src, subindices(A), eltype(subindices(A)))
mul!(dest::AbstractVector, A::IndexMatrix, src::AbstractMatrix) =
    mul!(dest, A, src, subindices(A), eltype(subindices(A)))
function mul!(dest::AbstractMatrix, A::IndexMatrix{T,I,false}, src::AbstractMatrix)  where {T,I}
    @assert size(dest, 2) == size(src, 2)
    for i in size(src,2)
        mul!(view(dest,:,i), A, view(src,:,i))
    end
end

function mul!(dest, A::IndexMatrix{T,I,false}, src, subindices, ::Type{INT}) where {T,I,INT<:Integer}
    for (i,j) in enumerate(subindices)
        dest[i] = src[j]
    end
    dest
end

function mul!(dest, A::IndexMatrix{T,I,true}, src, subindices, ::Type{INT}) where {T,I,INT<:Integer}
    fill!(dest, convert(T, 0))
    for (i,j) in enumerate(subindices)
        dest[j] = src[i]
    end
    dest
end

function mul!(dest, A::IndexMatrix{T,I,false}, src::AbstractArray{T,N}, subindices, ::Type{CART}) where {T,N,I,CART<:CartesianIndex}
    if N > 1
        @assert length(src) == size(A,2)
        for (i,j) in enumerate(subindices)
            dest[i] = src[j]
        end
    else
        for (i,j) in enumerate(subindices)
            dest[i] = src[A.linear[j]]
        end
    end
    dest
end

function mul!(dest::AbstractArray{T,N}, A::IndexMatrix{T,I,true}, src, subindices, ::Type{CART}) where {T,N,I,CART<:CartesianIndex}
    fill!(dest, convert(T, 0))
    if N > 1
        @assert length(dest) == size(A,1)
        for (i,j) in enumerate(subindices)
            dest[j] = src[i]
        end
    else
        for (i,j) in enumerate(subindices)
            dest[A.linear[j]] = src[i]
        end
    end
    dest
end

Base.Matrix(A::IndexMatrix) = matrix_by_mul(A)

Base.print_array(io::IO,A::IndexMatrix) = Base.print_array(io, Matrix(A))

Base.show(io::IO,A::IndexMatrix) = Base.show(io, Matrix(A))
