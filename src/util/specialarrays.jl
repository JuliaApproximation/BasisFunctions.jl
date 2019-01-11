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

isefficient(::HorizontalBandedMatrix) = true

Base.size(A::HorizontalBandedMatrix) = (A.m,A.n)
Base.size(A::HorizontalBandedMatrix, i::Int) = (i==1) ? A.m : A.n

Base.getindex(op::HorizontalBandedMatrix, i::Int, j::Int) =
    _get_horizontal_banded_index(op.array, op.step, op.offset, size(op,1), size(op,2), i, j)

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

isefficient(::VerticalBandedMatrix) = true

Base.size(A::VerticalBandedMatrix) = (A.m,A.n)
Base.size(A::VerticalBandedMatrix, i::Int) = (i==1) ? A.m : A.n

Base.getindex(op::VerticalBandedMatrix, i::Int, j::Int) =
    _get_vertical_banded_index(op.array, op.step, op.offset, size(op,1), size(op,2), i, j)

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
