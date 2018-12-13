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
