# poly_jacobi.jl

# A basis of Jacobi polynomials on the interval [-1,1]
immutable JacobiBasis{T <: AbstractFloat} <: OPS{T}
    n       ::  Int
    α       ::  T
    β       ::  T

    JacobiBasis(n, α = zero(T), β = zero(T)) = new(n, α, β)
end

name(b::JacobiBasis) = "Jacobi OPS"

JacobiBasis{T}(n::Int, ::Type{T}) = JacobiBasis{T}(n)

JacobiBasis{T <: Number}(n::Int, α::T, β::T) = JacobiBasis{T}(n, α, β)

instantiate{T}(::Type{JacobiBasis}, n, ::Type{T}) = JacobiBasis{T}(n)

similar(b::JacobiBasis, T, n) = JacobiBasis{T}(n, b.α, b.β)

name(b::JacobiBasis) = "Jacobi OPS"

left(b::JacobiBasis) = -1
left(b::JacobiBasis, idx) = -1

right(b::JacobiBasis) = 1
right(b::JacobiBasis, idx) = 1

grid{T}(b::JacobiBasis{T}) = JacobiGrid(b.n, jacobi_α(b), jacobi_β(b))


jacobi_α{T}(b::JacobiBasis{T}) = b.α
jacobi_β{T}(b::JacobiBasis{T}) = b.β

weight(b::JacobiBasis, x) = (x-1)^b.α * (x+1)^b.β


# See DLMF (18.9.2)
# http://dlmf.nist.gov/18.9#i
rec_An(b::JacobiBasis, n::Int) = (2*n + b.α + b.β + 1) * (2*n + b.α + b.β + 2) / (2 * (n+1) * (n + b.α + b.β + 1))

rec_Bn(b::JacobiBasis, n::Int) = (b.α^2 - b.β^2) * (2*n + b.α + b.β + 1) / (2 * (n+1) * (n + b.α + b.β + 1) * (2*n + b.α + b.β))

rec_Cn(b::JacobiBasis, n::Int) = (n + b.α) * (n + b.β) * (2*n + b.α + b.β + 2) / ((n+1) * (n + b.α + b.β + 1) * (2*n + b.α + b.β))

