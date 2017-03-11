# jacobi.jl

# A basis of Jacobi polynomials on the interval [-1,1]
immutable JacobiBasis{S,T} <: OPS{T}
    n       ::  Int
    α       ::  S
    β       ::  S

    JacobiBasis(n, α = zero(T), β = zero(T)) = new(n, α, β)
end

name(b::JacobiBasis) = "Jacobi OPS"

JacobiBasis{T}(n, ::Type{T} = Float64) = JacobiBasis(n, 0, 0, T)

JacobiBasis{S <: Number,T}(n, α::S, β::S, ::Type{T} = S) = JacobiBasis{S,T}(n, α, β)


instantiate{T}(::Type{JacobiBasis}, n, ::Type{T}) = JacobiBasis(n, T)

promote_eltype{S,T,T2}(b::JacobiBasis{S,T}, ::Type{T2}) = JacobiBasis{S,promote_type(T,T2)}(b.n, b.α, b.β)

resize(b::JacobiBasis, n) = JacobiBasis(n, b.α, b.β, eltype(b))

left{T}(b::JacobiBasis{T}) = -T(1)
left(b::JacobiBasis, idx) = left(b)

right{T}(b::JacobiBasis{T}) = T(1)
right(b::JacobiBasis, idx) = right(b)

#grid{S,T}(b::JacobiBasis{T}) = JacobiGrid{T}(b.n, jacobi_α(b), jacobi_β(b))


jacobi_α(b::JacobiBasis) = b.α
jacobi_β(b::JacobiBasis) = b.β

weight(b::JacobiBasis, x) = (x-1)^b.α * (x+1)^b.β


# See DLMF (18.9.2)
# http://dlmf.nist.gov/18.9#i
rec_An{S,T}(b::JacobiBasis{S,T}, n::Int) = T(2*n + b.α + b.β + 1) * (2*n + b.α + b.β + 2) / T(2 * (n+1) * (n + b.α + b.β + 1))

rec_Bn{S,T}(b::JacobiBasis{S,T}, n::Int) =
    T(b.α^2 - b.β^2) * (2*n + b.α + b.β + 1) / T(2 * (n+1) * (n + b.α + b.β + 1) * (2*n + b.α + b.β))

rec_Cn{S,T}(b::JacobiBasis{S,T}, n::Int) =
    T(n + b.α) * (n + b.β) * (2*n + b.α + b.β + 2) / T((n+1) * (n + b.α + b.β + 1) * (2*n + b.α + b.β))
