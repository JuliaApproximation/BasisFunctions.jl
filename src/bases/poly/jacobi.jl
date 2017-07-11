# jacobi.jl

# A basis of Jacobi polynomials on the interval [-1,1]
struct JacobiBasis{S,T} <: OPS{T}
    n       ::  Int
    α       ::  S
    β       ::  S

    JacobiBasis{S,T}(n, α = zero(T), β = zero(T)) where {S,T} = new(n, α, β)
end



name(b::JacobiBasis) = "Jacobi OPS"

JacobiBasis(n, ::Type{T} = Float64) where {T} = JacobiBasis(n, 0, 0, T)

JacobiBasis(n, α::S, β::S, ::Type{T} = S) where {S <: Number,T} = JacobiBasis{S,T}(n, α, β)


instantiate(::Type{JacobiBasis}, n, ::Type{T}) where {T} = JacobiBasis(n, T)

set_promote_domaintype(b::JacobiBasis{S,T}, ::Type{T2}) where {S,T,T2} =
    JacobiBasis{S,T2}(b.n, b.α, b.β)

resize(b::JacobiBasis, n) = JacobiBasis(n, b.α, b.β, eltype(b))

left(b::JacobiBasis{T}) where {T} = -T(1)
left(b::JacobiBasis, idx) = left(b)

right(b::JacobiBasis{T}) where {T} = T(1)
right(b::JacobiBasis, idx) = right(b)

#grid{S,T}(b::JacobiBasis{T}) = JacobiGrid{T}(b.n, jacobi_α(b), jacobi_β(b))


jacobi_α(b::JacobiBasis) = b.α
jacobi_β(b::JacobiBasis) = b.β

weight(b::JacobiBasis{T}, x) where {T} = (T(x)-1)^b.α * (T(x)+1)^b.β


# See DLMF (18.9.2)
# http://dlmf.nist.gov/18.9#i
rec_An(b::JacobiBasis{S,T}, n::Int) where {S,T} = T(2*n + b.α + b.β + 1) * (2*n + b.α + b.β + 2) / T(2 * (n+1) * (n + b.α + b.β + 1))

rec_Bn(b::JacobiBasis{S,T}, n::Int) where {S,T} =
    T(b.α^2 - b.β^2) * (2*n + b.α + b.β + 1) / T(2 * (n+1) * (n + b.α + b.β + 1) * (2*n + b.α + b.β))

rec_Cn(b::JacobiBasis{S,T}, n::Int) where {S,T} =
    T(n + b.α) * (n + b.β) * (2*n + b.α + b.β + 2) / T((n+1) * (n + b.α + b.β + 1) * (2*n + b.α + b.β))
