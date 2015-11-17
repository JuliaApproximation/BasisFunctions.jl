# poly_jacobi.jl

# A basis of Jacobi polynomials on the interval [-1,1]
immutable JacobiBasis{T <: AbstractFloat} <: OPS{T}
    n           ::  Int
    alpha       ::  T
    beta        ::  T
end

JacobiBasis(n::Int) = JacobiBasis(n, 0.0, 0.0)

name(b::JacobiBasis) = "Jacobi OPS"

isreal(b::JacobiBasis) = True()
isreal{B <: JacobiBasis}(::Type{B}) = True

left(b::JacobiBasis) = -1
left(b::JacobiBasis, idx) = -1

right(b::JacobiBasis) = 1
right(b::JacobiBasis, idx) = 1

grid{T}(b::JacobiBasis{T}) = JacobiGrid(b.n, jacobi_alpha(b), jacobi_beta(b))


jacobi_alpha{T}(b::JacobiBasis{T}) = b.alpha
jacobi_beta{T}(b::JacobiBasis{T}) = b.beta

weight(b::JacobiBasis, x) = (x-1)^b.alpha * (x+1)^b.beta


# See DLMF (18.9.2)
# http://dlmf.nist.gov/18.9#i
rec_An(b::JacobiBasis, n::Int) = (2*n + b.alpha + b.beta + 1) * (2*n + b.alpha + b.beta + 2) / (2 * (n+1) * (n + b.alpha + b.beta + 1))

rec_Bn(b::JacobiBasis, n::Int) = (b.alpha^2 - b.beta^2) * (2*n + b.alpha + b.beta + 1) / (2 * (n+1) * (n + b.alpha + b.beta + 1) * (2*n + b.alpha + b.beta))

rec_Cn(b::JacobiBasis, n::Int) = (n + b.alpha) * (n + b.beta) * (2*n + b.alpha + b.beta + 2) / ((n+1) * (n + b.alpha + b.beta + 1) * (2*n + b.alpha + b.beta))
