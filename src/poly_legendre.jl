# poly_legendre.jl

# A basis of Legendre polynomials on the interval [-1,1]
immutable LegendreBasis{T <: AbstractFloat} <: OPS{T}
    n           ::  Int

    LegendreBasis(n) = new(n)
end

# Constructor with a default numeric type
LegendreBasis(n::Int) = LegendreBasis{Float64}(n)

name(b::LegendreBasis) = "Legendre series"

isreal(b::LegendreBasis) = True()
isreal{B <: LegendreBasis}(::Type{B}) = True

left(b::LegendreBasis) = -1
left(b::LegendreBasis, idx) = -1

right(b::LegendreBasis) = 1
right(b::LegendreBasis, idx) = 1

grid(b::LegendreBasis) = LegendreGrid(b.n)


jacobi_alpha(b::LegendreBasis) = 0
jacobi_beta(b::LegendreBasis) = 0

weight{T}(b::LegendreBasis{T}, x) = ones(T,x)


# See DLMF, Table 18.9.1
# http://dlmf.nist.gov/18.9#i
rec_An{T}(b::LegendreBasis{T}, n::Int) = T(2*n+1)/T(n+1)

rec_Bn{T}(b::LegendreBasis{T}, n::Int) = zero(T)

rec_Cn{T}(b::LegendreBasis{T}, n::Int) = T(n)/T(n+1)




