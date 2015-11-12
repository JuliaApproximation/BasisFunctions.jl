# legendrebasis.jl

# A Legendre basis on the interval [a,b]
immutable LegendreBasis{T <: AbstractFloat} <: OPS{T}
    n           ::  Int

    LegendreBasis(n) = new(n)
end

# Constructor with a default numeric type
LegendreBasis(n::Int) = LegendreBasis{Float64}(n)

name(b::LegendreBasis) = "Legendre series"

isreal(b::LegendreBasis) = True()
isreal{B <: LegendreBasis}(::Type{B}) = True

left{T}(b::LegendreBasis{T}) = -one(T)
left{T}(b::LegendreBasis{T}, idx) = -one(T)

right{T}(b::LegendreBasis{T}) = one(T)
right{T}(b::LegendreBasis{T}, idx) = one(T)

grid{T}(b::LegendreBasis{T}) = LegendreGrid(b.n)


jacobi_alpha{T}(b::LegendreBasis{T}) = T(0)
jacobi_beta{T}(b::LegendreBasis{T}) = T(0)

weight(b::LegendreBasis, x) = ones(x)


# See DLMF, Table 18.9.1
# http://dlmf.nist.gov/18.9#i
rec_An{T}(b::LegendreBasis{T}, n::Int) = (2*n+1)/(n+1)

rec_Bn{T}(b::LegendreBasis{T}, n::Int) = zero(T)

rec_Cn{T}(b::LegendreBasis{T}, n::Int) = n/(n+1)




