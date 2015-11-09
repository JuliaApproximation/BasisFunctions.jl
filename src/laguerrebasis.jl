# laguerrebasis.jl

# A Laguerre polynomial basis
immutable LaguerreBasis{T <: AbstractFloat} <: OPS{T}
    n           ::  Int
    alpha       ::  T
end

LaguerreBasis(n) = LaguerreBasis(n, 0.0)


name(b::LaguerreBasis) = "Laguerre series"

isreal(b::LaguerreBasis) = True()
isreal{B <: LaguerreBasis}(::Type{B}) = True

left{T}(b::LaguerreBasis{T}) = zero(T)
left{T}(b::LaguerreBasis{T}, idx) = left(b)

right{T}(b::LaguerreBasis{T}) = inf(T)
right{T}(b::LaguerreBasis{T}, idx) = right(b)

grid{T}(b::LaguerreBasis{T}) = LaguerreGrid(b.n)

alpha(b::LaguerreBasis) = b.alpha


weight(b::LaguerreBasis, x) = exp(-x) * x^(b.alpha)


# See DLMF, Table 18.9.1
# http://dlmf.nist.gov/18.9#i
rec_An(b::LaguerreBasis, n::Int) = -1 / (n+1)

rec_Bn(b::LaguerreBasis, n::Int) = (2*n + b.alpha + 1) / (n+1)

rec_Cn(b::LaguerreBasis, n::Int) = (n + b.alpha) / (n+1)


