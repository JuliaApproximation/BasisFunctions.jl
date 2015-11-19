# poly_laguerre.jl

"A Laguerre polynomial basis"
immutable LaguerreBasis{T <: AbstractFloat} <: OPS{T}
    n       ::  Int
    α       ::  T

    LaguerreBasis(n, α = zero(T)) = new(n, α)
end

name(b::LaguerreBasis) = "Laguerre OPS"

LaguerreBasis{T}(n, ::Type{T} = Float64) = LaguerreBasis{T}(n)

LaguerreBasis{T <: AbstractFloat}(n, α::T) = LaguerreBasis{T}(n, α)

LaguerreBasis{T <: AbstractFloat}(n, α::T) = LaguerreBasis{T}(n, α)

instantiate{T}(::Type{LaguerreBasis}, n, ::Type{T}) = LaguerreBasis{T}(n)


name(b::LaguerreBasis) = "Laguerre series"

isreal(b::LaguerreBasis) = True()
isreal{B <: LaguerreBasis}(::Type{B}) = True

left(b::LaguerreBasis) = 0
left(b::LaguerreBasis, idx) = left(b)

right{T}(b::LaguerreBasis{T}) = convert(T, Inf)
right{T}(b::LaguerreBasis{T}, idx) = right(b)

#grid(b::LaguerreBasis) = LaguerreGrid(b.n)

jacobi_α(b::LaguerreBasis) = b.α


weight{T}(b::LaguerreBasis{T}, x) = exp(-T(x)) * T(x)^(b.α)


# See DLMF, Table 18.9.1
# http://dlmf.nist.gov/18.9#i
rec_An{T}(b::LaguerreBasis{T}, n::Int) = -T(1) / T(n+1)

rec_Bn(b::LaguerreBasis, n::Int) = (2*n + b.α + 1) / (n+1)

rec_Cn(b::LaguerreBasis, n::Int) = (n + b.α) / (n+1)


