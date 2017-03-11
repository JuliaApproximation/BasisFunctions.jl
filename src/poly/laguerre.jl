# laguerre.jl

"A Laguerre polynomial basis"
immutable LaguerreBasis{S,T} <: OPS{T}
    n       ::  Int
    α       ::  S
end

name(b::LaguerreBasis) = "Laguerre OPS"

LaguerreBasis{T}(n, ::Type{T} = Float64) = LaguerreBasis(n, 0, T)

LaguerreBasis{S <: Number,T}(n, α::S, ::Type{T} = S) = LaguerreBasis{S,T}(n, α)


instantiate{T}(::Type{LaguerreBasis}, n, ::Type{T}) = LaguerreBasis(n, T)

promote_eltype{S,T,T2}(b::LaguerreBasis{S,T}, ::Type{T2}) = LaguerreBasis{S,promote_type(T,T2)}(b.n, b.α)

resize(b::LaguerreBasis, n) = LaguerreBasis(n, b.α, eltype(b))

left{T}(b::LaguerreBasis{T}) = T(0)
left(b::LaguerreBasis, idx) = left(b)

right(b::LaguerreBasis) = convert(eltype(b), Inf)
right(b::LaguerreBasis, idx) = right(b)

#grid(b::LaguerreBasis) = LaguerreGrid(b.n)

jacobi_α(b::LaguerreBasis) = b.α


weight{S,T}(b::LaguerreBasis{S,T}, x) = exp(-x) * T(x)^(b.α)

function gramdiagonal!{S,T}(result, ::LaguerreBasis{S,T}; options...)
  for i in 1:length(result)
    result[i] = gamma(T(i+jacobi_α(b)))/factorial(i-1)
  end
end



# See DLMF, Table 18.9.1
# http://dlmf.nist.gov/18.9#i
rec_An{S,T}(b::LaguerreBasis{S,T}, n::Int) = -T(1) / T(n+1)

rec_Bn{S,T}(b::LaguerreBasis{S,T}, n::Int) = T(2*n + b.α + 1) / T(n+1)

rec_Cn{S,T}(b::LaguerreBasis{S,T}, n::Int) = T(n + b.α) / T(n+1)
