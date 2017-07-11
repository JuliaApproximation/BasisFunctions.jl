# laguerre.jl

"A Laguerre polynomial basis"
struct LaguerreBasis{S,T} <: OPS{T}
    n       ::  Int
    α       ::  S
end

name(b::LaguerreBasis) = "Laguerre OPS"

LaguerreBasis(n, ::Type{T} = Float64) where {T} = LaguerreBasis(n, 0, T)

LaguerreBasis(n, α::S, ::Type{T} = S) where {S <: Number,T}= LaguerreBasis{S,T}(n, α)


instantiate(::Type{LaguerreBasis}, n, ::Type{T}) where {T} = LaguerreBasis(n, T)

set_promote_domaintype(b::LaguerreBasis{S,T}, ::Type{T2}) where {S,T,T2} =
  LaguerreBasis{S,T2}(b.n, b.α)

resize(b::LaguerreBasis, n) = LaguerreBasis(n, b.α, eltype(b))

left(b::LaguerreBasis{T}) where {T} = T(0)
left(b::LaguerreBasis, idx) = left(b)

right(b::LaguerreBasis) = convert(domaintype(b), Inf)
right(b::LaguerreBasis, idx) = right(b)

#grid(b::LaguerreBasis) = LaguerreGrid(b.n)

jacobi_α(b::LaguerreBasis) = b.α


weight(b::LaguerreBasis{S,T}, x) where {S,T} = exp(-T(x)) * T(x)^(b.α)

function gramdiagonal!(result, ::LaguerreBasis{S,T}; options...) where {S,T}
    for i in 1:length(result)
        result[i] = gamma(T(i+jacobi_α(b)))/factorial(i-1)
    end
end



# See DLMF, Table 18.9.1
# http://dlmf.nist.gov/18.9#i
rec_An(b::LaguerreBasis{S,T}, n::Int) where {S,T} = -T(1) / T(n+1)

rec_Bn(b::LaguerreBasis{S,T}, n::Int) where {S,T} = T(2*n + b.α + 1) / T(n+1)

rec_Cn(b::LaguerreBasis{S,T}, n::Int) where {S,T} = T(n + b.α) / T(n+1)
