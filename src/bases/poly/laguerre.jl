# laguerre.jl

"A Laguerre polynomial basis"
struct LaguerreBasis{T} <: OPS{T}
    n       ::  Int
    α       ::  T

    LaguerreBasis{T}(n::Int, α = zero(T)) where {T} = new(n, α)
end

const LaguerreSpan{A, F <: LaguerreBasis} = Span{A,F}

name(b::LaguerreBasis) = "Laguerre OPS"

LaguerreBasis(n, ::Type{T} = Float64) where {T} = LaguerreBasis{T}(n)

LaguerreBasis(n, α::T) where {T <: Integer} = LaguerreBasis(n, float(α))

LaguerreBasis(n, α::T) where {T <: Number} = LaguerreBasis{T}(n, α)


instantiate(::Type{LaguerreBasis}, n, ::Type{T}) where {T} = LaguerreBasis{T}(n)

set_promote_domaintype(b::LaguerreBasis, ::Type{S}) where {S} =
    LaguerreBasis{S}(b.n, b.α)

resize(b::LaguerreBasis, n) = LaguerreBasis(n, b.α)

left(b::LaguerreBasis{T}) where {T} = T(0)
left(b::LaguerreBasis, idx) = left(b)

right(b::LaguerreBasis) = convert(domaintype(b), Inf)
right(b::LaguerreBasis, idx) = right(b)

#grid(b::LaguerreBasis) = LaguerreGrid(b.n)

jacobi_α(b::LaguerreBasis) = b.α


weight(b::LaguerreBasis{T}, x) where {T} = exp(-T(x)) * T(x)^(b.α)

function gramdiagonal!(result, ::LaguerreSpan; options...)
    T = eltype(result)
    for i in 1:length(result)
        result[i] = gamma(T(i+jacobi_α(b)))/factorial(i-1)
    end
end



# See DLMF, Table 18.9.1
# http://dlmf.nist.gov/18.9#i
rec_An(b::LaguerreBasis{T}, n::Int) where {T} = -T(1) / T(n+1)

rec_Bn(b::LaguerreBasis{T}, n::Int) where {T} = T(2*n + b.α + 1) / T(n+1)

rec_Cn(b::LaguerreBasis{T}, n::Int) where {T} = T(n + b.α) / T(n+1)
