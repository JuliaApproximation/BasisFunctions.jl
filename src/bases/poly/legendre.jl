# legendre.jl

"""
A basis of Legendre polynomials on the interval [-1,1].
"""
struct LegendreBasis{T} <: OPS{T}
    n           ::  Int
end

name(b::LegendreBasis) = "Legendre OPS"

# Constructor with a default numeric type
LegendreBasis(n::Int, ::Type{T} = Float64) where {T} = LegendreBasis{T}(n)

instantiate(::Type{LegendreBasis}, n, ::Type{T}) where {T} = LegendreBasis{T}(n)

set_promote_domaintype(b::LegendreBasis, ::Type{S}) where {S} = LegendreBasis{S}(b.n)

resize(b::LegendreBasis, n) = LegendreBasis(n, domaintype(b))


left(b::LegendreBasis{T}) where {T} = -T(1)
left(b::LegendreBasis, idx) = left(b)

right(b::LegendreBasis{T}) where {T} = T(1)
right(b::LegendreBasis, idx) = right(b)

#grid(b::LegendreBasis) = LegendreGrid(b.n)


jacobi_α(b::LegendreBasis{T}) where {T} = T(0)
jacobi_β(b::LegendreBasis{T}) where {T} = T(0)

weight(b::LegendreBasis{T}, x) where {T} = T(1)

function gramdiagonal!{T}(result, ::LegendreBasis{T}; options...)
    for i in 1:length(result)
        result[i] = T(2//(2(i-1)+1))
    end
end

# See DLMF, Table 18.9.1
# http://dlmf.nist.gov/18.9#i
rec_An(b::LegendreBasis{T}, n::Int) where {T} = T(2*n+1)/T(n+1)

rec_Bn(b::LegendreBasis{T}, n::Int) where {T} = zero(T)

rec_Cn(b::LegendreBasis{T}, n::Int) where {T} = T(n)/T(n+1)
