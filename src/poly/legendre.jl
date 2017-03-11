# legendre.jl

"""
A basis of Legendre polynomials on the interval [-1,1].
"""
immutable LegendreBasis{T} <: OPS{T}
    n           ::  Int
end

name(b::LegendreBasis) = "Legendre OPS"

# Constructor with a default numeric type
LegendreBasis{T}(n::Int, ::Type{T} = Float64) = LegendreBasis{T}(n)

instantiate{T}(::Type{LegendreBasis}, n, ::Type{T}) = LegendreBasis{T}(n)

promote_eltype{T,S}(b::LegendreBasis{T}, ::Type{S}) = LegendreBasis{promote_type(T,S)}(b.n)

resize(b::LegendreBasis, n) = LegendreBasis(n, eltype(b))


left{T}(b::LegendreBasis{T}) = -T(1)
left{T}(b::LegendreBasis{T}, idx) = -T(1)

right{T}(b::LegendreBasis{T}) = T(1)
right{T}(b::LegendreBasis{T}, idx) = T(1)

#grid(b::LegendreBasis) = LegendreGrid(b.n)


jacobi_α{T}(b::LegendreBasis{T}) = T(0)
jacobi_β{T}(b::LegendreBasis{T}) = T(0)

weight{T}(b::LegendreBasis{T}, x) = T(1)

function gramdiagonal!{T}(result, ::LegendreBasis{T}; options...)
  for i in 1:length(result)
    result[i] = T(2//(2(i-1)+1))
  end
end

# See DLMF, Table 18.9.1
# http://dlmf.nist.gov/18.9#i
rec_An{T}(b::LegendreBasis{T}, n::Int) = T(2*n+1)/T(n+1)

rec_Bn{T}(b::LegendreBasis{T}, n::Int) = zero(T)

rec_Cn{T}(b::LegendreBasis{T}, n::Int) = T(n)/T(n+1)
