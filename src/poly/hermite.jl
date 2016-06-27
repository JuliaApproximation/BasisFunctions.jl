# hermite.jl

"A Hermite polynomial basis."
immutable HermiteBasis{T} <: OPS{T}
    n           ::  Int
end

@compat (b::HermiteBasis)(x...) = call_set(b, x...)

name(b::HermiteBasis) = "Hermite OPS"

# Constructor with a default numeric type
HermiteBasis{T}(n::Int, ::Type{T} = Float64) = HermiteBasis{T}(n)

instantiate{T}(::Type{HermiteBasis}, n, ::Type{T}) = HermiteBasis{T}(n)

promote_eltype{T,S}(b::HermiteBasis{T}, ::Type{S}) = HermiteBasis{promote_type(T,S)}(b.n)

resize(b::HermiteBasis, n) = HermiteBasis(n, eltype(b))


left{T}(b::HermiteBasis{T}) = -convert(T, Inf)
left{T}(b::HermiteBasis{T}, idx) = left(b)

right{T}(b::HermiteBasis{T}) = convert(T, Inf)
right{T}(b::HermiteBasis{T}, idx) = right(b)

#grid(b::HermiteBasis) = HermiteGrid(b.n)


weight{T}(b::HermiteBasis{T}, x) = exp(-T(x)^2)


# See DLMF, Table 18.9.1
# http://dlmf.nist.gov/18.9#i
rec_An(b::HermiteBasis, n::Int) = 2

rec_Bn(b::HermiteBasis, n::Int) = 0

rec_Cn(b::HermiteBasis, n::Int) = 2*n
