# hermite.jl

"A Hermite polynomial basis."
struct HermiteBasis{T} <: OPS{T}
    n           ::  Int
end

const HermiteSpan{A, F <: HermiteBasis} = Span{A,F}

name(b::HermiteBasis) = "Hermite OPS"

# Constructor with a default numeric type
HermiteBasis(n::Int, ::Type{T} = Float64) where {T} = HermiteBasis{T}(n)

instantiate(::Type{HermiteBasis}, n, ::Type{T}) where {T} = HermiteBasis{T}(n)

set_promote_domaintype(b::HermiteBasis, ::Type{S}) where {S} = HermiteBasis{S}(b.n)

resize(b::HermiteBasis, n) = HermiteBasis(n, eltype(b))


left(b::HermiteBasis{T}) where {T} = -convert(T, Inf)
left(b::HermiteBasis, idx) = left(b)

right(b::HermiteBasis{T}) where {T} = convert(T, Inf)
right(b::HermiteBasis, idx) = right(b)

#grid(b::HermiteBasis) = HermiteGrid(b.n)


weight(b::HermiteBasis{T}, x) where {T} = exp(-T(x)^2)


# See DLMF, Table 18.9.1
# http://dlmf.nist.gov/18.9#i
rec_An(b::HermiteBasis, n::Int) = 2

rec_Bn(b::HermiteBasis, n::Int) = 0

rec_Cn(b::HermiteBasis, n::Int) = 2*n

function gramdiagonal!(result, ::HermiteSpan; options...)
    T = eltype(result)
    for i in 1:length(result)
        result[i] = sqrt(T(pi))*(1<<(i-1))*factorial(i-1)
    end
end
