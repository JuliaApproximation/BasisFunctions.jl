# laguerre.jl

"""
A basis of the classicale Laguerre polynomials. These polynomials are orthogonal
on the positive halfline `[0,∞)` with respect to the weight function
`w(x)=exp(-x)`.
"""
struct LaguerrePolynomials{T} <: OPS{T,T}
    n       ::  Int
    α       ::  T

    LaguerrePolynomials{T}(n::Int, α = zero(T)) where {T} = new(n, α)
end

const LaguerreSpan{A,S,T,D <: LaguerrePolynomials} = Span{A,S,T,D}

name(b::LaguerrePolynomials) = "Laguerre OPS"

LaguerrePolynomials(n, ::Type{T} = Float64) where {T} = LaguerrePolynomials{T}(n)

LaguerrePolynomials(n, α::T) where {T <: Integer} = LaguerrePolynomials(n, float(α))

LaguerrePolynomials(n, α::T) where {T <: Number} = LaguerrePolynomials{T}(n, α)


instantiate(::Type{LaguerrePolynomials}, n, ::Type{T}) where {T} = LaguerrePolynomials{T}(n)

dict_promote_domaintype(b::LaguerrePolynomials, ::Type{S}) where {S} =
    LaguerrePolynomials{S}(b.n, b.α)

resize(b::LaguerrePolynomials, n) = LaguerrePolynomials(n, b.α)

left(b::LaguerrePolynomials{T}) where {T} = T(0)
left(b::LaguerrePolynomials, idx) = left(b)

right(b::LaguerrePolynomials) = convert(domaintype(b), Inf)
right(b::LaguerrePolynomials, idx) = right(b)

first_moment(b::LaguerrePolynomials{T}) where {T} = gamma(b.α+1)

jacobi_α(b::LaguerrePolynomials) = b.α


weight(b::LaguerrePolynomials{T}, x) where {T} = exp(-T(x)) * T(x)^(b.α)

function gramdiagonal!(result, b::LaguerreSpan; options...)
    T = eltype(result)
    for i in 1:length(result)
        result[i] = gamma(T(i+jacobi_α(dictionary(b))))/factorial(i-1)
    end
end



# See DLMF, Table 18.9.1
# http://dlmf.nist.gov/18.9#i
rec_An(b::LaguerrePolynomials{T}, n::Int) where {T} = -T(1) / T(n+1)

rec_Bn(b::LaguerrePolynomials{T}, n::Int) where {T} = T(2*n + b.α + 1) / T(n+1)

rec_Cn(b::LaguerrePolynomials{T}, n::Int) where {T} = T(n + b.α) / T(n+1)

domain(b::LaguerrePolynomials{T}) where {T} = halfline(T)
