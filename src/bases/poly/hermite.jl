
"""
A basis of the classicale Hermite polynomials. These polynomials are orthogonal
on the real line `(-∞,∞)` with respect to the weight function
`w(x)=exp(-x^2)`.
"""
struct HermitePolynomials{T} <: OPS{T,T}
    n           ::  Int
end



name(b::HermitePolynomials) = "Hermite OPS"

instantiate(::Type{HermitePolynomials}, n, ::Type{T}) where {T} = HermitePolynomials{T}(n)

HermitePolynomials(n::Int) = HermitePolynomials{Float64}(n)

similar(b::HermitePolynomials, ::Type{T}, n::Int) where {T} = HermitePolynomials{T}(n)

support(b::HermitePolynomials{T}) where {T} = DomainSets.FullSpace(T)

first_moment(b::HermitePolynomials{T}) where {T} = sqrt(T(pi))

weight(b::HermitePolynomials{T}, x) where {T} = exp(-T(x)^2)


# See DLMF, Table 18.9.1
# http://dlmf.nist.gov/18.9#i
rec_An(b::HermitePolynomials, n::Int) = 2

rec_Bn(b::HermitePolynomials, n::Int) = 0

rec_Cn(b::HermitePolynomials, n::Int) = 2*n

function gramdiagonal!(result, ::HermitePolynomials; options...)
    T = eltype(result)
    for i in 1:length(result)
        result[i] = sqrt(T(pi))*(1<<(i-1))*factorial(i-1)
    end
end
