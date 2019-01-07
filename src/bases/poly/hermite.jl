
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

measure(b::HermitePolynomials{T}) where {T} = HermiteMeasure{T}()


# See DLMF, Table 18.9.1
# http://dlmf.nist.gov/18.9#i
rec_An(b::HermitePolynomials, n::Int) = 2

rec_Bn(b::HermitePolynomials, n::Int) = 0

rec_Cn(b::HermitePolynomials, n::Int) = 2*n

function innerproduct(d1::HermitePolynomials, i::PolynomialDegree, d2::HermitePolynomials, j::PolynomialDegree, measure::HermiteMeasure; options...)
	T = coefficienttype(d1)
	if i == j
		sqrt(convert(T, pi)) * (1<<value(i)) * factorial(value(i))
	else
		zero(T)
	end
end
