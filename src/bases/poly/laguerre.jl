
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



name(b::LaguerrePolynomials) = "Laguerre OPS"

LaguerrePolynomials(n::Int) = LaguerrePolynomials{Float64}(n)

LaguerrePolynomials(n::Int, α::T) where {T <: AbstractFloat} = LaguerrePolynomials{T}(n, α)

LaguerrePolynomials(n::Int, α::Integer) = LaguerrePolynomials(n, float(α))


instantiate(::Type{LaguerrePolynomials}, n, ::Type{T}) where {T} = LaguerrePolynomials{T}(n)

similar(b::LaguerrePolynomials, ::Type{T}, n::Int) where {T} = LaguerrePolynomials{T}(n, b.α)

support(b::LaguerrePolynomials{T}) where {T} = HalfLine{T}()

first_moment(b::LaguerrePolynomials{T}) where {T} = gamma(b.α+1)

jacobi_α(b::LaguerrePolynomials) = b.α


measure(b::LaguerrePolynomials) = LaguerreMeasure(b.α)

iscompatible(d1::LaguerrePolynomials, d2::LaguerrePolynomials) = d1.α == d2.α

iscompatible(dict::LaguerrePolynomials, measure::LaguerreMeasure) = dict.α == measure.α

function innerproduct_native(d1::LaguerrePolynomials, i::PolynomialDegree, d2::LaguerrePolynomials, j::PolynomialDegree, measure::LaguerreMeasure; options...)
	T = coefficienttype(d1)
	if iscompatible(d1, d2) && iscompatible(d1, measure)
		if i == j
			gamma(convert(T, value(i)+1+jacobi_α(d1))) / convert(T, factorial(value(i)))
		else
			zero(T)
		end
	else
		innerproduct1(d1, i, d2, j, measure; options...)
	end
end



# See DLMF, Table 18.9.1
# http://dlmf.nist.gov/18.9#i
rec_An(b::LaguerrePolynomials{T}, n::Int) where {T} = -T(1) / T(n+1)

rec_Bn(b::LaguerrePolynomials{T}, n::Int) where {T} = T(2*n + b.α + 1) / T(n+1)

rec_Cn(b::LaguerrePolynomials{T}, n::Int) where {T} = T(n + b.α) / T(n+1)
