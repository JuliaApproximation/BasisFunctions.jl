
"""
A basis of Legendre polynomials on the interval `[-1,1]`. These classical
polynomials are orthogonal with respect to the weight function `w(x) = 1`.
"""
struct LegendrePolynomials{T} <: OPS{T,T}
    n   ::  Int
end

name(b::LegendrePolynomials) = "Legendre OPS"

LegendrePolynomials(n::Int) = LegendrePolynomials{Float64}(n)

instantiate(::Type{LegendrePolynomials}, n, ::Type{T}) where {T} = LegendrePolynomials{T}(n)

similar(b::LegendrePolynomials, ::Type{T}, n::Int) where {T} = LegendrePolynomials{T}(n)

support(b::LegendrePolynomials{T}) where {T} = ChebyshevInterval{T}()

first_moment(b::LegendrePolynomials{T}) where {T} = T(2)

jacobi_α(b::LegendrePolynomials{T}) where {T} = T(0)
jacobi_β(b::LegendrePolynomials{T}) where {T} = T(0)

measure(b::LegendrePolynomials{T}) where {T} = LegendreMeasure{T}()

function innerproduct_native(d1::LegendrePolynomials, i::PolynomialDegree, d2::LegendrePolynomials, j::PolynomialDegree, m::LegendreMeasure; options...)
	T = coefficienttype(d1)
	if i == j
		2 / convert(T, 2*value(i)+1)
	else
		zero(T)
	end
end

# See DLMF, Table 18.9.1
# http://dlmf.nist.gov/18.9#i
rec_An(b::LegendrePolynomials{T}, n::Int) where {T} = T(2*n+1)/T(n+1)

rec_Bn(b::LegendrePolynomials{T}, n::Int) where {T} = zero(T)

rec_Cn(b::LegendrePolynomials{T}, n::Int) where {T} = T(n)/T(n+1)
