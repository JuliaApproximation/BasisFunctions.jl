
"""
A basis of Legendre polynomials on the interval `[-1,1]`. These classical
polynomials are orthogonal with respect to the weight function `w(x) = 1`.
"""
struct Legendre{T} <: OPS{T,T}
    n   ::  Int
end

Legendre(n::Int) = Legendre{Float64}(n)

instantiate(::Type{Legendre}, n, ::Type{T}) where {T} = Legendre{T}(n)

similar(b::Legendre, ::Type{T}, n::Int) where {T} = Legendre{T}(n)

support(b::Legendre{T}) where {T} = ChebyshevInterval{T}()

first_moment(b::Legendre{T}) where {T} = T(2)

jacobi_α(b::Legendre{T}) where {T} = T(0)
jacobi_β(b::Legendre{T}) where {T} = T(0)

measure(b::Legendre{T}) where {T} = LegendreMeasure{T}()

function innerproduct_native(d1::Legendre, i::PolynomialDegree, d2::Legendre, j::PolynomialDegree, m::LegendreMeasure; options...)
	T = coefficienttype(d1)
	if i == j
		2 / convert(T, 2*value(i)+1)
	else
		zero(T)
	end
end

function quadraturenormalization(gb, grid::OPSNodes{<:Legendre}, ::LegendreMeasure; options...)
	T = eltype(grid)
	x, w = gauss_rule(Legendre{T}(length(grid)))
	DiagonalOperator(gb, w)
end

# See DLMF, Table 18.9.1
# http://dlmf.nist.gov/18.9#i
rec_An(b::Legendre{T}, n::Int) where {T} = T(2*n+1)/T(n+1)

rec_Bn(b::Legendre{T}, n::Int) where {T} = zero(T)

rec_Cn(b::Legendre{T}, n::Int) where {T} = T(n)/T(n+1)
