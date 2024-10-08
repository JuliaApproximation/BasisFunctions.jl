
"""
A basis of Legendre polynomials on the interval `[-1,1]`. These classical
polynomials are orthogonal with respect to the weight function `w(x) = 1`.
"""
struct Legendre{T} <: IntervalOPS{T}
    n   ::  Int
end

Legendre(n::Int) = Legendre{Float64}(n)

Legendre(d::PolynomialDegree) = Legendre(value(d)+1)
Legendre{T}(d::PolynomialDegree) where T = Legendre{T}(value(d)+1)

similar(b::Legendre, ::Type{T}, n::Int) where {T} = Legendre{T}(n)

to_legendre_dict(dict::IntervalOPS{T}) where T = Legendre{T}(length(dict))
to_legendre(f) = to_legendre(expansion(f))
to_legendre(f::Expansion) = to_legendre(dictionary(f), coefficients(f))
to_legendre(dict::Dictionary, coef) =
	conversion(dict, to_legendre_dict(dict)) * expansion(dict, coef)

first_moment(b::Legendre{T}) where {T} = T(2)

jacobi_α(b::Legendre{T}) where {T} = T(0)
jacobi_β(b::Legendre{T}) where {T} = T(0)

measure(b::Legendre{T}) where {T} = LegendreWeight{T}()
isorthogonal(::Legendre, ::LegendreWeight) = true
issymmetric(::Legendre) = true
interpolation_grid(dict::Legendre{T}) where T = LegendreNodes{T}(length(dict))
iscompatiblegrid(dict::Legendre, grid::LegendreNodes) = length(dict) == length(grid)
isorthogonal(dict::Legendre, measure::GaussLegendre) = opsorthogonal(dict, measure)

gauss_rule(dict::Legendre{T}) where T = GaussLegendre{T}(length(dict))

function dict_innerproduct_native(d1::Legendre, i::PolynomialDegree, d2::Legendre, j::PolynomialDegree, m::LegendreWeight; options...)
	T = coefficienttype(d1)
	if i == j
		2 / convert(T, 2*value(i)+1)
	else
		zero(T)
	end
end

# See DLMF, Table 18.9.1
# http://dlmf.nist.gov/18.9#i
rec_An(b::Legendre{T}, n::Int) where {T} = T(2*n+1)/T(n+1)
rec_Bn(b::Legendre{T}, n::Int) where {T} = zero(T)
rec_Cn(b::Legendre{T}, n::Int) where {T} = T(n)/T(n+1)

show(io::IO, b::Legendre{Float64}) = print(io, "Legendre($(length(b)))")
show(io::IO, b::Legendre{T}) where T = print(io, "Legendre{$(T)}($(length(b)))")

hasderivative(b::Legendre) = true
function differentiation(::Type{T}, src::Legendre, dest::Legendre, order::Int; options...) where {T}
	@assert order >= 0
	if order == 0
		IdentityOperator{T}(src, dest)
	elseif order == 1
		A = zeros(T, length(dest), length(src))
		n = length(src)
		for i in 0:n-1
			for j in i-1:-2:0
				A[j+1,i+1] = 2*(i-1-2*((i-j)>>1))+1
			end
		end
		ArrayOperator{T}(A, src, dest)
	else
		error("Higher order differentiation of Legendre not implemented.")
	end
end


hasx(b::Legendre) = length(b) >= 2
coefficients_of_x(b::Legendre) = (c=zeros(b); c[2]=1; c)
