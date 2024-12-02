
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
isequaldict(b1::Legendre, b2::Legendre) = length(b1)==length(b2)

to_legendre_dict(dict::Dictionary{T,T}) where T = Legendre{T}(length(dict))
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

function dict_innerproduct_native(b1::Legendre, i::PolynomialDegree,
		b2::Legendre, j::PolynomialDegree, m::LegendreWeight; options...)
	T = promote_type(domaintype(b1), domaintype(b2))
	i == j ? legendre_hn(value(i), T) : zero(T)
end

rec_An(b::Legendre{T}, n::Int) where T = legendre_rec_An(n, T)
rec_Bn(b::Legendre{T}, n::Int) where T = legendre_rec_Bn(n, T)
rec_Cn(b::Legendre{T}, n::Int) where T = legendre_rec_Cn(n, T)

same_ops_family(b1::Legendre, b2::Legendre) = true

show(io::IO, b::Legendre{Float64}) = print(io, "Legendre($(length(b)))")
show(io::IO, b::Legendre{T}) where T = print(io, "Legendre{$(T)}($(length(b)))")

# hasderivative(b::Legendre) = true
function differentiation_using_recurrences(::Type{T}, src::Legendre; options...) where {T}
	n = length(src)
	A = zeros(T, n, n)
	for i in 0:n-1
		for j in i-1:-2:0
			A[j+1,i+1] = 2*(i-1-2*((i-j)>>1))+1
		end
	end
	UpperTriangular(A)
end


hasx(b::Legendre) = length(b) >= 2
coefficients_of_x(b::Legendre) = (c=zeros(b); c[2]=1; c)
