
"""
A basis of the classical Jacobi polynomials on the interval `[-1,1]`.
These polynomials are orthogonal with respect to the weight function
```
w(x) = (1-x)^α (1+x)^β.
```
"""
struct JacobiPolynomials{T} <: OPS{T,T}
    n       ::  Int
    α       ::  T
    β       ::  T

    JacobiPolynomials{T}(n, α = zero(T), β = zero(T)) where {T} = new{T}(n, α, β)
end



name(b::JacobiPolynomials) = "Jacobi OPS"

JacobiPolynomials(n::Int) = JacobiPolynomials{Float64}(n)

JacobiPolynomials(n::Int, α::Number, β::Number) = JacobiPolynomials(n, promote(α, β)...)

JacobiPolynomials(n::Int, α::T, β::T) where {T <: AbstractFloat} = JacobiPolynomials{T}(n, α, β)

JacobiPolynomials(n::Int, α::Integer, β::Integer) = JacobiPolynomials(n, float(α), float(β))

similar(b::JacobiPolynomials, ::Type{T}, n::Int) where {T} = JacobiPolynomials{T}(n, b.α, b.β)

instantiate(::Type{JacobiPolynomials}, n::Int, ::Type{T}) where {T} = JacobiPolynomials{T}(n)

support(b::JacobiPolynomials{T}) where {T} = ChebyshevInterval{T}()

first_moment(b::JacobiPolynomials{T}) where {T} = (b.α+b.β+1≈0) ?
    T(2).^(b.α+b.β+1)*gamma(b.α+1)*gamma(b.β+1) :
    T(2).^(b.α+b.β+1)*gamma(b.α+1)*gamma(b.β+1)/(b.α+b.β+1)/gamma(b.α+b.β+1)
    # 2^(b.α+b.β) / (b.α+b.β+1) * gamma(b.α+1) * gamma(b.β+1) / gamma(b.α+b.β+1)


jacobi_α(b::JacobiPolynomials) = b.α
jacobi_β(b::JacobiPolynomials) = b.β

measure(b::JacobiPolynomials) = JacobiMeasure(b.α, b.β)

iscompatible(d1::JacobiPolynomials, d2::JacobiPolynomials) = d1.α == d2.α && d1.β == d2.β

iscompatible(dict::JacobiPolynomials, measure::JacobiMeasure) =
	dict.α == measure.α && dict.β == measure.β


# See DLMF (18.9.2)
# http://dlmf.nist.gov/18.9#i
function rec_An(b::JacobiPolynomials{T}, n::Int) where {T}
    if (n == 0) && (b.α + b.β+1 == 0)
        one(T)/2*(b.α+b.β)+1
    else
        T(2*n + b.α + b.β + 1) * (2*n + b.α + b.β + 2) / T(2 * (n+1) * (n + b.α + b.β + 1))
    end
end

function rec_Bn(b::JacobiPolynomials{T}, n::Int) where {T}
    if (n == 0) && ((b.α + b.β + 1 == 0) || (b.α+b.β == 0))
        one(T)/2*(b.α-b.β)
    else
        T(b.α^2 - b.β^2) * (2*n + b.α + b.β + 1) / T(2 * (n+1) * (n + b.α + b.β + 1) * (2*n + b.α + b.β))
    end
end

rec_Cn(b::JacobiPolynomials{T}, n::Int) where {T} =
    T(n + b.α) * (n + b.β) * (2*n + b.α + b.β + 2) / T((n+1) * (n + b.α + b.β + 1) * (2*n + b.α + b.β))

function innerproduct_native(d1::JacobiPolynomials, i::PolynomialDegree, d2::JacobiPolynomials, j::PolynomialDegree, measure::JacobiMeasure; options...)
	T = coefficienttype(d1)
	if iscompatible(d1, d2) && iscompatible(d1, measure)
		if i == j
			a = d1.α
			b = d1.β
			n = convert(T, value(i))
			2^(a+b+1)/(2n+a+b+1) * gamma(n+a+1)*gamma(n+b+1)/factorial(n)/gamma(n+a+b+1)
		else
			zero(T)
		end
	else
		innerproduct1(d1, i, d2, j, measure; options...)
	end
end


# TODO: move to its own file and make more complete
# Or better yet: implement in terms of Jacobi polynomials
struct UltrasphericalBasis{T} <: OPS{T,T}
	n		::	Int
	alpha	::	T
end

jacobi_α(b::UltrasphericalBasis) = b.α
jacobi_β(b::UltrasphericalBasis) = b.α

weight(b::UltrasphericalBasis, x) = (1-x)^(b.α) * (1+x)^(b.α)
