
"A basis of Chebyshev polynomials of the second kind on the interval `[-1,1]`."
struct ChebyshevU{T} <: OPS{T}
    n			::	Int
end



ChebyshevU(n::Int) = ChebyshevU{Float64}(n)

instantiate(::Type{ChebyshevU}, n, ::Type{T}) where {T} = ChebyshevU{T}(n)

similar(b::ChebyshevU, ::Type{T}, n::Int) where {T} = ChebyshevU{T}(n)

name(b::ChebyshevU) = "Chebyshev polynomials (second kind)"

function unsafe_eval_element(b::ChebyshevU, idx::PolynomialDegree, x::Real)
    # Don't use the formula when |x|=1, because it will generate NaN's
    d = degree(idx)
    abs(x) < 1 ? sin((d+1)*acos(x))/sqrt(1-x^2) : recurrence_eval(b, idx, x)
end

first_moment(b::ChebyshevU{T}) where {T} = convert(T, pi)/2

interpolation_grid(b::ChebyshevU{T}) where {T} = ChebyshevUNodes{T}(b.n)

iscompatible(dict::ChebyshevU, grid::ChebyshevUNodes) = length(dict) == length(grid)
issymmetric(::ChebyshevU) = true
measure(dict::ChebyshevU{T}) where {T} = ChebyshevUMeasure{T}()
hasmeasure(::ChebyshevU) = true

function innerproduct_native(b1::ChebyshevU, i::PolynomialDegree, b2::ChebyshevU, j::PolynomialDegree, m::ChebyshevUMeasure;
			T = coefficienttype(b1), options...)
	if i == j
		convert(T, pi)/2
	else
		zero(T)
	end
end


# Parameters alpha and beta of the corresponding Jacobi polynomial
jacobi_α(b::ChebyshevU{T}) where {T} = one(T)/2
jacobi_β(b::ChebyshevU{T}) where {T} = one(T)/2


# See DLMF, Table 18.9.1
# http://dlmf.nist.gov/18.9#i
rec_An(b::ChebyshevU{T}, n::Int) where {T} = convert(T, 2)

rec_Bn(b::ChebyshevU{T}, n::Int) where {T} = zero(T)

rec_Cn(b::ChebyshevU{T}, n::Int) where {T} = one(T)

support(b::ChebyshevU{T}) where {T} = ChebyshevInterval{T}()
