
"A basis of Chebyshev polynomials of the second kind on the interval `[-1,1]`."
struct ChebyshevU{T} <: OPS{T}
    n			::	Int
end

ChebyshevU(n::Int) = ChebyshevU{Float64}(n)

similar(b::ChebyshevU, ::Type{T}, n::Int) where {T} = ChebyshevU{T}(n)

show(io::IO, d::ChebyshevU{Float64}) = print(io, "ChebyshevU($(length(d)))")
show(io::IO, d::ChebyshevU{T}) where T = print(io, "ChebyshevU{$(T)}($(length(d)))")

function unsafe_eval_element(b::ChebyshevU, idx::PolynomialDegree, x::Real)
    # Don't use the formula when |x|=1, because it will generate NaN's
    d = degree(idx)
    abs(x) < 1 ? sin((d+1)*acos(x))/sqrt(1-x^2) : recurrence_eval(b, idx, x)
end

first_moment(b::ChebyshevU{T}) where {T} = convert(T, pi)/2

interpolation_grid(b::ChebyshevU{T}) where {T} = ChebyshevUNodes{T}(b.n)

iscompatiblegrid(dict::ChebyshevU, grid::ChebyshevUNodes) = length(dict) == length(grid)
issymmetric(::ChebyshevU) = true
measure(dict::ChebyshevU{T}) where {T} = ChebyshevUWeight{T}()
hasmeasure(::ChebyshevU) = true

function dict_innerproduct_native(b1::ChebyshevU, i::PolynomialDegree, b2::ChebyshevU, j::PolynomialDegree, m::ChebyshevUWeight;
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


"A Chebyshev polynomial of the second kind"
struct ChebyshevUPolynomial{T} <: OrthogonalPolynomial{T}
    degree  ::  Int
end

ChebyshevUPolynomial(args...; options...) = ChebyshevUPolynomial{Float64}(args...; options...)
ChebyshevUPolynomial{T}(p::ChebyshevUPolynomial) where {T} = ChebyshevUPolynomial{T}(p.degree)
ChebyshevUPolynomial{T}(; degree) where {T} = ChebyshevUPolynomial(degree)

show(io::IO, p::ChebyshevUPolynomial) = print(io, "U_$(degree(p))(x) (Chebyshev polynomial of the second kind)")

convert(::Type{TypedFunction{T,T}}, p::ChebyshevUPolynomial) where {T} = ChebyshevUPolynomial{T}(p.degree)

dictionary(p::ChebyshevUPolynomial{T}) where {T} = ChebyshevU{T}(degree(p)+1)
index(p::ChebyshevUPolynomial) = degree(p)+1

(p::ChebyshevUPolynomial{T})(x) where {T} = eval_element(ChebyshevU{T}(degree(p)+1), degree(p)+1, x)

basisfunction(dict::ChebyshevU, idx) = basisfunction(dict, native_index(dict, idx))
basisfunction(dict::ChebyshevU{T}, idx::PolynomialDegree) where {T} = ChebyshevUPolynomial{T}(degree(idx))
