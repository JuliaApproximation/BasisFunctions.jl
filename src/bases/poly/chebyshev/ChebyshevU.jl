
"A basis of Chebyshev polynomials of the second kind on the interval `[-1,1]`."
struct ChebyshevU{T} <: IntervalOPS{T}
    n			::	Int
end

ChebyshevU(n::Int) = ChebyshevU{Float64}(n)

ChebyshevU(d::PolynomialDegree) = ChebyshevU(value(d)+1)
ChebyshevU{T}(d::PolynomialDegree) where T = ChebyshevU{T}(value(d)+1)

similar(b::ChebyshevU, ::Type{T}, n::Int) where {T} = ChebyshevU{T}(n)
isequaldict(b1::ChebyshevU, b2::ChebyshevU) = length(b1)==length(b2)

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

function dict_innerproduct_native(b1::ChebyshevU, i::PolynomialDegree,
            b2::ChebyshevU, j::PolynomialDegree, m::ChebyshevUWeight; options...)
    T = promote_type(domaintype(b1), domaintype(b2))
	i == j ? chebyshevu_hn(value(i), T) : zero(T)
end


# Parameters alpha and beta of the corresponding Jacobi polynomial
jacobi_α(b::ChebyshevU{T}) where T = one(T)/2
jacobi_β(b::ChebyshevU{T}) where T = one(T)/2


# Recurrence relation
rec_An(b::ChebyshevU{T}, n::Int) where T = chebyshevu_rec_An(n, T)
rec_Bn(b::ChebyshevU{T}, n::Int) where T = chebyshevu_rec_Bn(n, T)
rec_Cn(b::ChebyshevU{T}, n::Int) where T = chebyshevu_rec_Cn(n, T)

same_ops_family(b1::ChebyshevU, b2::ChebyshevU) = true

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

hasx(b::ChebyshevU) = length(b) >= 2
coefficients_of_x(b::ChebyshevU) = (c=zeros(b); c[2]=one(eltype(c))/2; c)
