
"""
A basis of the classicale Hermite polynomials. These polynomials are orthogonal
on the real line `(-∞,∞)` with respect to the weight function
`w(x)=exp(-x^2)`.
"""
struct Hermite{T} <: OPS{T}
    n           ::  Int
end

Hermite(n::Int) = Hermite{Float64}(n)

similar(b::Hermite, ::Type{T}, n::Int) where {T} = Hermite{T}(n)

support(b::Hermite{T}) where {T} = DomainSets.FullSpace(T)

first_moment(b::Hermite{T}) where {T} = sqrt(T(pi))

measure(b::Hermite{T}) where {T} = HermiteWeight{T}()
interpolation_grid(b::Hermite{T}) where T = HermiteNodes{T}(length(b))
iscompatiblegrid(b::Hermite,grid::HermiteNodes) = length(b) == length(grid)
isorthogonal(dict::Hermite, measure::GaussHermite) = opsorthogonal(dict, measure)
isorthogonal(::Hermite, ::HermiteWeight) = true
issymmetric(::Hermite) = true

gauss_rule(dict::Hermite{T}) where T = GaussHermite{T}(length(dict))

# See DLMF, Table 18.9.1
# http://dlmf.nist.gov/18.9#i
rec_An(b::Hermite, n::Int) = 2
rec_Bn(b::Hermite, n::Int) = 0
rec_Cn(b::Hermite, n::Int) = 2*n

function dict_innerproduct_native(d1::Hermite, i::PolynomialDegree, d2::Hermite, j::PolynomialDegree, measure::HermiteWeight; options...)
	T = coefficienttype(d1)
	if i == j
		sqrt(convert(T, pi)) * (1<<value(i)) * factorial(value(i))
	else
		zero(T)
	end
end

show(io::IO, b::Hermite{Float64}) = print(io, "Hermite($(length(b)))")
show(io::IO, b::Hermite{T}) where T = print(io, "Hermite{$(T)}($(length(b)))")
