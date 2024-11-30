
"""
A basis of the classicale Laguerre polynomials. These polynomials are orthogonal
on the positive halfline `[0,∞)` with respect to the weight function
`w(x)=exp(-x)`.
"""
struct Laguerre{T} <: OPS{T}
    n       ::  Int
    α       ::  T

	Laguerre{T}(n::Int, α = 0) where {T} = new{T}(n, α)
end

Laguerre(n::Int; α = 0) = Laguerre(n, α)
Laguerre(n::Int, α::T) where {T <: AbstractFloat} = Laguerre{T}(n, α)
Laguerre(n::Int, α::S) where {S} = Laguerre(n, float(α))

Laguerre(d::PolynomialDegree, args...; options...) =
	Laguerre(value(d)+1, args...; options...)
Laguerre{T}(d::PolynomialDegree, args...; options...) where T =
	Laguerre{T}(value(d)+1, args...; options...)

similar(b::Laguerre, ::Type{T}, n::Int) where {T} = Laguerre{T}(n, b.α)
isequaldict(b1::Laguerre, b2::Laguerre) = length(b1)==length(b2) &&
	b1.α == b2.α

support(b::Laguerre{T}) where {T} = HalfLine{T}()

first_moment(b::Laguerre{T}) where {T} = gamma(b.α+1)

laguerre_α(b::Laguerre) = b.α


measure(b::Laguerre) = LaguerreWeight(b.α)

iscompatible(b1::Laguerre, b2::Laguerre) = b1.α == b2.α
interpolation_grid(b::Laguerre) = LaguerreNodes(length(b), laguerre_α(b))
iscompatiblegrid(b::Laguerre, grid::LaguerreNodes) = length(b) == length(grid) && laguerre_α(b) ≈ laguerre_α(grid)
isorthogonal(b::Laguerre, μ::GaussLaguerre) = laguerre_α(b) ≈ laguerre_α(μ) && opsorthogonal(b, μ)

isorthonormal(b::Laguerre, μ::LaguerreWeight) = isorthogonal(b, μ) && laguerre_α(b) == 0
isorthonormal(b::Laguerre, μ::GaussLaguerre) = isorthogonal(b, μ) && laguerre_α(b) == 0
issymmetric(::Laguerre) = false

isorthogonal(b::Laguerre, μ::LaguerreWeight) = laguerre_α(b) == laguerre_α(μ)

gauss_rule(dict::Laguerre) = GaussLaguerre(length(dict), dict.α)

function dict_innerproduct_native(b1::Laguerre, i::PolynomialDegree,
		b2::Laguerre, j::PolynomialDegree, measure::LaguerreWeight; options...)
	T = promote_type(domaintype(b1), domaintype(b2))
	if iscompatible(b1, b2) && isorthogonal(b1, measure)
		i == j ? laguerre_hn(value(i), laguerre_α(b1)) : zero(T)
	else
		dict_innerproduct1(b1, i, b2, j, measure; options...)
	end
end

# recurrence relation
rec_An(b::Laguerre, n::Int) = laguerre_rec_An(n, b.α)
rec_Bn(b::Laguerre, n::Int) = laguerre_rec_Bn(n, b.α)
rec_Cn(b::Laguerre, n::Int) = laguerre_rec_Cn(n, b.α)

coefficients_of_x(b::Laguerre) = (c=zeros(b); c[1]=1; c[2]=-1; c)

same_ops_family(b1::Laguerre, b2::Laguerre) = laguerre_α(b1) == laguerre_α(b2)

## Printing

function show(io::IO, b::Laguerre{Float64})
	if laguerre_α(b) == 0
		print(io, "Laguerre($(length(b)))")
	else
		print(io, "Laguerre($(length(b)); α = $(laguerre_α(b)))")
	end
end

function show(io::IO, b::Laguerre{T}) where {T}
	if laguerre_α(b) == 0
		print(io, "Laguerre{$(T)}($(length(b)))")
	else
		print(io, "Laguerre{$(T)}($(length(b)); α = $(laguerre_α(b)))")
	end
end
