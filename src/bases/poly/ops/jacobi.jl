
# Supporting functions

# See DLMF (18.9.2)
# http://dlmf.nist.gov/18.9#i
function jacobi_rec_An(α::T, β::T, n::Int) where T
    if (n == 0) && (α + β + 1 == 0)
        one(T)/2*(α+β)+1
    else
        T(2*n + α + β + 1) * (2n + α + β + 2) / T(2 * (n+1) * (n + α + β + 1))
    end
end
function jacobi_rec_Bn(α::T, β::T, n::Int) where T
    if (n == 0) && ((α + β + 1 == 0) || (α+β == 0))
        one(T)/2*(α-β)
    else
        T(α^2 - β^2) * (2*n + α + β + 1) / T(2 * (n+1) * (n + α + β + 1) * (2*n + α + β))
    end
end
function jacobi_rec_Cn(α::T, β::T, n::Int) where T
    T(n + α) * (n + β) * (2*n + α + β + 2) / T((n+1) * (n + α + β + 1) * (2*n + α + β))
end

# TODO: move these elsewhere. But where?
# The packages GridArrays (which defines the nodes) and DomainIntegrals (which define the
# jacobi_x functions) do not depend on each other.
jacobi_α(x::GridArrays.JacobiNodes) = x.α
jacobi_β(x::GridArrays.JacobiNodes) = x.β
jacobi_α(w::GridArrays.JacobiWeights) = w.α
jacobi_β(w::GridArrays.JacobiWeights) = w.β


"Abstract supertype of Jacobi polynomials."
abstract type AbstractJacobi{T} <: IntervalOPS{T} end

iscompatible(b1::AbstractJacobi, b2::AbstractJacobi) =
	jacobi_α(b1) == jacobi_α(b2) && jacobi_β(b1) == jacobi_β(b2)
isequaldict(b1::AbstractJacobi, b2::AbstractJacobi) =
	length(b1)==length(b2) && iscompatible(b1,b2)

isorthogonal(b::AbstractJacobi, μ::DomainIntegrals.AbstractJacobiWeight) =
	jacobi_α(b) == jacobi_α(μ) && jacobi_β(b) == jacobi_β(μ)

issymmetric(dict::AbstractJacobi) = jacobi_α(dict)≈jacobi_β(dict)

rec_An(b::AbstractJacobi, n::Int) = jacobi_rec_An(jacobi_α(b), jacobi_β(b), n)
rec_Bn(b::AbstractJacobi, n::Int) = jacobi_rec_Bn(jacobi_α(b), jacobi_β(b), n)
rec_Cn(b::AbstractJacobi, n::Int) = jacobi_rec_Cn(jacobi_α(b), jacobi_β(b), n)

interpolation_grid(b::AbstractJacobi) = JacobiNodes(length(b), jacobi_α(b), jacobi_β(b))
iscompatiblegrid(b::AbstractJacobi, grid::JacobiNodes) = 
	length(b) == length(grid) &&
	jacobi_α(b) ≈ jacobi_α(grid) && jacobi_β(b) ≈ jacobi_β(grid)
isorthogonal(b::AbstractJacobi, μ::GaussJacobi) =
	jacobi_α(b) ≈ jacobi_α(μ) && jacobi_β(b) ≈ jacobi_β(μ) && opsorthogonal(b, μ)

gauss_rule(b::AbstractJacobi) = GaussJacobi(length(b), jacobi_α(b), jacobi_β(b))

first_moment(b::AbstractJacobi) = moment(measure(b))

function dict_innerproduct_native(d1::AbstractJacobi, i::PolynomialDegree, d2::AbstractJacobi, j::PolynomialDegree, measure::AbstractJacobiWeight; options...)
	T = coefficienttype(d1)
	if iscompatible(d1, d2) && isorthogonal(d1, measure)
		if i == j
			a = jacobi_α(d1)
			b = jacobi_β(d1)
			n = value(i)
			2^(a+b+1)/(2n+a+b+1) * gamma(n+a+1)*gamma(n+b+1)/factorial(n)/gamma(n+a+b+1)
		else
			zero(T)
		end
	else
		dict_innerproduct1(d1, i, d2, j, measure; options...)
	end
end



"""
A basis of the classical Jacobi polynomials on the interval `[-1,1]`.
These polynomials are orthogonal with respect to the weight function
```
w(x) = (1-x)^α (1+x)^β.
```
"""
struct Jacobi{T} <: AbstractJacobi{T}
    n       ::  Int
    α       ::  T
    β       ::  T

    function Jacobi{T}(n, α = 0, β = 0) where T
		@assert α > -1 && β > -1
		new{T}(n, α, β)
	end
end

Jacobi(n::Int; α = 0, β = 0) = Jacobi(n, α, β)
Jacobi(n::Int, α, β) = Jacobi(n, promote(α, β)...)
Jacobi(n::Int, α::T, β::T) where {T <: AbstractFloat} = Jacobi{T}(n, α, β)
Jacobi(n::Int, α::S, β::S) where {S} = Jacobi(n, float(α), float(β))

Jacobi(d::PolynomialDegree, args...; options...) =
	Jacobi(value(d)+1, args...; options...)
Jacobi{T}(d::PolynomialDegree, args...; options...) where T =
	Jacobi{T}(value(d)+1, args...; options...)

const JacobiExpansion{T,C} = Expansion{T,T,Jacobi{T},C}

similar(b::Jacobi, ::Type{T}, n::Int) where {T} = Jacobi{T}(n, b.α, b.β)

jacobi_α(b::Jacobi) = b.α
jacobi_β(b::Jacobi) = b.β

measure(b::Jacobi) = JacobiWeight(b.α, b.β)



"""
The basis of ultraspherical orthogonal polynomials on `[-1,1]`.

They are orthogonal with respect to the weight function
```
w(x) = (1-x^2)^(λ-1/2).
```
"""
struct Ultraspherical{T} <: AbstractJacobi{T}
	n		::	Int
	λ	    ::	T
end

const Gegenbauer = Ultraspherical

jacobi_α(b::Ultraspherical{T}) where T = b.λ - one(T)/2
jacobi_β(b::Ultraspherical{T}) where T = b.λ - one(T)/2

measure(b::Ultraspherical{T}) where T = DomainIntegrals.UltrasphericalWeight(b.λ)


## Printing
function show(io::IO, b::Jacobi{Float64})
	if jacobi_α(b) == 0 && jacobi_β(b) == 0
		print(io, "Jacobi($(length(b)))")
	else
		print(io, "Jacobi($(length(b)); α = $(jacobi_α(b)), β = $(jacobi_β(b)))")
	end
end

function show(io::IO, b::Jacobi{T}) where T
	if jacobi_α(b) == 0 && jacobi_β(b) == 0
		print(io, "Jacobi{$(T)}($(length(b)))")
	else
		print(io, "Jacobi{$(T)}($(length(b)); α = $(jacobi_α(b)), β = $(jacobi_β(b)))")
	end
end

show(io::IO, b::Ultraspherical{Float64}) = print(io, "Ultraspherical($(length(b)), $(b.λ))")
show(io::IO, b::Ultraspherical{T}) where T =
	print(io, "Ultraspherical{$(T)}($(length(b)), $(b.λ))")
