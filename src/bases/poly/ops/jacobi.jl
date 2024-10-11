
"Abstract supertype of Jacobi polynomials."
abstract type AbstractJacobi{T} <: IntervalOPS{T} end

iscompatible(b1::AbstractJacobi, b2::AbstractJacobi) =
	jacobi_α(b1) == jacobi_α(b2) && jacobi_β(b1) == jacobi_β(b2)
isequaldict(b1::AbstractJacobi, b2::AbstractJacobi) =
	length(b1)==length(b2) && iscompatible(b1,b2)

isorthogonal(b::AbstractJacobi, μ::DomainIntegrals.AbstractJacobiWeight) =
	jacobi_α(b) == jacobi_α(μ) && jacobi_β(b) == jacobi_β(μ)

issymmetric(dict::AbstractJacobi) = jacobi_α(dict)≈jacobi_β(dict)

interpolation_grid(b::AbstractJacobi) = JacobiNodes(length(b), jacobi_α(b), jacobi_β(b))
iscompatiblegrid(b::AbstractJacobi, grid::JacobiNodes) = 
	length(b) == length(grid) &&
	jacobi_α(b) ≈ jacobi_α(grid) && jacobi_β(b) ≈ jacobi_β(grid)
isorthogonal(b::AbstractJacobi, μ::GaussJacobi) =
	jacobi_α(b) ≈ jacobi_α(μ) && jacobi_β(b) ≈ jacobi_β(μ) && opsorthogonal(b, μ)

gauss_rule(b::AbstractJacobi) = GaussJacobi(length(b), jacobi_α(b), jacobi_β(b))

first_moment(b::AbstractJacobi) = moment(measure(b))



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

rec_An(b::Jacobi, n::Int) = jacobi_rec_An(jacobi_α(b), jacobi_β(b), n)
rec_Bn(b::Jacobi, n::Int) = jacobi_rec_Bn(jacobi_α(b), jacobi_β(b), n)
rec_Cn(b::Jacobi, n::Int) = jacobi_rec_Cn(jacobi_α(b), jacobi_β(b), n)


function dict_innerproduct_native(b1::Jacobi, i::PolynomialDegree,
		b2::Jacobi, j::PolynomialDegree, μ::JacobiWeight; options...)
	T = promote_type(domaintype(b1), domaintype(b2))
	if iscompatible(b1, b2) && isorthogonal(b1, μ)
		if i == j
			jacobi_hn(jacobi_α(b1), jacobi_β(b1), value(i))
		else
			zero(T)
		end
	else
		dict_innerproduct1(b1, i, b2, j, μ; options...)
	end
end



"""
The basis of ultraspherical (or Gegenbauer) orthogonal polynomials on `[-1,1]`.

They are orthogonal with respect to the weight function
```
w(x) = (1-x^2)^(λ-1/2).
```
This is a Jacobi weight, but ultraspherical polynomials are normalized
differently.
"""
struct Ultraspherical{T} <: AbstractJacobi{T}
	n		::	Int
	λ	    ::	T
end

const Gegenbauer = Ultraspherical

jacobi_α(b::Ultraspherical{T}) where T = b.λ - one(T)/2
jacobi_β(b::Ultraspherical{T}) where T = b.λ - one(T)/2

similar(b::Ultraspherical, ::Type{T}, n::Int) where T = Ultraspherical{T}(n, b.λ)

measure(b::Ultraspherical{T}) where T = UltrasphericalWeight(b.λ)

rec_An(b::Ultraspherical, n::Int) = ultraspherical_rec_An(b.λ, n)
rec_Bn(b::Ultraspherical, n::Int) = ultraspherical_rec_Bn(b.λ, n)
rec_Cn(b::Ultraspherical, n::Int) = ultraspherical_rec_Cn(b.λ, n)


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
