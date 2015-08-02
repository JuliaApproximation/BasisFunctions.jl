# polynomialbasis.jl


abstract PolynomialBasis{T} <: AbstractBasis1d{T}

immutable MonomialBasis{T} <: PolynomialBasis{T}
end


abstract OrthogonalPolynomialBasis{T} <: PolynomialBasis{T}

typealias OPS{T} OrthogonalPolynomialBasis{T}


length(o::OrthogonalPolynomialBasis) = o.n


## immutable ChebyshevBasis{T} <: OPS{T}
## 	n	::	Int
## end

## ChebyshevBasis{T}(n::Int, ::Type{T} = Float64) = ChebyshevBasis{T}(n)

## jacobi_alpha{T}(b::ChebyshevBasis{T}) = -one(T)/2
## jacobi_beta{T}(b::ChebyshevBasis{T}) = -one(T)/2

## weight(b::ChebyshevBasis, x) = 1/sqrt(1-x^2)

## call(b::ChebyshevBasis, idx::Int, x) = cos( (idx-1) * acos(x))

## grid(b::ChebyshevBasis) = ChebyshevIIGrid(length(b))

## left{T}(b::ChebyshevBasis{T}) = -one(T)
## right{T}(b::ChebyshevBasis{T}) = one(T)

## function apply!(op::TransformOperator, src::DiscreteGridSpace{ChebyshevIIGrid}, dest::ChebyshevBasis, coef_dest, coef_src)
## 	println(22)
## end


immutable LegendreBasis{T} <: OPS{T}
	n	::	Int
end

jacobi_alpha{T}(b::LegendreBasis{T}) = T(0)
jacobi_beta{T}(b::LegendreBasis{T}) = T(0)

weight(b::LegendreBasis, x) = 1



immutable UltrasphericalBasis{T} <: OPS{T}
	n		::	Int
	alpha	::	T
end

jacobi_alpha(b::UltrasphericalBasis) = b.alpha
jacobi_beta(b::UltrasphericalBasis) = b.alpha

weight(b::UltrasphericalBasis, x) = (1-x)^(b.alpha) * (1+x)^(b.alpha)




immutable JacobiBasis{T} <: OPS{T}
	n		::	Int
	alpha	::	T
	beta	::	T
end

jacobi_alpha(c::JacobiBasis) = c.alpha
jacobi_beta(c::JacobiBasis) = c.beta

weight(b::JacobiBasis, x) = (1-x)^(b.alpha) * (1+x)^(b.beta)



immutable HermiteBasis{T} <: OPS{T}
	n	::	Int
end

weight(b::HermiteBasis, x) = exp(-x^2)



immutable LaguerreBasis{T} <: OPS{T}
	n	::	Int
	alpha	::	T
end

laguerre_alpha(b::LaguerreBasis) = b.alpha

weight(b::LaguerreBasis, x) = x^(b.alpha) * exp(-x)





