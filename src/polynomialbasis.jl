# polynomialbasis.jl


abstract PolynomialBasis{T} <: AbstractBasis1d{T}

immutable MonomialBasis{T} <: PolynomialBasis{T}
end


abstract OrthogonalPolynomialBasis{T} <: PolynomialBasis{T}

abstract AbstractJacobiBasis{T} <: OrthogonalPolynomialBasis{T}

immutable ChebyshevBasis{T} <: AbstractJacobiBasis{T}
	n	::	Int
end

alpha{T}(b::ChebyshevBasis{T}) = -one(T)/2
beta{T}(b::ChebyshevBasis{T}) = -one(T)/2

weight(b::ChebyshevBasis, x) = 1/sqrt(1-x^2)

immutable LegendreBasis{T} <: AbstractJacobiBasis{T}
	n	::	Int
end

alpha(b::LegendreBasis) = 0
beta(b::LegendreBasis) = 0

weight(b::LegendreBasis, x) = 1

immutable UltrasphericalBasis{T} <: AbstractJacobiBasis{T}
	n		::	Int
	alpha	::	T
end

alpha(b::UltrasphericalBasis) = b.alpha
beta(b::UltrasphericalBasis) = b.alpha

weight(b::UltrasphericalBasis, x) = (1-x)^(b.alpha) * (1+x)^(b.alpha)

immutable JacobiBasis{T} <: AbstractJacobiBasis{T}
	n		::	Int
	alpha	::	T
	beta	::	T
end

alpha(c::JacobiBasis) = c.alpha
beta(c::JacobiBasis) = c.beta

weight(b::JacobiBasis, x) = (1-x)^(b.alpha) * (1+x)^(b.beta)

immutable HermiteBasis{T} <: OrthogonalPolynomialBasis{T}
	n	::	Int
end

weight(b::HermiteBasis, x) = exp(-x^2)

immutable LaguerreBasis{T} <: OrthogonalPolynomialBasis{T}
	n	::	Int
	alpha	::	T
end

alpha(b::LaguerreBasis) = b.alpha

weight(b::LaguerreBasis, x) = x^(b.alpha) * exp(-x)





