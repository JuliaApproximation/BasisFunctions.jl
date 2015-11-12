# bf_polynomials.jl


abstract PolynomialBasis{T} <: AbstractBasis1d{T}



abstract OrthogonalPolynomialBasis{T} <: PolynomialBasis{T}

typealias OPS{T} OrthogonalPolynomialBasis{T}


isreal(ops::PolynomialBasis) = True()
isreal{O <: PolynomialBasis}(::Type{O}) = True



#######################
# The monomial basis
#######################

# A basis of the monomials x^i
immutable MonomialBasis{T} <: PolynomialBasis{T}
    n   ::  Int
end


length(o::OrthogonalPolynomialBasis) = o.n



# Evaluate an orthogonal polynomial using the three term recurrence relation.
# The recurrence relation is assumed to have the form
#
#    p_{n+1}(x) = (A_n x - B_n) * p_n(x) - C_n * p_{n-1}(x)
#
# with the coefficients implemented by the rec_An, rec_Bn and rec_Cn functions.
function recurrence_eval{T,S <: Number}(b::OPS{T}, idx::Int, x::S)
    z0 = one(T)
    z1 = rec_An(b, 0) * T(x) + rec_Bn(b, 0)

    if idx == 1
        return z0
    end
    if idx == 2
        return z1
    end

    z = z1
    for i = 1:idx-2
        z = (rec_An(b, i)*x + rec_Bn(b, i)) * z1 - rec_Cn(b, i) * z0
        z0 = z1
        z1 = z
    end
    z
end


# Default evaluation of an orthogonal polynomial: invoke the recurrence relation
call{T}(b::OPS{T}, idx::Int, x::T) = recurrence_eval(b, idx, x)
call{T}(b::OPS{T}, idx::Int, x::Complex{T}) = recurrence_eval(b, idx, x)

# Convert argument x of numeric type S to type T
call{T, S <: Number}(b::OPS{T}, idx::Int, x::S) = call(b, idx, T(s))
call{T, S <: Number}(b::OPS{T}, idx::Int, x::Complex{S}) = call(b, idx, Complex{T}(x))



# TODO: move to its own file and make more complete
# Or better yet: implement in terms of Jacobi polynomials
immutable UltrasphericalBasis{T} <: OPS{T}
	n		::	Int
	alpha	::	T
end

jacobi_alpha(b::UltrasphericalBasis) = b.alpha
jacobi_beta(b::UltrasphericalBasis) = b.alpha

weight(b::UltrasphericalBasis, x) = (1-x)^(b.alpha) * (1+x)^(b.alpha)





