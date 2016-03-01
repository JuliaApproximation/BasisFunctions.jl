# bf_polynomials.jl


abstract PolynomialBasis{T} <: AbstractBasis1d{T}

# Indices of polynomials naturally start at 0
natural_index(b::PolynomialBasis, idx) = idx-1
logical_index(b::PolynomialBasis, idxn) = idxn+1


abstract OrthogonalPolynomialBasis{T} <: PolynomialBasis{T}

typealias OPS{T} OrthogonalPolynomialBasis{T}


is_orthogonal{B <: OPS}(::Type{B}) = True
is_biorthogonal{B <: OPS}(::Type{B}) = True

approx_length(b::OPS, n::Int) = n

derivative_space(b::OPS, order::Int) = similar(b, b.n-order)
antiderivative_space(b::OPS, order::Int) = similar(b,b.n+order)

length(o::OrthogonalPolynomialBasis) = o.n


function apply!{B <: OPS}(op::Extension, dest::B, src::B, coef_dest, coef_src)
    @assert length(dest) > length(src)

    for i = 1:length(src)
        coef_dest[i] = coef_src[i]
    end
    for i = length(src)+1:length(dest)
        coef_dest[i] = 0
    end
    coef_dest
end


function apply!{B <: OPS}(op::Restriction, dest::B, src::B, coef_dest, coef_src)
    @assert length(dest) < length(src)

    for i = 1:length(dest)
        coef_dest[i] = coef_src[i]
    end
    coef_dest
end

has_extension(b::OPS) = true


#######################
# The monomial basis
#######################

# A basis of the monomials x^i
immutable MonomialBasis{T} <: PolynomialBasis{T}
    n   ::  Int
end




# Evaluate an orthogonal polynomial using the three term recurrence relation.
# The recurrence relation is assumed to have the form
#
#    p_{n+1}(x) = (A_n x - B_n) * p_n(x) - C_n * p_{n-1}(x)
#
# with the coefficients implemented by the rec_An, rec_Bn and rec_Cn functions.
function recurrence_eval{T,S <: Number}(b::OPS{T}, idx::Int, x::S)
    z0 = one(promote_type(T,S))
    z1 = rec_An(b, 0) * x + rec_Bn(b, 0)

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
call_element(b::OPS, idx::Int, x) = recurrence_eval(b, idx, x)



# TODO: move to its own file and make more complete
# Or better yet: implement in terms of Jacobi polynomials
immutable UltrasphericalBasis{T} <: OPS{T}
	n		::	Int
	alpha	::	T
end

jacobi_α(b::UltrasphericalBasis) = b.α
jacobi_β(b::UltrasphericalBasis) = b.α

weight(b::UltrasphericalBasis, x) = (1-x)^(b.α) * (1+x)^(b.α)





