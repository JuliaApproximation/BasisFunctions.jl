#######################
# The monomial basis
#######################

"""
A basis of the monomials `x^i`.
"""
struct Monomials{T} <: PolynomialBasis{T,T}
    n   ::  Int     # the degrees go from 0 to n-1
end


unsafe_eval_element(b::Monomials, idxn::PolynomialDegree, x) = x^degree(idx)

function unsafe_eval_element_derivative(b::Monomials, idxn::PolynomialDegree, x)
    i = degree(idxn)
    T = codomaintype(b)
    i == 0 ? zero(T) : i*x^(i-1)
end

similar(b::Monomials, ::Type{T}, n::Int) where {T} = Monomials{T}(n)
