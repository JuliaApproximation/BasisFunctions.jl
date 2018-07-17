# monomials.jl

#######################
# The monomial basis
#######################

"""
A basis of the monomials x^i.
"""
struct Monomials{T} <: PolynomialBasis{T}
    n   ::  Int     # the degrees go from 0 to n-1
end


eval_element(b::Monomials, idx, x) = x^(idx-1)

eval_element_derivative(b::Monomials, idx, x) = idx == 1 ? zero(codomaintype(b)) : (idx-1)*x^(idx-2)
