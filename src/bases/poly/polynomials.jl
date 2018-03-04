# polynomials.jl

"PolynomialBasis is the abstract supertype of all univariate polynomials."
abstract type PolynomialBasis{S,T} <: Dictionary{S,T}
end

# The native index of a polynomial basis is the degree, which starts from 0 rather
# than from 1. Since it is an integer, it is wrapped in a different type.
struct PolynomialDegree <: NativeIndex
	index	::	Int
end

# Indices of polynomials naturally start at 0
native_index(b::PolynomialBasis, idx::Int) = PolynomialDegree(idx-1)
linear_index(b::PolynomialBasis, idxn::PolynomialDegree) = index(idxn)+1

is_basis(b::PolynomialBasis) = true

function subdict(b::PolynomialBasis, idx::OrdinalRange)
    if (step(idx) == 1) && (first(idx) == 1) && (last(idx) <= length(b))
        resize(b, last(idx))
    else
        subdict(b, idx)
    end
end
