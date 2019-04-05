
"PolynomialBasis is the abstract supertype of all univariate polynomials."
abstract type PolynomialBasis{S,T} <: Dictionary{S,T}
end


##################
# Native indices
##################

# The native index of a polynomial basis is the degree, which starts from 0 rather
# than from 1. Since it is an integer, it is wrapped in a different type.
const PolynomialDegree = ShiftedIndex{1}

degree(idx::PolynomialDegree) = value(idx)

Base.show(io::IO, idx::BasisFunctions.ShiftedIndex{1}) =
	print(io, "Index shifted by 1: $(degree(idx))")

ordering(b::PolynomialBasis) = ShiftedIndexList{1}(length(b))



isbasis(b::PolynomialBasis) = true

function subdict(b::PolynomialBasis, idx::OrdinalRange)
    if (step(idx) == 1) && (first(idx) == 1) && (last(idx) <= length(b))
        resize(b, last(idx))
    else
        LargeSubdict(b, idx)
    end
end
