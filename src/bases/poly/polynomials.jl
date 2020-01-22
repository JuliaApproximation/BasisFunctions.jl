
"PolynomialBasis is the abstract supertype of all univariate polynomials."
abstract type PolynomialBasis{S,T} <: Dictionary{S,T}
end


##################
# Native indices
##################

# The native index of a polynomial basis is the degree, which starts from 0 rather
# than from 1.
struct PolynomialDegree <: AbstractShiftedIndex{1}
	value	::	Int
end

degree(idx::PolynomialDegree) = value(idx)

Base.show(io::IO, idx::PolynomialDegree) =
	print(io, "Polynomial degree $(degree(idx))")

ordering(b::PolynomialBasis) = ShiftedIndexList(length(b), PolynomialDegree)



isbasis(b::PolynomialBasis) = true

function subdict(b::PolynomialBasis, idx::OrdinalRange)
    if (step(idx) == 1) && (first(idx) == 1) && (last(idx) <= length(b))
        resize(b, last(idx))
    else
        LargeSubdict(b, idx)
    end
end
