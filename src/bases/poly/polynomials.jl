
"PolynomialBasis is the abstract supertype of all univariate polynomials."
abstract type PolynomialBasis{T} <: Dictionary{T,T}
end

span_isequal(d1::PolynomialBasis, d2::PolynomialBasis) = length(d1) == length(d2)
span_issubset(d1::PolynomialBasis, d2::PolynomialBasis) = length(d1) <= length(d2)


##################
# Native indices
##################

"""
The native index of a polynomial basis is the degree, which starts from 0 rather
than from 1.
"""
struct PolynomialDegree <: AbstractShiftedIndex{1}
	value	::	Int
end

degree(idx::PolynomialDegree) = value(idx)

Base.show(io::IO, idx::PolynomialDegree) =
	print(io, "Polynomial degree $(degree(idx))")

ordering(b::PolynomialBasis) = ShiftedIndexList(length(b), PolynomialDegree)

isbasis(b::PolynomialBasis) = true

function sub(b::PolynomialBasis, idx::OrdinalRange)
    if (step(idx) == 1) && (first(idx) == 1) && (last(idx) <= length(b))
        resize(b, last(idx))
    else
        defaultsub(b, idx)
    end
end

"Supertype of polynomial basis functions."
abstract type Polynomial{T} <: AbstractBasisFunction{T,T} end

degree(p::Polynomial) = p.degree
