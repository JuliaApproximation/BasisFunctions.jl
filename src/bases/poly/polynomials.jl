# polynomials.jl

"PolynomialBasis is the abstract supertype of all univariate polynomials."
abstract type PolynomialBasis{S,T} <: Dictionary{S,T}
end


##################
# Native indices
##################

# The native index of a polynomial basis is the degree, which starts from 0 rather
# than from 1. Since it is an integer, it is wrapped in a different type.
const PolynomialDegree = NativeIndex{:degree}

degree(idx::PolynomialDegree) = value(idx)

"""
`DegreeIndexList` defines the map from native indices to linear indices
for a finite polynomial basis. Note that we assume that the elements in the
polynomial basis are ordered according to polynomial degree.
"""
struct DegreeIndexList <: IndexList{PolynomialDegree}
	n	::	Int
end

length(list::DegreeIndexList) = list.n
size(list::DegreeIndexList) = (list.n,)

getindex(list::DegreeIndexList, idx::Int) = PolynomialDegree(idx-1)
getindex(list::DegreeIndexList, idxn::PolynomialDegree) = value(idxn)+1

ordering(b::PolynomialBasis) = DegreeIndexList(length(b))



is_basis(b::PolynomialBasis) = true

function subdict(b::PolynomialBasis, idx::OrdinalRange)
    if (step(idx) == 1) && (first(idx) == 1) && (last(idx) <= length(b))
        resize(b, last(idx))
    else
        subdict(b, idx)
    end
end
