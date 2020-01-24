
export Monomials,
    Monomial

"A basis of the monomials `x^i`."
struct Monomials{T} <: PolynomialBasis{T}
    n   ::  Int     # the degrees go from 0 to n-1
end

Monomials(n) = Monomials{Float64}(n)

support(dict::Monomials{T}) where {T} = DomainSets.FullSpace{T}()

size(dict::Monomials) = (dict.n,)

unsafe_eval_element(b::Monomials, idxn::PolynomialDegree, x) = x^degree(idxn)

function unsafe_eval_element_derivative(b::Monomials, idxn::PolynomialDegree, x)
    i = degree(idxn)
    T = codomaintype(b)
    i == 0 ? zero(T) : i*x^(i-1)
end

similar(b::Monomials, ::Type{T}, n::Int) where {T} = Monomials{T}(n)



struct Monomial{T} <: Polynomial{T}
    degree  ::  Int
end

Monomial{T}(m::Monomial) where {T} = Monomial{T}(m.degree)

convert(::Type{TypedFunction{T,T}}, d::Monomial) where {T} = Monomial{T}(d.degree)

degree(m::Monomial) = m.degree

(m::Monomial)(x) = x^degree(m)

(*)(m1::Monomial, m2::Monomial) = (*)(promote(m1,m2)...)

(*)(m1::Monomial{T}, m2::Monomial{T}) where {T} = Monomial{T}(degree(m1)+degree(m2))


basisfunction(dict::Monomials, idx) = basisfunction(dict, native_index(dict, idx))
basisfunction(dict::Monomials{T}, idx::PolynomialDegree) where {T} = Monomial{T}(degree(idx))
