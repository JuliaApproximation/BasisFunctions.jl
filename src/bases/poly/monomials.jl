
export Monomials,
    Monomial

"A basis of the monomials `x^i`."
struct Monomials{T} <: PolynomialBasis{T}
    n   ::  Int     # the degrees go from 0 to n-1
end

Monomials(n) = Monomials{Float64}(n)

show(io::IO, b::Monomials{Float64}) = print(io, "Monomials($(length(b)))")
show(io::IO, b::Monomials{T}) where T = print(io, "Monomials{$(T)}($(length(b)))")

support(dict::Monomials{T}) where {T} = DomainSets.FullSpace{T}()

size(dict::Monomials) = (dict.n,)

unsafe_eval_element(b::Monomials, idxn::PolynomialDegree, x) = x^degree(idxn)

function unsafe_eval_element_derivative(b::Monomials{T}, idxn::PolynomialDegree, x, order) where {T}
    @assert order > 0
    i = degree(idxn)
    if order > i
        zero(T)
    elseif order == 1
        i*x^(i-1)
    else
        factorial(i) / factorial(i-order) * x^(i-order)
    end
end

similar(b::Monomials, ::Type{T}, n::Int) where {T} = Monomials{T}(n)

extension(::Type{T}, src::Monomials, dest::Monomials; options...) where {T} =
    IndexExtension{T}(src, dest, 1:length(src))

restriction(::Type{T}, src::Monomials, dest::Monomials; options...) where {T} =
    IndexRestriction{T}(src, dest, 1:length(dest))


"A monomial basis function `x^i`."
struct Monomial{T} <: Polynomial{T}
    degree  ::  Int
end

Monomial{T}(p::Monomial) where {T} = Monomial{T}(p.degree)
Monomial(args...) = Monomial{Float64}(args...)
Monomial{T}(i::PolynomialDegree) where {T} = Monomial{T}(value(i))

show(io::IO, p::Monomial) = print(io, "x^$(degree(p)) (monomial)")

convert(::Type{TypedFunction{T,T}}, p::Monomial) where {T} = Monomial{T}(p.degree)

support(::Monomial{T}) where {T} = DomainSets.FullSpace{T}()

(m::Monomial)(x) = x^degree(m)

(*)(p1::Monomial, p2::Monomial) = (*)(promote(p1,p2)...)

(*)(p1::Monomial{T}, p2::Monomial{T}) where {T} = Monomial{T}(degree(p1)+degree(p2))


basisfunction(dict::Monomials, idx) = basisfunction(dict, native_index(dict, idx))
basisfunction(dict::Monomials{T}, idx::PolynomialDegree) where {T} = Monomial{T}(degree(idx))

dictionary(p::Monomial{T}) where {T} = Monomials{T}(degree(p)+1)
