
export Monomials,
    Monomial

"A basis of the monomials `x^i`."
struct Monomials{T} <: PolynomialBasis{T}
    n   ::  Int     # the degrees go from 0 to n-1
end

Monomials(n) = Monomials{Float64}(n)

show(io::IO, b::Monomials{Float64}) = print(io, "Monomials($(length(b)))")
show(io::IO, b::Monomials{T}) where T = print(io, "Monomials{$(T)}($(length(b)))")

to_monomials_dict(dict::PolynomialBasis{T}) where T = Monomials{T}(length(dict))
to_monomials(f) = to_monomials(expansion(f))
to_monomials(f::Expansion) = to_monomials(dictionary(f), coefficients(f))
to_monomials(dict::Dictionary, coef) = conversion(dict, to_monomials_dict(dict)) * expansion(dict, coef)

support(dict::Monomials{T}) where {T} = DomainSets.FullSpace{T}()

size(dict::Monomials) = (dict.n,)

unsafe_eval_element(b::Monomials, idxn::PolynomialDegree, x) = x^degree(idxn)

function unsafe_eval_element_derivative(b::Monomials{T}, idxn::PolynomialDegree, x, order) where {T}
    @assert order > 0
    i = degree(idxn)
    if order > i
        zero(T)
    elseif order == 1
        one(T)*i*x^(i-1)
    else
        factorial(i) / T(factorial(i-order)) * x^(i-order)
    end
end

hasderivative(b::Monomials) = true

function derivative_dict(Φ::Monomials, order::Int; options...)
	n = length(Φ)
	@assert n-order >= 0
	similar(Φ, n-order)
end

function differentiation(::Type{T}, src::Monomials, dest::Monomials, order::Int; options...) where {T}
	n = length(src)
	m = length(dest)
	@assert m >= n-order
	if orderiszero(order)
		@assert src == dest
		IdentityOperator{T}(src)
	else
		A = zeros(T, m, n)
		for i in 1:n-order
			A[i,i+1] = i
		end
		op = ArrayOperator(A, src, dest)
		if order == 1
			op
		else
			differentiation(T, src, dest, order-1; options...) * op
		end
	end
end

similar(b::Monomials, ::Type{T}, n::Int) where {T} = Monomials{T}(n)

complex(d::Monomials{T}) where {T} = similar(d, complex(T))

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

function show(io::IO, p::Monomial)
	if degree(p) == 0
		print(io, "1 (monomial)")
	elseif degree(p) == 1
		print(io, "x (monomial)")
	else
		print(io, "x^$(degree(p)) (monomial)")
	end
end

convert(::Type{TypedFunction{T,T}}, p::Monomial) where {T} = Monomial{T}(p.degree)

(*)(p1::Monomial, p2::Monomial) = (*)(promote(p1,p2)...)
(*)(p1::Monomial{T}, p2::Monomial{T}) where {T} = Monomial{T}(degree(p1)+degree(p2))

basisfunction(dict::Monomials, idx) = basisfunction(dict, native_index(dict, idx))
basisfunction(dict::Monomials{T}, idx::PolynomialDegree) where {T} = Monomial{T}(degree(idx))

dictionary(p::Monomial{T}) where {T} = Monomials{T}(degree(p)+1)
index(p::Monomial) = degree(p)+1


# Integrals and moments

function integral(f::Monomial{T}, w::LegendreWeight) where T
	d = degree(f)
	(one(T)^(d+1) - (-one(T))^(d+1))/(d+1)
end

# line below could be more general, but we have to be careful about complex conjugation
dict_innerproduct_native(b1::Monomial, i, b2::Monomial, j, μ::LegendreWeight; options...) =
	integral(b1[i]*b2[j], μ)
