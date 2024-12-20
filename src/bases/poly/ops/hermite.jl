
"""
A basis of the classical Hermite polynomials. These polynomials are orthogonal
on the real line `(-∞,∞)` with respect to the weight function
`w(x)=exp(-x^2)`.
"""
struct Hermite{T} <: OPS{T}
    n           ::  Int
end

Hermite(n::Int) = Hermite{Float64}(n)

Hermite(d::PolynomialDegree) = Hermite(value(d)+1)
Hermite{T}(d::PolynomialDegree) where T = Hermite{T}(value(d)+1)

similar(b::Hermite, ::Type{T}, n::Int) where {T} = Hermite{T}(n)
isequaldict(b1::Hermite, b2::Hermite) = length(b1)==length(b2)

support(b::Hermite{T}) where {T} = DomainSets.FullSpace(T)

first_moment(b::Hermite{T}) where {T} = sqrt(T(pi))

measure(b::Hermite{T}) where {T} = HermiteWeight{T}()
interpolation_grid(b::Hermite{T}) where T = HermiteNodes{T}(length(b))
iscompatiblegrid(b::Hermite,grid::HermiteNodes) = length(b) == length(grid)
isorthogonal(dict::Hermite, measure::GaussHermite) = opsorthogonal(dict, measure)
isorthogonal(::Hermite, ::HermiteWeight) = true
issymmetric(::Hermite) = true

gauss_rule(dict::Hermite{T}) where T = GaussHermite{T}(length(dict))

# recurrence relation
rec_An(b::Hermite{T}, n::Int) where T = hermite_rec_An(n, T)
rec_Bn(b::Hermite{T}, n::Int) where T = hermite_rec_Bn(n, T)
rec_Cn(b::Hermite{T}, n::Int) where T = hermite_rec_Cn(n, T)

coefficients_of_x(b::Hermite{T}) where {T} = (c=zeros(b); c[2]=one(T)/2; c)

same_ops_family(b1::Hermite, b2::Hermite) = true

dict_innerproduct_native(d1::Hermite{T}, i::PolynomialDegree, d2::Hermite, j::PolynomialDegree, measure::HermiteWeight; options...) where T =
	i == j ? hermite_hn(value(i), T) : zero(T)

show(io::IO, b::Hermite{Float64}) = print(io, "Hermite($(length(b)))")
show(io::IO, b::Hermite{T}) where T = print(io, "Hermite{$(T)}($(length(b)))")
