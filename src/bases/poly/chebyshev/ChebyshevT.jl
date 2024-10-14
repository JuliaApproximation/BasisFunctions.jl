
"""
A basis of Chebyshev polynomials of the first kind on the interval `[-1,1]`.
"""
struct ChebyshevT{T} <: IntervalOPS{T}
    n			::	Int
end

show(io::IO, d::ChebyshevT{Float64}) = print(io, "ChebyshevT($(length(d)))")
show(io::IO, d::ChebyshevT{T}) where T = print(io, "ChebyshevT{$(T)}($(length(d)))")

ChebyshevT(n::Int) = ChebyshevT{Float64}(n)

ChebyshevT(d::PolynomialDegree) = ChebyshevT(value(d)+1)
ChebyshevT{T}(d::PolynomialDegree) where T = ChebyshevT{T}(value(d)+1)

similar(b::ChebyshevT, ::Type{T}, n::Int) where {T} = ChebyshevT{T}(n)
isequaldict(b1::ChebyshevT, b2::ChebyshevT) = length(b1)==length(b2)

to_chebyshev_dict(dict::Dictionary{T,T}) where T = ChebyshevT{T}(length(dict))
to_chebyshev(f) = to_chebyshev(expansion(f))
to_chebyshev(f::Expansion) = to_chebyshev(dictionary(f), coefficients(f))
to_chebyshev(dict::Dictionary, coef) =
	conversion(dict, to_chebyshev_dict(dict)) * expansion(dict, coef)

convert(::Type{ChebyshevT{T}}, d::ChebyshevT) where {T} = ChebyshevT{T}(d.n)

"Mapped Chebyshev polynomials."
struct MappedChebyshev{T} <: MappedDict{T,T}
	superdict	::	ChebyshevT{T}
	map			::	ScalarAffineMap{T}
end

mapped_dict(dict::ChebyshevT{T}, map::ScalarAffineMap{S}) where {S,T} =
	MappedChebyshev{promote_type(S,T)}(dict, map)

hasinterpolationgrid(b::ChebyshevT) = true
hasderivative(b::ChebyshevT) = true
hasderivative(b::ChebyshevT, order::Int) = true
hasantiderivative(b::ChebyshevT) = true

hasgrid_transform(b::ChebyshevT, gb, ::ChebyshevNodes) = length(b) == length(gb)
hasgrid_transform(b::ChebyshevT, gb, ::ChebyshevExtremae) = length(b) == length(gb)
hasgrid_transform(b::ChebyshevT, gb, ::AbstractGrid) = false


first_moment(b::ChebyshevT{T}) where {T} = convert(T, pi)

interpolation_grid(b::ChebyshevT{T}) where {T} = ChebyshevNodes{T}(length(b))
iscompatiblegrid(dict::ChebyshevT, grid::ChebyshevNodes) = length(dict) == length(grid)
secondgrid(b::ChebyshevT{T}) where {T} = ChebyshevExtremae{T}(length(b))
transformgrid_extremae(b::ChebyshevT{T}) where {T} = ChebyshevExtremae{T}(length(b))

# extends the default definition at transform.jl
function transform_dict(s::ChebyshevT{T}; chebyshevpoints=:nodes, options...) where {T}
	if chebyshevpoints == :nodes
    	GridBasis(s)
	else
		@assert chebyshevpoints == :extremae
		GridBasis{T}(secondgrid(s))
	end
end

# Parameters alpha and beta of the corresponding Jacobi polynomial
jacobi_α(b::ChebyshevT{T}) where T = -one(T)/2
jacobi_β(b::ChebyshevT{T}) where T = -one(T)/2

# Recurrence relation
rec_An(b::ChebyshevT{T}, n::Int) where T = chebyshevt_rec_An(n, T)
rec_Bn(b::ChebyshevT{T}, n::Int) where T = chebyshevt_rec_Bn(n, T)
rec_Cn(b::ChebyshevT{T}, n::Int) where T = chebyshevt_rec_Cn(n, T)

# We can define this O(1) evaluation method, but only for points that are
# real and lie in [-1,1]
# Note that if x is not Real, recurrence_eval will be called by the OPS supertype
function unsafe_eval_element(b::ChebyshevT, idx::PolynomialDegree, x::Real)
    abs(x) <= 1 ? chebyshev_eval(degree(idx), x) : recurrence_eval(b, idx, x)
end

"Evaluate the Chebyshev polynomial of degree `k` in the point `x ∈ [-1,1]`."
chebyshev_eval(k, x) = cos(k*acos(x))

# The version below is safe for points outside [-1,1] too.
# If we don't define anything, evaluation will default to using the three-term
# recurence relation.
# unsafe_eval_element{T <: Real}(b::ChebyshevT, idx::Int, x::T) = real(cos((idx-1)*acos(x+0im)))


unsafe_moment(dict::ChebyshevT, idx; measure = lebesguemeasure(support(dict)), options...) =
	unsafe_moment(dict, idx, measure; options...)


unsafe_moment(dict::ChebyshevT, idx, measure; options...) =
	dict_innerproduct(dict, idx, dict, PolynomialDegree(0), measure; options...)

function unsafe_moment(dict::ChebyshevT{T}, idx, ::Union{LegendreWeight,Lebesgue}; options...) where {T}
    n = degree(idx)
    if n == 0
        T(2)
    else
        isodd(n) ? zero(T) : -T(2)/((n+1)*(n-1))
    end
end

hasx(b::ChebyshevT) = length(b) >= 2
coefficients_of_x(b::ChebyshevT) = (c=zeros(b); c[2]=1; c)

## Inner products

hasmeasure(dict::ChebyshevT) = true
measure(dict::ChebyshevT{T}) where T = ChebyshevTWeight{T}()
issymmetric(::ChebyshevT) = true

function dict_innerproduct_native(b1::ChebyshevT, i::PolynomialDegree,
		b2::ChebyshevT, j::PolynomialDegree, m::ChebyshevTWeight; options...)
	T = promote_type(domaintype(b1),domaintype(b2))
	innerproduct_chebyshev_full(i, j, T)
end

innerproduct_chebyshev_full(i, j, T) =
	i == j ? chebyshevt_hn(value(i), T) : zero(T)

function dict_innerproduct_native(b1::ChebyshevT, i::PolynomialDegree,
		b2::ChebyshevT, j::PolynomialDegree, measure::Union{LegendreWeight,Lebesgue}; options...)
	n1 = degree(i)
	n2 = degree(j)
	(unsafe_moment(b1, PolynomialDegree(n1+n2), measure) + unsafe_moment(b1, PolynomialDegree(abs(n1-n2)), measure))/2
end



###################################################################################
# Methods to transform from ChebyshevT to ChebyshevNodes and ChebyshevExtremae
###################################################################################

evaluation(::Type{T}, dict::ChebyshevT, gb::GridBasis, grid::ChebyshevNodes; options...) where {T} =
	resize_and_transform(T, dict, gb, grid; chebyshevpoints = :nodes, options...)

evaluation(::Type{T}, dict::ChebyshevT, gb::GridBasis, grid::ChebyshevExtremae; options...) where {T} =
	resize_and_transform(T, dict, gb, grid; chebyshevpoints = :extremae, options...)

function chebyshev_transform_nodes(dict::ChebyshevT, T; options...)
	grid = interpolation_grid(dict)
	n = length(dict)
	# We compose the inverse DCT with a diagonal scaling
	F = inverse_chebyshev_operator(dict, GridBasis{T}(grid), T; options...)
	d = zeros(T, n)
	fill!(d, sqrt(T(n)/2))
	d[1] *= sqrt(T(2))
	for i in 1:length(d)
		d[i] *= (-1)^(i-1)
	end
	F * DiagonalOperator(dict, d)
end

function chebyshev_transform_extremae(dict::ChebyshevT, T; options...)
	grid = transformgrid_extremae(dict)
	F = inverse_chebyshevI_operator(dict, GridBasis{T}(grid), T; options...)
	d = zeros(T, length(dict))
	fill!(d, one(T)/2)
	d[1] *= 2
	d[end] *= 2
	F * DiagonalOperator(dict, d)
end

function transform_to_grid(T, src::ChebyshevT, dest::GridBasis, grid::ChebyshevNodes; options...)
	@assert length(src) == length(grid)
	chebyshev_transform_nodes(src, T; options...)
end

function transform_to_grid(T, src::ChebyshevT, dest::GridBasis, grid::ChebyshevExtremae; options...)
	@assert length(src) == length(grid)
	chebyshev_transform_extremae(src, T; options...)
end

transform_from_grid(T, src::GridBasis, dest::ChebyshevT, grid; options...) =
	inv(transform_to_grid(T, dest, src, grid; options...))

function expansion_roots(dict::ChebyshevT, coefficients::AbstractVector)
	@assert length(dict) == length(coefficients)
	T = eltype(coefficients)
	n = length(dict)-1
	# construct the colleague matrix (according to ATAP)
	C = zeros(T, n, n)
	C[1,2] = 1
	for i in 1:n-1
		C[i+1,i] = one(T)/2
		if i < n-1
			C[i+1,i+2] = one(T)/2
		end
	end
	for i in 1:n
		C[n,i] -= coefficients[i]/(2coefficients[end])
	end
	eigvals(C)
end

iscompatible(src1::ChebyshevT, src2::ChebyshevT) = true

function expansion_multiply(src1::ChebyshevT, src2::ChebyshevT, coef_src1, coef_src2)
    @assert domaintype(src1) == domaintype(src2)
    T = promote_type(eltype(coef_src1), eltype(coef_src2))
    dest = ChebyshevT{T}(length(src1)+length(src2)-1)
    coef_dest = zeros(dest)
    for i = 1:length(src1)
        for j = 1:length(src2)
            coef_dest[i+j-1] += one(T)/2*coef_src1[i]*coef_src2[j]
            coef_dest[abs(i-j)+1] += one(T)/2*coef_src1[i]*coef_src2[j]
        end
    end
    dest, coef_dest
end

"A Chebyshev polynomial of the first kind."
struct ChebyshevTPolynomial{T} <: OrthogonalPolynomial{T}
    degree  ::  Int
end

const ChebyshevPolynomial = ChebyshevTPolynomial

ChebyshevTPolynomial(args...; options...) = ChebyshevTPolynomial{Float64}(args...; options...)
ChebyshevTPolynomial{T}(p::ChebyshevTPolynomial) where {T} = ChebyshevTPolynomial{T}(p.degree)

ChebyshevTPolynomial{T}(; degree) where {T} = ChebyshevTPolynomial(degree)

show(io::IO, p::ChebyshevTPolynomial{Float64}) = print(io, "ChebyshevTPolynomial($(p.degree))")

convert(::Type{TypedFunction{T,T}}, p::ChebyshevTPolynomial) where {T} = ChebyshevTPolynomial{T}(p.degree)

basisfunction(dict::ChebyshevT, idx) = basisfunction(dict, native_index(dict, idx))
basisfunction(dict::ChebyshevT{T}, idx::PolynomialDegree) where {T} = ChebyshevTPolynomial{T}(degree(idx))

dictionary(p::ChebyshevTPolynomial{T}) where {T} = ChebyshevT{T}(degree(p)+1)
index(p::ChebyshevTPolynomial) = degree(p)+1
