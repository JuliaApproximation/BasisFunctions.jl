
############################################
# Chebyshev polynomials of the first kind
############################################

"""
A basis of Chebyshev polynomials of the first kind on the interval `[-1,1]`.
"""
struct ChebyshevT{T} <: OPS{T,T}
    n			::	Int
end

name(b::ChebyshevT) = "Chebyshev series (first kind)"

ChebyshevT(n::Int) = ChebyshevT{Float64}(n)

# Convenience constructor: map the Chebyshev basis to the interval [a,b]
ChebyshevT{T}(n, a, b) where {T} = rescale(ChebyshevT{T}(n), a, b)

function ChebyshevT(n::Int, a::Number, b::Number)
    T = float(promote_type(typeof(a),typeof(b)))
    ChebyshevT{T}(n, a, b)
end

instantiate(::Type{ChebyshevT}, n, ::Type{T}) where {T} = ChebyshevT{T}(n)

similar(b::ChebyshevT, ::Type{T}, n::Int) where {T} = ChebyshevT{T}(n)


hasinterpolationgrid(b::ChebyshevT) = true
hasderivative(b::ChebyshevT) = true
hasantiderivative(b::ChebyshevT) = true

hasgrid_transform(b::ChebyshevT, gb, ::ChebyshevNodes) = length(b) == length(gb)
hasgrid_transform(b::ChebyshevT, gb, ::ChebyshevExtremae) = length(b) == length(gb)
hasgrid_transform(b::ChebyshevT, gb, ::AbstractGrid) = false


first_moment(b::ChebyshevT{T}) where {T} = convert(T, pi)

interpolation_grid(b::ChebyshevT{T}) where {T} = ChebyshevNodes{T}(length(b))
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
jacobi_α(b::ChebyshevT{T}) where {T} = -one(T)/2
jacobi_β(b::ChebyshevT{T}) where {T} = -one(T)/2



# See DLMF, Table 18.9.1
# http://dlmf.nist.gov/18.9#i
rec_An(b::ChebyshevT{T}, n::Int) where {T} = n==0 ? one(T) : 2one(T)

rec_Bn(b::ChebyshevT{T}, n::Int) where {T} = zero(T)

rec_Cn(b::ChebyshevT{T}, n::Int) where {T} = one(T)

support(b::ChebyshevT{T}) where {T} = ChebyshevInterval{T}()

# We can define this O(1) evaluation method, but only for points that are
# real and lie in [-1,1]
# Note that if x is not Real, recurrence_eval will be called by the OPS supertype
function unsafe_eval_element(b::ChebyshevT, idx::PolynomialDegree, x::Real)
    abs(x) <= 1 ? cos(degree(idx)*acos(x)) : recurrence_eval(b, idx, x)
end

# The version below is safe for points outside [-1,1] too.
# If we don't define anything, evaluation will default to using the three-term
# recurence relation.
# unsafe_eval_element{T <: Real}(b::ChebyshevT, idx::Int, x::T) = real(cos((idx-1)*acos(x+0im)))

function unsafe_eval_element_derivative(b::ChebyshevT, idx::PolynomialDegree, x)
    T = codomaintype(b)
    d = degree(idx)
    if d == 0
        T(0)
    else
        d * unsafe_eval_element(ChebyshevU(length(b)), idx-1, x)
    end
end

unsafe_moment(dict::ChebyshevT, idx; measure = lebesguemeasure(support(dict)), options...) =
	unsafe_moment(dict, idx, measure; options...)


unsafe_moment(dict::ChebyshevT, idx, measure; options...) =
	innerproduct(dict, idx, dict, PolynomialDegree(0), measure; options...)
function unsafe_moment(dict::ChebyshevT{T}, idx, ::LegendreMeasure; options...) where {T}
    n = degree(idx)
    if n == 0
        T(2)
    else
        isodd(n) ? zero(T) : -T(2)/((n+1)*(n-1))
    end
end


##################
# Differentiation
##################

# Chebyshev differentiation is so common that we make it its own type
struct ChebyshevDifferentiation{T} <: DictionaryOperator{T}
	src		::	Dictionary
	dest	::	Dictionary
    order   ::  Int
end

ChebyshevDifferentiation(src::Dictionary, dest::Dictionary, order::Int = 1) =
	ChebyshevDifferentiation{op_eltype(src,dest)}(src, dest, order)

ChebyshevDifferentiation(src::Dictionary, order::Int = 1) =
    ChebyshevDifferentiation(src, src, order)

order(op::ChebyshevDifferentiation) = op.order

string(op::ChebyshevDifferentiation) = "Chebyshev differentiation matrix of order $(order(op)) and size $(length(src(op)))"

similar_operator(op::ChebyshevDifferentiation, src, dest) =
    ChebyshevDifferentiation(src, dest, order(op))

wrap_operator(src, dest, op::ChebyshevDifferentiation) = similar_operator(op, src, dest)

# TODO: this allocates lots of memory...
function apply!(op::ChebyshevDifferentiation, coef_dest, coef_src)
    #	@assert period(dest)==period(src)
    n = length(coef_src)
    T = eltype(coef_src)
    tempc = coef_src[:]
    tempr = coef_src[:]
    for o = 1:order(op)
        tempr = zeros(T,n)
        # 'even' summation
        s = 0
        for i=(n-1):-2:2
            s = s+2*i*tempc[i+1]
            tempr[i] = s
        end
        # 'uneven' summation
        s = 0
        for i=(n-2):-2:2
            s = s+2*i*tempc[i+1]
            tempr[i] = s
        end
        # first row
        s = 0
        for i=2:2:n
            s = s+(i-1)*tempc[i]
        end
        tempr[1]=s
        tempc = tempr
    end
    coef_dest[1:n-order(op)] = tempr[1:n-order(op)]
    coef_dest
end

struct ChebyshevAntidifferentiation{T} <: DictionaryOperator{T}
	src		::	Dictionary
	dest	::	Dictionary
    order   ::  Int
end

ChebyshevAntidifferentiation(src::Dictionary, dest::Dictionary, order::Int = 1) =
	ChebyshevAntidifferentiation{op_eltype(src,dest)}(src, dest, order)

ChebyshevAntidifferentiation(src::Dictionary, order::Int = 1) =
    ChebyshevAntidifferentiation(src, src, order)

order(op::ChebyshevAntidifferentiation) = op.order

string(op::ChebyshevAntidifferentiation) = "Chebyshev antidifferentiation matrix of order $(order(op)) and size $(length(src(op)))"

similar_operator(op::ChebyshevAntidifferentiation, src, dest) =
    ChebyshevAntidifferentiation(src, dest, order(op))

wrap_operator(src, dest, op::ChebyshevAntidifferentiation) = similar_operator(op, src, dest)

# TODO: this allocates lots of memory...
function apply!(op::ChebyshevAntidifferentiation, coef_dest, coef_src)
    #	@assert period(dest)==period(src)
    T = eltype(coef_src)
    tempc = zeros(T,length(coef_dest))
    tempc[1:length(coef_src)] = coef_src[1:length(coef_src)]
    tempr = zeros(T,length(coef_dest))
    tempr[1:length(coef_src)] = coef_src[1:length(coef_src)]
    for o = 1:order(op)
        n = length(coef_src)+o
        tempr = zeros(T,n)
        tempr[2]+=tempc[1]
        tempr[3]=tempc[2]/4
        tempr[1]+=tempc[2]/4
        for i = 3:n-1
            tempr[i-1]-=tempc[i]/(2*(i-2))
            tempr[i+1]+=tempc[i]/(2*(i))
            tempr[1]+=real(-1im^i)*tempc[i]*(2*i-2)/(2*i*(i-2))
        end
        tempc = tempr
    end
    coef_dest[:]=tempr[:]
    coef_dest
end

differentiation_operator(src::ChebyshevT, dest::ChebyshevT, order::Int;
			T = op_eltype(src, dest), options...) =
    ChebyshevDifferentiation{T}(src, dest, order)

antidifferentiation_operator(src::ChebyshevT, dest::ChebyshevT, order::Int;
			T = op_eltype(src, dest), options...) =
    ChebyshevAntidifferentiation{T}(src, dest, order)


## Inner products

hasmeasure(dict::ChebyshevT) = true
measure(dict::ChebyshevT{T}) where T = ChebyshevMeasure{T}()

innerproduct_native(b1::ChebyshevT, i::PolynomialDegree, b2::ChebyshevT, j::PolynomialDegree, m::ChebyshevTMeasure;
			T = coefficienttype(b1), options...) =
	innerproduct_chebyshev_full(i, j, T)

function innerproduct_chebyshev_full(i, j, T)
	if i == j
		if i == PolynomialDegree(0)
			convert(T, pi)
		else
			convert(T, pi)/2
		end
	else
		zero(T)
	end
end

function gramoperator(dict::ChebyshevT; T = coefficienttype(dict), options...)
	diag = zeros(T, length(dict))
	fill!(diag, convert(T, pi)/2)
	diag[1] = convert(T,pi)
	DiagonalOperator(diag, src=dict)
end

function innerproduct_native(b1::ChebyshevT, i::PolynomialDegree, b2::ChebyshevT, j::PolynomialDegree, measure::LegendreMeasure; options...)
	n1 = degree(i)
	n2 = degree(j)
	(unsafe_moment(b1, PolynomialDegree(n1+n2), measure) + unsafe_moment(b1, PolynomialDegree(abs(n1-n2)), measure))/2
end

## Extension and restriction

function extension_operator(s1::ChebyshevT, s2::ChebyshevT; T = op_eltype(s1,s2), options...)
    @assert length(s2) >= length(s1)
    IndexExtensionOperator(s1, s2, 1:length(s1); T=T)
end

function restriction_operator(s1::ChebyshevT, s2::ChebyshevT; T = op_eltype(s1,s2), options...)
    @assert length(s2) <= length(s1)
    IndexRestrictionOperator(s1, s2, 1:length(s2); T=T)
end


###################################################################################
# Methods to transform from ChebyshevT to ChebyshevNodes and ChebyshevExtremae
###################################################################################

grid_evaluation_operator(dict::ChebyshevT, gb::GridBasis, grid::ChebyshevNodes; options...) =
	resize_and_transform(dict, gb, grid; chebyshevpoints = :nodes, options...)

grid_evaluation_operator(dict::ChebyshevT, gb::GridBasis, grid::ChebyshevExtremae; options...) =
	resize_and_transform(dict, gb, grid; chebyshevpoints = :extremae, options...)

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

function transform_to_grid(src::ChebyshevT, dest::GridBasis, grid::ChebyshevNodes;
			T = op_eltype(src, dest), options...)
	@assert length(src) == length(grid)
	chebyshev_transform_nodes(src, T; options...)
end

function transform_to_grid(src::ChebyshevT, dest::GridBasis, grid::ChebyshevExtremae;
			T = op_eltype(src, dest), options...)
	@assert length(src) == length(grid)
	chebyshev_transform_extremae(src, T; options...)
end

transform_from_grid(src::GridBasis, dest::ChebyshevT, grid; options...) =
	inv(transform_to_grid(dest, src, grid; options...))





iscompatible(src1::ChebyshevT, src2::ChebyshevT) = true

function (*)(src1::ChebyshevT, src2::ChebyshevT, coef_src1, coef_src2)
    @assert domaintype(src1) == domaintype(src2)
    T = promote_type(eltype(coef_src1), eltype(coef_src2))
    dest = ChebyshevT{T}(length(src1)+length(src2))
    coef_dest = zeros(dest)
    for i = 1:length(src1)
        for j = 1:length(src2)
            coef_dest[i+j-1] += one(T)/2*coef_src1[i]*coef_src2[j]
            coef_dest[abs(i-j)+1] += one(T)/2*coef_src1[i]*coef_src2[j]
        end
    end
    (dest,coef_dest)
end

############################################
# Chebyshev polynomials of the second kind
############################################

"A basis of Chebyshev polynomials of the second kind on the interval `[-1,1]`."
struct ChebyshevU{T} <: OPS{T,T}
    n			::	Int
end



ChebyshevU(n::Int) = ChebyshevU{Float64}(n)

instantiate(::Type{ChebyshevU}, n, ::Type{T}) where {T} = ChebyshevU{T}(n)

similar(b::ChebyshevU, ::Type{T}, n::Int) where {T} = ChebyshevU{T}(n)

name(b::ChebyshevU) = "Chebyshev series (second kind)"

function unsafe_eval_element(b::ChebyshevU, idx::PolynomialDegree, x::Real)
    # Don't use the formula when |x|=1, because it will generate NaN's
    d = degree(idx)
    abs(x) < 1 ? sin((d+1)*acos(x))/sqrt(1-x^2) : recurrence_eval(b, idx, x)
end

first_moment(b::ChebyshevU{T}) where {T} = convert(T, pi)/2

interpolation_grid(b::ChebyshevU{T}) where {T} = ChebyshevNodes{T}(b.n)

measure(dict::ChebyshevU{T}) where {T} = ChebyshevUMeasure{T}()

function innerproduct_native(b1::ChebyshevU, i::PolynomialDegree, b2::ChebyshevU, j::PolynomialDegree, m::ChebyshevUMeasure;
			T = coefficienttype(b1), options...)
	if i == j
		convert(T, pi)/2
	else
		zero(T)
	end
end

gramoperator(dict::ChebyshevU; T = coefficienttype(dict), options...) =
	ScalingOperator(dict, convert(T, pi)/2)

# Parameters alpha and beta of the corresponding Jacobi polynomial
jacobi_α(b::ChebyshevU{T}) where {T} = one(T)/2
jacobi_β(b::ChebyshevU{T}) where {T} = one(T)/2


# See DLMF, Table 18.9.1
# http://dlmf.nist.gov/18.9#i
rec_An(b::ChebyshevU{T}, n::Int) where {T} = convert(T, 2)

rec_Bn(b::ChebyshevU{T}, n::Int) where {T} = zero(T)

rec_Cn(b::ChebyshevU{T}, n::Int) where {T} = one(T)

support(b::ChebyshevU{T}) where {T} = ChebyshevInterval{T}()
