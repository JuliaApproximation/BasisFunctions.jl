# chebyshev.jl


############################################
# Chebyshev polynomials of the first kind
############################################


"""
A basis of Chebyshev polynomials of the first kind on the interval `[-1,1]`.
"""
struct ChebyshevBasis{T} <: OPS{T,T}
    n			::	Int
end

ChebyshevT = ChebyshevBasis
## Most of these methods apply to Chebyshev-like dictionaries as well
ChebyshevTLike = Union{ChebyshevBasis, ComplexifiedDict{D} where D<:ChebyshevBasis}



name(b::ChebyshevBasis) = "Chebyshev series (first kind)"

ChebyshevBasis(n::Int) = ChebyshevBasis{Float64}(n)

# Convenience constructor: map the Chebyshev basis to the interval [a,b]
ChebyshevBasis{T}(n, a, b) where {T} = rescale(ChebyshevBasis{T}(n), a, b)

function ChebyshevBasis(n::Int, a::Number, b::Number)
    T = float(promote_type(typeof(a),typeof(b)))
    ChebyshevBasis{T}(n, a, b)
end

instantiate(::Type{ChebyshevBasis}, n, ::Type{T}) where {T} = ChebyshevBasis{T}(n)

dict_promote_domaintype(b::ChebyshevBasis{T}, ::Type{S}) where {T,S} = ChebyshevBasis{promote_type(S,T)}(b.n)
dict_promote_coeftype(b::ChebyshevBasis{T}, ::Type{S}) where {T,S<:Real} = ChebyshevBasis{promote_type(S,T)}(b.n)

resize(b::ChebyshevBasis{T}, n) where {T} = ChebyshevBasis{T}(n)



has_grid(b::ChebyshevTLike) = true
has_derivative(b::ChebyshevTLike) = true
has_antiderivative(b::ChebyshevTLike) = true

has_grid_transform(b::ChebyshevTLike, gb, ::ChebyshevNodeGrid) = length(b) == length(gb)
has_grid_transform(b::ChebyshevTLike, gb, ::ChebyshevExtremaGrid) = length(b) == length(gb)
has_grid_transform(b::ChebyshevTLike, gb, ::AbstractGrid) = false


first_moment(b::ChebyshevBasis{T}) where {T} = T(pi)

grid(b::ChebyshevTLike) = ChebyshevNodeGrid(length(b), domaintype(b))
secondgrid(b::ChebyshevTLike) = ChebyshevExtremaGrid(length(b), domaintype(b))

# extends the default definition at transform.jl
transform_dict(s::ChebyshevTLike; nodegrid=true, options...) =
    nodegrid ? gridbasis(s) : gridbasis(secondgrid(s), coeftype(s))

# The weight function
weight(b::ChebyshevBasis{T}, x) where {T} = 1/sqrt(1-T(x)^2)

# Parameters alpha and beta of the corresponding Jacobi polynomial
jacobi_α(b::ChebyshevTLike) = -1//2
jacobi_β(b::ChebyshevTLike) = -1//2



# See DLMF, Table 18.9.1
# http://dlmf.nist.gov/18.9#i
rec_An(b::ChebyshevTLike, n::Int) = n==0 ? 1 : 2

rec_Bn(b::ChebyshevTLike, n::Int) = 0

rec_Cn(b::ChebyshevTLike, n::Int) = 1

support(b::ChebyshevBasis{T}) where {T} = ChebyshevInterval{T}()

# We can define this O(1) evaluation method, but only for points that are
# real and lie in [-1,1]
# Note that if x is not Real, recurrence_eval will be called by the OPS supertype
function unsafe_eval_element(b::ChebyshevTLike, idx::PolynomialDegree, x::Real)
    abs(x) <= 1 ? cos(degree(idx)*acos(x)) : recurrence_eval(b, idx, x)
end

# The version below is safe for points outside [-1,1] too.
# If we don't define anything, evaluation will default to using the three-term
# recurence relation.
# unsafe_eval_element{T <: Real}(b::ChebyshevBasis, idx::Int, x::T) = real(cos((idx-1)*acos(x+0im)))

function unsafe_eval_element_derivative(b::ChebyshevTLike, idx::PolynomialDegree, x)
    T = codomaintype(b)
    d = degree(idx)
    if d == 0
        T(0)
    else
        d * unsafe_eval_element(ChebyshevU(length(b)), idx-1, x)
    end
end

function unsafe_moment(b::ChebyshevBasis{T}, idx::PolynomialDegree) where {T}
    d = degree(idx)
    if d == 0
        T(2)
    else
        isodd(d) ? zero(T) : -T(2)/((d+1)*(d-1))
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

differentiation_operator(src::ChebyshevTLike, dest::ChebyshevTLike, order::Int; options...) =
    ChebyshevDifferentiation(src, dest, order)

antidifferentiation_operator(src::ChebyshevTLike, dest::ChebyshevTLike, order::Int; options...) =
    ChebyshevAntidifferentiation(src, dest, order)

function gramdiagonal!(result, ::ChebyshevTLike; options...)
    T = eltype(result)
    for i in 1:length(result)
        (i==1) ? result[i] = T(pi) : result[i] = T(pi)/2
    end
end

function UnNormalizedGram(s::ChebyshevTLike, oversampling)
    A = coeftype(s)
    d = A(length_oversampled_grid(s, oversampling))/2*ones(A,length(s))
    d[1] = length_oversampled_grid(s, oversampling)
    DiagonalOperator(s, s, d)
end

function extension_operator(s1::ChebyshevTLike, s2::ChebyshevTLike; options...)
    @assert length(s2) >= length(s1)
    IndexExtensionOperator(s1, s2, 1:length(s1))
end

function restriction_operator(s1::ChebyshevTLike, s2::ChebyshevTLike; options...)
    @assert length(s2) <= length(s1)
    IndexRestrictionOperator(s1, s2, 1:length(s2))
end


################################################################
# Methods to transform from ChebyshevTLike to ChebyshevNodeGrid
###############################################################

transform_from_grid(src, dest::ChebyshevTLike, grid::ChebyshevNodeGrid; options...) =
	_forward_chebyshev_operator(src, dest, coeftype(dest); options...)

transform_to_grid(src::ChebyshevTLike, dest, grid::ChebyshevNodeGrid; options...) =
	_backward_chebyshev_operator(src, dest, coeftype(src); options...)

# These are the generic fallbacks
_forward_chebyshev_operator(src, dest, ::Type{T}; options...) where {T <: Number} =
	FastChebyshevTransform(src, dest)

_backward_chebyshev_operator(src, dest, ::Type{T}; options...) where {T <: Number} =
	InverseFastChebyshevTransform(src, dest)

# But for some types we use FFTW
for op in (:Float32, :Float64, :(Complex{Float32}), :(Complex{Float64}))
    @eval _forward_chebyshev_operator(src, dest, ::Type{$(op)}; options...) =
	   FastChebyshevTransformFFTW(src, dest; options...)
    @eval _backward_chebyshev_operator(src, dest, ::Type{$(op)}; options...) =
       	InverseFastChebyshevTransformFFTW(src, dest; options...)
end



function transform_to_grid_tensor(::Type{F}, ::Type{G}, s1, s2, grid; options...) where {F <: ChebyshevTLike,G <: ChebyshevNodeGrid}
	_backward_chebyshev_operator(s1, s2, coeftype(s1); options...)
end

function transform_from_grid_tensor(::Type{F}, ::Type{G}, s1, s2, grid; options...) where {F <: ChebyshevTLike,G <: ChebyshevNodeGrid}
	_forward_chebyshev_operator(s1, s2, coeftype(s2); options...)
end



function transform_from_grid_post(src, dest::ChebyshevTLike, grid::ChebyshevNodeGrid; options...)
    ELT = coeftype(dest)
    scaling = ScalingOperator(dest, 1/sqrt(ELT(length(dest)/2)))
    coefscaling = CoefficientScalingOperator(dest, 1, 1/sqrt(ELT(2)))
    flip = UnevenSignFlipOperator(dest)
	scaling * coefscaling * flip
end

transform_to_grid_pre(src::ChebyshevTLike, dest, grid::ChebyshevNodeGrid; options...) =
    inv(transform_from_grid_post(dest, src, grid; options...))


##################################################################
# Methods to transform from ChebyshevTLike to ChebyshevExtremaGrid
##################################################################

transform_from_grid(src, dest::ChebyshevTLike, grid::ChebyshevExtremaGrid; options...) =
_chebyshevI_operator(src, dest, coeftype(dest); options...)

transform_to_grid(src::ChebyshevTLike, dest, grid::ChebyshevExtremaGrid; options...) =
_chebyshevI_operator(src, dest, coeftype(src); options...)

# These are the generic fallbacks
_chebyshevI_operator(src, dest, ::Type{T}; options...) where {T <: Number} =
    FastChebyshevITransform(src, dest)

# But for some types we use FFTW
for op in (:Float32, :Float64, :(Complex{Float32}), :(Complex{Float64}))
  @eval _chebyshevI_operator(src, dest, ::Type{$(op)}; options...) =
   FastChebyshevITransformFFTW(src, dest; options...)
end

function transform_to_grid_tensor(::Type{F}, ::Type{G}, s1, s2, grid; options...) where {F <: ChebyshevTLike,G <: ChebyshevExtremaGrid}
    _chebyshevI_operator(s1, s2, eltype(s1, s2); options...)
end

function transform_from_grid_tensor(::Type{F}, ::Type{G}, s1, s2, grid; options...) where {F <: ChebyshevTLike,G <: ChebyshevExtremaGrid}
    _chebyshevI_operator(s1, s2, eltype(s1, s2); options...)
end

function transform_to_grid_pre(src::ChebyshevTLike, dest, grid::ChebyshevExtremaGrid; options...)
    T = coeftype(src)
    coefscaling1 = CoefficientScalingOperator(src, 1, T(2))
    coefscaling2 = CoefficientScalingOperator(src, length(src), T(2))
    coefscaling1 * coefscaling2
end

function transform_to_grid_post(src::ChebyshevTLike, dest, grid::ChebyshevExtremaGrid; options...)
  ELT = coeftype(src)
  ScalingOperator(dest, 1/ELT(2))
end

function transform_from_grid_post(src, dest::ChebyshevTLike, grid::ChebyshevExtremaGrid; options...)
    # Inverse DCT is unnormalized, applying DCT and its inverse gives N times the original. N=2(length-1)
    ELT = coeftype(dest)
    scaling = ScalingOperator(dest, 1/(2*ELT(length(dest)-1)))
    scaling * inv(transform_to_grid_pre(dest, src, grid; options...))
end

transform_from_grid_pre(src, dest::ChebyshevTLike, grid::ChebyshevExtremaGrid; options...) =
    inv(transform_to_grid_post(dest, src, grid; options...))






Is_compatible(src1::ChebyshevTLike, src2::ChebyshevTLike) = true

function (*)(src1::ChebyshevBasis, src2::ChebyshevBasis, coef_src1, coef_src2)
    @assert domaintype(src1) == domaintype(src2)
    T = promote_type(eltype(coef_src1), eltype(coef_src2))
    dest = ChebyshevBasis(length(src1)+length(src2), domaintype(src1))
    coef_dest = zeros(T, length(dest))
    for i = 1:length(src1)
        for j = 1:length(src2)
            coef_dest[i+j-1] += 1/2*coef_src1[i]*coef_src2[j]
            coef_dest[abs(i-j)+1] += 1/2*coef_src1[i]*coef_src2[j]
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

dict_promote_domaintype(b::ChebyshevU, ::Type{S}) where {S} =
    ChebyshevU{S}(b.n)

resize(b::ChebyshevU{T}, n) where {T} = ChebyshevU{T}(n)

name(b::ChebyshevU) = "Chebyshev series (second kind)"

function unsafe_eval_element(b::ChebyshevU, idx::PolynomialDegree, x::Real)
    # Don't use the formula when |x|=1, because it will generate NaN's
    d = degree(idx)
    abs(x) < 1 ? sin((d+1)*acos(x))/sqrt(1-x^2) : recurrence_eval(b, idx, x)
end

first_moment(b::ChebyshevU{T}) where {T} = T(pi)/2

grid(b::ChebyshevU{T}) where {T} = ChebyshevNodeGrid{T}(b.n)


Gram(s::ChebyshevU; options...) = ScalingOperator(s, s, coeftype(s)(pi)/2)

# The weight function
weight(b::ChebyshevU{T}, x) where {T} = sqrt(1-T(x)^2)

# Parameters alpha and beta of the corresponding Jacobi polynomial
jacobi_α(b::ChebyshevU) = 1//2
jacobi_β(b::ChebyshevU) = 1//2


# See DLMF, Table 18.9.1
# http://dlmf.nist.gov/18.9#i
rec_An(b::ChebyshevU, n::Int) = 2

rec_Bn(b::ChebyshevU, n::Int) = 0

rec_Cn(b::ChebyshevU, n::Int) = 1

support(b::ChebyshevU{T}) where {T} = ChebyshevInterval{T}()
