
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



name(b::ChebyshevBasis) = "Chebyshev series (first kind)"

ChebyshevBasis(n::Int) = ChebyshevBasis{Float64}(n)

# Convenience constructor: map the Chebyshev basis to the interval [a,b]
ChebyshevBasis{T}(n, a, b) where {T} = rescale(ChebyshevBasis{T}(n), a, b)

function ChebyshevBasis(n::Int, a::Number, b::Number)
    T = float(promote_type(typeof(a),typeof(b)))
    ChebyshevBasis{T}(n, a, b)
end

instantiate(::Type{ChebyshevBasis}, n, ::Type{T}) where {T} = ChebyshevBasis{T}(n)

similar(b::ChebyshevBasis, ::Type{T}, n::Int) where {T} = ChebyshevBasis{T}(n)


has_interpolationgrid(b::ChebyshevBasis) = true
has_derivative(b::ChebyshevBasis) = true
has_antiderivative(b::ChebyshevBasis) = true

has_grid_transform(b::ChebyshevBasis, gb, ::ChebyshevNodes) = length(b) == length(gb)
has_grid_transform(b::ChebyshevBasis, gb, ::ChebyshevExtremae) = length(b) == length(gb)
has_grid_transform(b::ChebyshevBasis, gb, ::AbstractGrid) = false


first_moment(b::ChebyshevBasis{T}) where {T} = convert(T, pi)

interpolation_grid(b::ChebyshevBasis{T}) where {T} = ChebyshevNodes{T}(length(b))
secondgrid(b::ChebyshevBasis{T}) where {T} = ChebyshevExtremae{T}(length(b))

# extends the default definition at transform.jl
transform_dict(s::ChebyshevBasis; nodegrid=true, options...) =
    nodegrid ? GridBasis(s) : GridBasis{coefficienttype(s)}(secondgrid(s))

# The weight function
weight(b::ChebyshevBasis{T}, x) where {T} = one(T)/sqrt(1-T(x)^2)

# Parameters alpha and beta of the corresponding Jacobi polynomial
jacobi_α(b::ChebyshevBasis{T}) where {T} = -one(T)/2
jacobi_β(b::ChebyshevBasis{T}) where {T} = -one(T)/2



# See DLMF, Table 18.9.1
# http://dlmf.nist.gov/18.9#i
rec_An(b::ChebyshevBasis{T}, n::Int) where {T} = n==0 ? one(T) : 2one(T)

rec_Bn(b::ChebyshevBasis{T}, n::Int) where {T} = zero(T)

rec_Cn(b::ChebyshevBasis{T}, n::Int) where {T} = one(T)

support(b::ChebyshevBasis{T}) where {T} = ChebyshevInterval{T}()

# We can define this O(1) evaluation method, but only for points that are
# real and lie in [-1,1]
# Note that if x is not Real, recurrence_eval will be called by the OPS supertype
function unsafe_eval_element(b::ChebyshevBasis, idx::PolynomialDegree, x::Real)
    abs(x) <= 1 ? cos(degree(idx)*acos(x)) : recurrence_eval(b, idx, x)
end

# The version below is safe for points outside [-1,1] too.
# If we don't define anything, evaluation will default to using the three-term
# recurence relation.
# unsafe_eval_element{T <: Real}(b::ChebyshevBasis, idx::Int, x::T) = real(cos((idx-1)*acos(x+0im)))

function unsafe_eval_element_derivative(b::ChebyshevBasis, idx::PolynomialDegree, x)
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

differentiation_operator(src::ChebyshevBasis, dest::ChebyshevBasis, order::Int; options...) =
    ChebyshevDifferentiation(src, dest, order)

antidifferentiation_operator(src::ChebyshevBasis, dest::ChebyshevBasis, order::Int; options...) =
    ChebyshevAntidifferentiation(src, dest, order)

function gramdiagonal!(result, ::ChebyshevBasis; options...)
    T = eltype(result)
    for i in 1:length(result)
        (i==1) ? result[i] = T(pi) : result[i] = T(pi)/2
    end
end

function UnNormalizedGram(s::ChebyshevBasis, oversampling)
    A = coefficienttype(s)
    d = A(length_oversampled_grid(s, oversampling))/2*ones(A,length(s))
    d[1] = length_oversampled_grid(s, oversampling)
    DiagonalOperator(s, s, d)
end

function extension_operator(s1::ChebyshevBasis, s2::ChebyshevBasis; T = op_eltype(s1,s2), options...)
    @assert length(s2) >= length(s1)
    IndexExtensionOperator{T}(s1, s2, 1:length(s1))
end

function restriction_operator(s1::ChebyshevBasis, s2::ChebyshevBasis; T = op_eltype(s1,s2), options...)
    @assert length(s2) <= length(s1)
    IndexRestrictionOperator{T}(s1, s2, 1:length(s2))
end


################################################################
# Methods to transform from ChebyshevBasis to ChebyshevNodes
###############################################################

transform_from_grid(src, dest::ChebyshevBasis, grid::ChebyshevNodes; T = coefficienttype(dest), options...) =
	_forward_chebyshev_operator(src, dest, T; options...)

transform_to_grid(src::ChebyshevBasis, dest, grid::ChebyshevNodes; T = coefficienttype(src), options...) =
	_backward_chebyshev_operator(src, dest, T; options...)

# These are the generic fallbacks
_forward_chebyshev_operator(src, dest, ::Type{T}; options...) where {T <: Number} =
	FastChebyshevTransform(src, dest; T = T)

_backward_chebyshev_operator(src, dest, ::Type{T}; options...) where {T <: Number} =
	InverseFastChebyshevTransform(src, dest; T=T)

# But for some types we use FFTW
for op in (:Float32, :Float64, :(Complex{Float32}), :(Complex{Float64}))
    @eval _forward_chebyshev_operator(src, dest, T::Type{$(op)}; options...) =
	   FastChebyshevTransformFFTW(src, dest; T = T, options...)
    @eval _backward_chebyshev_operator(src, dest, T::Type{$(op)}; options...) =
       	InverseFastChebyshevTransformFFTW(src, dest; T = T, options...)
end



function transform_to_grid_tensor(::Type{F}, ::Type{G}, s1, s2, grid; T = coefficienttype(s1), options...) where {F <: ChebyshevBasis,G <: ChebyshevNodes}
	_backward_chebyshev_operator(s1, s2, T; options...)
end

function transform_from_grid_tensor(::Type{F}, ::Type{G}, s1, s2, grid; T = coefficienttype(s2), options...) where {F <: ChebyshevBasis,G <: ChebyshevNodes}
	_forward_chebyshev_operator(s1, s2, T; options...)
end

const AlternatingSignOperator{T} = DiagonalOperator{T,AlternatingSigns{T}}

AlternatingSignOperator(src) = AlternatingSignOperator{coefficienttype(src)}(src)
AlternatingSignOperator{T}(src) where {T} = AlternatingSignOperator{T}(src, src, Diagonal(AlternatingSigns{T}(length(src))))

string(op::AlternatingSignOperator) = "Alternating sign operator of length $(size(op,1))"


const CoefficientScalingOperator{T} = DiagonalOperator{T,ScaledEntry{T}}

CoefficientScalingOperator(src::Dictionary, index::Int, scalar) =
	CoefficientScalingOperator{coefficienttype(src)}(src, index, scalar)
CoefficientScalingOperator(src::Dictionary, dest::Dictionary, index::Int, scalar) =
	CoefficientScalingOperator{op_eltype(src,dest)}(src, index, scalar)
CoefficientScalingOperator{T}(src::Dictionary, index::Int, scalar) where {T} =
	CoefficientScalingOperator{T}(src, src, Diagonal(ScaledEntry{T}(length(src), index, scalar)))

string(op::CoefficientScalingOperator) = "Scaling of coefficient $(op.index) by $(op.scalar)"

function transform_from_grid_post(src, dest::ChebyshevBasis, grid::ChebyshevNodes;
			T = coefficienttype(dest), options...)
    scaling = ScalingOperator{T}(dest, 1/sqrt(convert(T, length(dest))/2))
    coefscaling = CoefficientScalingOperator{T}(dest, 1, 1/sqrt(convert(T, 2)))
    flip = AlternatingSignOperator{T}(dest)
	scaling * coefscaling * flip
end

transform_to_grid_pre(src::ChebyshevBasis, dest, grid::ChebyshevNodes; options...) =
    inv(transform_from_grid_post(dest, src, grid; options...))


##################################################################
# Methods to transform from ChebyshevBasis to ChebyshevExtremae
##################################################################

transform_from_grid(src, dest::ChebyshevBasis, grid::ChebyshevExtremae;
			T = coefficienttype(dest), options...) =
		_chebyshevI_operator(src, dest, T; options...)

transform_to_grid(src::ChebyshevBasis, dest, grid::ChebyshevExtremae;
			T = coefficienttype(src), options...) =
	_chebyshevI_operator(src, dest, T; options...)

# These are the generic fallbacks
_chebyshevI_operator(src, dest, ::Type{T}; options...) where {T <: Number} =
    FastChebyshevITransform(src, dest)

# But for some types we use FFTW
for op in (:Float32, :Float64, :(Complex{Float32}), :(Complex{Float64}))
	@eval _chebyshevI_operator(src, dest, ::Type{$(op)}; options...) =
		FastChebyshevITransformFFTW(src, dest; options...)
end

function transform_to_grid_tensor(::Type{F}, ::Type{G}, s1, s2, grid; T = eltype(s1,s2), options...) where {F <: ChebyshevBasis,G <: ChebyshevExtremae}
    _chebyshevI_operator(s1, s2, T; options...)
end

function transform_from_grid_tensor(::Type{F}, ::Type{G}, s1, s2, grid; T = eltype(s1,s2), options...) where {F <: ChebyshevBasis,G <: ChebyshevExtremae}
    _chebyshevI_operator(s1, s2, T; options...)
end

function transform_to_grid_pre(src::ChebyshevBasis, dest, grid::ChebyshevExtremae;
			T = coefficienttype(src), options...)
    coefscaling1 = CoefficientScalingOperator{T}(src, 1, 2)
    coefscaling2 = CoefficientScalingOperator{T}(src, length(src), 2)
    coefscaling1 * coefscaling2
end

function transform_to_grid_post(src::ChebyshevBasis, dest, grid::ChebyshevExtremae;
			T = coefficienttype(src), options...)
	ScalingOperator{T}(dest, one(T)/2)
end

function transform_from_grid_post(src, dest::ChebyshevBasis, grid::ChebyshevExtremae;
			T = coefficienttype(dest), options...)
    # Inverse DCT is unnormalized, applying DCT and its inverse gives N times the original. N=2(length-1)
    scaling = ScalingOperator(dest, 1/(2*T(length(dest)-1)))
    scaling * inv(transform_to_grid_pre(dest, src, grid; T = T, options...))
end

transform_from_grid_pre(src, dest::ChebyshevBasis, grid::ChebyshevExtremae; options...) =
    inv(transform_to_grid_post(dest, src, grid; options...))



is_compatible(src1::ChebyshevBasis, src2::ChebyshevBasis) = true

function (*)(src1::ChebyshevBasis, src2::ChebyshevBasis, coef_src1, coef_src2)
    @assert domaintype(src1) == domaintype(src2)
    T = promote_type(eltype(coef_src1), eltype(coef_src2))
    dest = ChebyshevBasis{T}(length(src1)+length(src2))
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


Gram(s::ChebyshevU; options...) = ScalingOperator(s, s, coefficienttype(s)(pi)/2)

# The weight function
weight(b::ChebyshevU{T}, x) where {T} = sqrt(1-convert(T, x)^2)

# Parameters alpha and beta of the corresponding Jacobi polynomial
jacobi_α(b::ChebyshevU{T}) where {T} = one(T)/2
jacobi_β(b::ChebyshevU{T}) where {T} = one(T)/2


# See DLMF, Table 18.9.1
# http://dlmf.nist.gov/18.9#i
rec_An(b::ChebyshevU{T}, n::Int) where {T} = convert(T, 2)

rec_Bn(b::ChebyshevU{T}, n::Int) where {T} = zero(T)

rec_Cn(b::ChebyshevU{T}, n::Int) where {T} = one(T)

support(b::ChebyshevU{T}) where {T} = ChebyshevInterval{T}()
