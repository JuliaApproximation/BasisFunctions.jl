# chebyshev.jl


############################################
# Chebyshev polynomials of the first kind
############################################


"""
A basis of Chebyshev polynomials of the first kind on the interval [-1,1].
"""
struct ChebyshevBasis{T} <: OPS{T}
    n			::	Int
end

ChebyshevT{T} = ChebyshevBasis{T}

const ChebyshevSpan{A,F<:ChebyshevBasis} = Span{A,F}

name(b::ChebyshevBasis) = "Chebyshev series (first kind)"


ChebyshevBasis(n, ::Type{T} = Float64) where {T} = ChebyshevBasis{T}(n)

ChebyshevBasis(n, a, b, ::Type{T} = promote_type(typeof(a),typeof(b))) where {T} =
    rescale( ChebyshevBasis(n,float(T)), a, b)

instantiate{T}(::Type{ChebyshevBasis}, n, ::Type{T}) = ChebyshevBasis{T}(n)

set_promote_domaintype(b::ChebyshevBasis, ::Type{S}) where {S} = ChebyshevBasis{S}(b.n)

resize(b::ChebyshevBasis, n) = ChebyshevBasis(n, domaintype(b))

has_grid(b::ChebyshevBasis) = true
has_derivative(b::ChebyshevBasis) = true
has_antiderivative(b::ChebyshevBasis) = true

has_grid_transform(b::ChebyshevBasis, gs, ::ChebyshevNodeGrid) = length(b) == length(gs)
has_grid_transform(b::ChebyshevBasis, gs, ::ChebyshevExtremaGrid) = length(b) == length(gs)
has_grid_transform(b::ChebyshevBasis, gs, ::AbstractGrid) = false


left(b::ChebyshevBasis) = -one(domaintype(b))
left(b::ChebyshevBasis, idx) = left(b)

right(b::ChebyshevBasis) = one(domaintype(b))
right(b::ChebyshevBasis, idx) = right(b)

grid(b::ChebyshevBasis) = ChebyshevNodeGrid(b.n, domaintype(b))
secondgrid(b::ChebyshevBasis) = ChebyshevExtremaGrid(b.n, domaintype(b))

# extends the default definition at transform.jl
transform_space(s::ChebyshevSpan; nodegrid=true, options...) =
    nodegrid ? gridspace(s) : gridspace(secondgrid(set(s)), coeftype(s))

# The weight function
weight(b::ChebyshevBasis{T}, x) where {T} = 1/sqrt(1-T(x)^2)

# Parameters alpha and beta of the corresponding Jacobi polynomial
jacobi_α(b::ChebyshevBasis) = -1//2
jacobi_β(b::ChebyshevBasis) = -1//2



# See DLMF, Table 18.9.1
# http://dlmf.nist.gov/18.9#i
rec_An(b::ChebyshevBasis, n::Int) = n==0 ? 1 : 2

rec_Bn(b::ChebyshevBasis, n::Int) = 0

rec_Cn(b::ChebyshevBasis, n::Int) = 1



# We can define this O(1) evaluation method, but only for points that are
# real and lie in [-1,1]
# Note that if x is not Real, recurrence_eval will be called by the OPS supertype
function eval_element(b::ChebyshevBasis, idx::Int, x::Real)
    abs(x) <= 1 ? cos((idx-1)*acos(x)) : recurrence_eval(b, idx, x)
end

# The version below is safe for points outside [-1,1] too.
# If we don't define anything, evaluation will default to using the three-term
# recurence relation.
# eval_element{T <: Real}(b::ChebyshevBasis, idx::Int, x::T) = real(cos((idx-1)*acos(x+0im)))

function eval_element_derivative(b::ChebyshevBasis, idx::Int, x)
    T = rangetype(b)
    if idx == 1
        T(0)
    else
        (idx-1) * eval_element(ChebyshevU(length(b)), idx-1, x)
    end
end

function moment(b::ChebyshevBasis{T}, idx::Int) where {T}
    n = idx-1
    if n == 0
        T(2)
    else
        isodd(n) ? zero(T) : -T(2)/((n+1)*(n-1))
    end
end

function apply!(op::Differentiation, dest::ChebyshevBasis, src::ChebyshevBasis, result, coef)
    #	@assert period(dest)==period(src)
    n = length(src)
    T = eltype(coef)
    tempc = coef[:]
    tempr = coef[:]
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
    result[1:n-order(op)] = tempr[1:n-order(op)]
    result
end

function apply!(op::AntiDifferentiation, dest::ChebyshevBasis, src::ChebyshevBasis, result, coef)
    #	@assert period(dest)==period(src)
    T = eltype(coef)
    tempc = zeros(T,length(result))
    tempc[1:length(src)] = coef[1:length(src)]
    tempr = zeros(T,length(result))
    tempr[1:length(src)] = coef[1:length(src)]
    for o = 1:order(op)
        n = length(src)+o
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
    result[:]=tempr[:]
    result
end

function gramdiagonal!(result, ::ChebyshevSpan; options...)
    T = eltype(result)
    for i in 1:length(result)
        i==1? result[i] = T(pi) : result[i] = T(pi)/2
    end
end

function UnNormalizedGram(s::ChebyshevSpan{A}, oversampling) where {A}
    d = A(length_oversampled_grid(set(s), oversampling))/2*ones(A,length(s))
    d[1] = length_oversampled_grid(set(s), oversampling)
    DiagonalOperator(s, s, d)
end


################################################################
# Methods to transform from ChebyshevBasis to ChebyshevNodeGrid
###############################################################

transform_from_grid(src, dest::ChebyshevSpan, grid::ChebyshevNodeGrid; options...) =
	_forward_chebyshev_operator(src, dest, coeftype(dest); options...)

transform_to_grid(src::ChebyshevSpan, dest, grid::ChebyshevNodeGrid; options...) =
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



function transform_to_grid_tensor(::Type{F}, ::Type{G}, s1, s2, grid; options...) where {F <: ChebyshevSpan,G <: ChebyshevNodeGrid}
	_backward_chebyshev_operator(s1, s2, coeftype(s1); options...)
end

function transform_from_grid_tensor(::Type{F}, ::Type{G}, s1, s2, grid; options...) where {F <: ChebyshevSpan,G <: ChebyshevNodeGrid}
	_forward_chebyshev_operator(s1, s2, coeftype(s2); options...)
end



function transform_from_grid_post(src, dest::ChebyshevSpan, grid::ChebyshevNodeGrid; options...)
    ELT = coeftype(dest)
    scaling = ScalingOperator(dest, 1/sqrt(ELT(length(dest)/2)))
    coefscaling = CoefficientScalingOperator(dest, 1, 1/sqrt(ELT(2)))
    flip = UnevenSignFlipOperator(dest)
	scaling * coefscaling * flip
end

transform_to_grid_pre(src::ChebyshevSpan, dest, grid::ChebyshevNodeGrid; options...) =
    inv(transform_from_grid_post(dest, src, grid; options...))


##################################################################
# Methods to transform from ChebyshevBasis to ChebyshevExtremaGrid
##################################################################

transform_from_grid(src, dest::ChebyshevSpan, grid::ChebyshevExtremaGrid; options...) =
_chebyshevI_operator(src, dest, coeftype(dest); options...)

transform_to_grid(src::ChebyshevSpan, dest, grid::ChebyshevExtremaGrid; options...) =
_chebyshevI_operator(src, dest, coeftype(src); options...)

# These are the generic fallbacks
_chebyshevI_operator(src, dest, ::Type{T}; options...) where {T <: Number} =
    FastChebyshevITransform(src, dest)

# But for some types we use FFTW
for op in (:Float32, :Float64, :(Complex{Float32}), :(Complex{Float64}))
  @eval _chebyshevI_operator(src, dest, ::Type{$(op)}; options...) =
   FastChebyshevITransformFFTW(src, dest; options...)
end

function transform_to_grid_tensor(::Type{F}, ::Type{G}, s1, s2, grid; options...) where {F <: ChebyshevSpan,G <: ChebyshevExtremaGrid}
    _chebyshevI_operator(s1, s2, eltype(s1, s2); options...)
end

function transform_from_grid_tensor(::Type{F}, ::Type{G}, s1, s2, grid; options...) where {F <: ChebyshevSpan,G <: ChebyshevExtremaGrid}
    _chebyshevI_operator(s1, s2, eltype(s1, s2); options...)
end

function transform_to_grid_pre(src::ChebyshevSpan, dest, grid::ChebyshevExtremaGrid; options...)
    T = coeftype(src)
    coefscaling1 = CoefficientScalingOperator(src, 1, T(2))
    coefscaling2 = CoefficientScalingOperator(src, length(src), T(2))
    coefscaling1 * coefscaling2
end

function transform_to_grid_post(src::ChebyshevSpan, dest, grid::ChebyshevExtremaGrid; options...)
  ELT = coeftype(src)
  ScalingOperator(dest, 1/ELT(2))
end

function transform_from_grid_post(src, dest::ChebyshevSpan, grid::ChebyshevExtremaGrid; options...)
    # Inverse DCT is unnormalized, applying DCT and its inverse gives N times the original. N=2(length-1)
    ELT = coeftype(dest)
    scaling = ScalingOperator(dest, 1/(2*ELT(length(dest)-1)))
    scaling * inv(transform_to_grid_pre(dest, src, grid; options...))
end

transform_from_grid_pre(src, dest::ChebyshevSpan, grid::ChebyshevExtremaGrid; options...) =
    inv(transform_to_grid_post(dest, src, grid; options...))






is_compatible(src1::ChebyshevBasis, src2::ChebyshevBasis) = true

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

"A basis of Chebyshev polynomials of the second kind (on the interval [-1,1])."
struct ChebyshevU{T} <: OPS{T}
    n			::	Int
end

const ChebyshevUSpace{A,F<:ChebyshevU} = Span{A,F}

ChebyshevU(n, ::Type{T} = Float64) where {T} = ChebyshevU{T}(n)

instantiate(::Type{ChebyshevU}, n, ::Type{T}) where {T} = ChebyshevU{T}(n)

set_promote_domaintype(b::ChebyshevU, ::Type{S}) where {S} =
    ChebyshevU{S}(b.n)

resize(b::ChebyshevU{T}, n) where {T} = ChebyshevU{T}(n)

name(b::ChebyshevU) = "Chebyshev series (second kind)"

function eval_element(b::ChebyshevU, idx::Int, x::Real)
    # Don't use the formula when |x|=1, because it will generate NaN's
    abs(x) < 1 ? sin(idx*acos(x))/sqrt(1-x^2) : recurrence_eval(b, idx, x)
end

left(b::ChebyshevU{T}) where {T} = -one(T)
left(b::ChebyshevU, idx) = left(b)

right(b::ChebyshevU{T}) where {T} = one(T)
right(b::ChebyshevU, idx) = right(b)

grid(b::ChebyshevU{T}) where {T} = ChebyshevNodeGrid{T}(b.n)

Gram(s::ChebyshevUSpace{A}; options...) where {A} = ScalingOperator(s, s, A(pi)/2)

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
