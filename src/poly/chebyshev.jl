# chebyshev.jl


############################################
# Chebyshev polynomials of the first kind
############################################


"""
A basis of Chebyshev polynomials of the first kind on the interval [-1,1].
"""
immutable ChebyshevBasis{T} <: OPS{T}
    n			::	Int

    ChebyshevBasis(n) = new(n)
end

typealias ChebyshevBasisFirstKind{T} ChebyshevBasis{T}

name(b::ChebyshevBasis) = "Chebyshev series (first kind)"


ChebyshevBasis{T}(n, ::Type{T} = Float64) = ChebyshevBasis{T}(n)

ChebyshevBasis{T}(n, a, b, ::Type{T} = promote_type(typeof(a),typeof(b))) = rescale( ChebyshevBasis(n,floatify(T)), a, b)

instantiate{T}(::Type{ChebyshevBasis}, n, ::Type{T}) = ChebyshevBasis{T}(n)

promote_eltype{T,S}(b::ChebyshevBasis{T}, ::Type{S}) = ChebyshevBasis{promote_type(T,S)}(b.n)

resize(b::ChebyshevBasis, n) = ChebyshevBasis(n, eltype(b))


has_grid(b::ChebyshevBasis) = true
has_derivative(b::ChebyshevBasis) = true
has_antiderivative(b::ChebyshevBasis) = true

has_grid_transform(b::ChebyshevBasis, dgs, ::ChebyshevNodeGrid) = length(b) == length(dgs)
has_grid_transform(b::ChebyshevBasis, dgs, ::ChebyshevExtremaGrid) = length(b) == length(dgs)
has_grid_transform(b::ChebyshevBasis, dgs, ::AbstractGrid) = false


left(b::ChebyshevBasis) = -one(numtype(b))
left(b::ChebyshevBasis, idx) = left(b)

right(b::ChebyshevBasis) = one(numtype(b))
right(b::ChebyshevBasis, idx) = right(b)

grid{T}(b::ChebyshevBasis{T}) = ChebyshevNodeGrid(b.n,numtype(b))
secondgrid{T}(b::ChebyshevBasis{T}) = ChebyshevExtremaGrid(b.n,numtype(b))
# extends the default definition at transform.jl
transform_set(set::ChebyshevBasis; nodegrid=true, options...) =
    nodegrid ? DiscreteGridSpace(grid(set), eltype(set)) : DiscreteGridSpace(secondgrid(set), eltype(set))

# The weight function
weight{T}(b::ChebyshevBasis{T}, x) = 1/sqrt(1-T(x)^2)

# Parameters alpha and beta of the corresponding Jacobi polynomial
jacobi_α(b::ChebyshevBasis) = -1//2
jacobi_β(b::ChebyshevBasis) = -1//2



# See DLMF, Table 18.9.1
# http://dlmf.nist.gov/18.9#i
rec_An(b::ChebyshevBasis, n::Int) = n==0 ? 1 : 2

rec_Bn(b::ChebyshevBasis, n::Int) = 0

rec_Cn(b::ChebyshevBasis, n::Int) = 1


# We can define this O(1) evaluation method, but only for points in [-1,1]
# The version below is safe for points outside [-1,1] too.
# If we don't define anything, evaluation will default to using the three-term
# recurence relation.

# eval_element(b::ChebyshevBasis, idx::Int, x) = cos((idx-1)*acos(x))

# eval_element{T <: Real}(b::ChebyshevBasis, idx::Int, x::T) = real(cos((idx-1)*acos(x+0im)))

function moment{T}(b::ChebyshevBasis{T}, idx::Int)
    n = idx-1
    if n == 0
        T(2)
    else
        isodd(n) ? zero(T) : -T(2)/((n+1)*(n-1))
    end
end

function apply!{T}(op::Differentiation, dest::ChebyshevBasis{T}, src::ChebyshevBasis{T}, result, coef)
    #	@assert period(dest)==period(src)
    n = length(src)
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

function apply!{T}(op::AntiDifferentiation, dest::ChebyshevBasis{T}, src::ChebyshevBasis{T}, result, coef)
    #	@assert period(dest)==period(src)
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



################################################################
# Methods to transform from ChebyshevBasis to ChebyshevNodeGrid
###############################################################
transform_from_grid(src, dest::ChebyshevBasis, grid::ChebyshevNodeGrid; options...) =
	_forward_chebyshev_operator(src, dest, eltype(src,dest); options...)

transform_to_grid(src::ChebyshevBasis, dest, grid::ChebyshevNodeGrid; options...) =
	_backward_chebyshev_operator(src, dest, eltype(src,dest); options...)

# These are the generic fallbacks
_forward_chebyshev_operator{T <: Number}(src, dest, ::Type{T}; options...) =
	FastChebyshevTransform(src, dest)

_backward_chebyshev_operator{T <: Number}(src, dest, ::Type{T}; options...) =
	InverseFastChebyshevTransform(src, dest)

# But for some types we use FFTW
for op in (:Float32, :Float64, :(Complex{Float32}), :(Complex{Float64}))
    @eval _forward_chebyshev_operator(src, dest, ::Type{$(op)}; options...) =
	   FastChebyshevTransformFFTW(src, dest; options...)
    @eval _backward_chebyshev_operator(src, dest, ::Type{$(op)}; options...) =
       	InverseFastChebyshevTransformFFTW(src, dest; options...)
end



function transform_to_grid_tensor{F <: ChebyshevBasis,G <: ChebyshevNodeGrid}(::Type{F}, ::Type{G}, s1, s2, grid; options...)
	_backward_chebyshev_operator(s1, s2, eltype(s1, s2); options...)
end

function transform_from_grid_tensor{F <: ChebyshevBasis,G <: ChebyshevNodeGrid}(::Type{F}, ::Type{G}, s1, s2, grid; options...)
	_forward_chebyshev_operator(s1, s2, eltype(s1, s2); options...)
end



function transform_from_grid_post(src, dest::ChebyshevBasis, grid::ChebyshevNodeGrid; options...)
    ELT = eltype(dest)
    scaling = ScalingOperator(dest, 1/sqrt(ELT(length(dest)/2)))
    coefscaling = CoefficientScalingOperator(dest, 1, 1/sqrt(ELT(2)))
    flip = UnevenSignFlipOperator(dest)
	scaling * coefscaling * flip
end

transform_to_grid_pre(src::ChebyshevBasis, dest, grid::ChebyshevNodeGrid; options...) =
    inv(transform_from_grid_post(dest, src, grid; options...))


##################################################################
# Methods to transform from ChebyshevBasis to ChebyshevExtremaGrid
##################################################################
transform_from_grid(src, dest::ChebyshevBasis, grid::ChebyshevExtremaGrid; options...) =
_chebyshevI_operator(src, dest, eltype(src,dest); options...)

transform_to_grid(src::ChebyshevBasis, dest, grid::ChebyshevExtremaGrid; options...) =
_chebyshevI_operator(src, dest, eltype(src,dest); options...)

# These are the generic fallbacks
_chebyshevI_operator{T <: Number}(src, dest, ::Type{T}; options...) =
FastChebyshevITransform(src, dest)

# But for some types we use FFTW
for op in (:Float32, :Float64, :(Complex{Float32}), :(Complex{Float64}))
  @eval _chebyshevI_operator(src, dest, ::Type{$(op)}; options...) =
   FastChebyshevITransformFFTW(src, dest; options...)
end

function transform_to_grid_tensor{F <: ChebyshevBasis,G <: ChebyshevExtremaGrid}(::Type{F}, ::Type{G}, s1, s2, grid; options...)
  _chebyshevI_operator(s1, s2, eltype(s1, s2); options...)
end

function transform_from_grid_tensor{F <: ChebyshevBasis,G <: ChebyshevExtremaGrid}(::Type{F}, ::Type{G}, s1, s2, grid; options...)
  _chebyshevI_operator(s1, s2, eltype(s1, s2); options...)
end

function transform_to_grid_pre(src::ChebyshevBasis, dest, grid::ChebyshevExtremaGrid; options...)
    ELT = eltype(src)
    coefscaling1 = CoefficientScalingOperator(src, 1, ELT(2))
    coefscaling2 = CoefficientScalingOperator(src, length(src), ELT(2))
  coefscaling1 * coefscaling2
end

function transform_to_grid_post(src::ChebyshevBasis, dest, grid::ChebyshevExtremaGrid; options...)
  ELT = eltype(src)
  ScalingOperator(dest, 1/ELT(2))
end

function transform_from_grid_post(src, dest::ChebyshevBasis, grid::ChebyshevExtremaGrid; options...)
    # Inverse DCT is unnormalized, applying DCT and its inverse gives N times the original. N=2(length-1)
    ELT = eltype(dest)
    scaling = ScalingOperator(dest, 1/(2*ELT(length(dest)-1)))
  scaling * inv(transform_to_grid_pre(dest, src, grid; options...))
end

transform_from_grid_pre(src, dest::ChebyshevBasis, grid::ChebyshevExtremaGrid; options...) =
  inv(transform_to_grid_post(dest, src, grid; options...))






is_compatible(src1::ChebyshevBasis, src2::ChebyshevBasis) = true

function (*)(src1::ChebyshevBasis, src2::ChebyshevBasis, coef_src1, coef_src2)
    dest = ChebyshevBasis(length(src1)+length(src2),eltype(src1,src2))
    coef_dest = zeros(eltype(dest),length(dest))
    for i = 1:length(src1)
        for j = 1:length(src2)
            coef_dest[i+j-1]+=1/2*coef_src1[i]*coef_src2[j]
            coef_dest[abs(i-j)+1]+=1/2*coef_src1[i]*coef_src2[j]
        end
    end
    (dest,coef_dest)
end

############################################
# Chebyshev polynomials of the second kind
############################################

"A basis of Chebyshev polynomials of the second kind (on the interval [-1,1])."
immutable ChebyshevBasisSecondKind{T} <: OPS{T}
    n			::	Int
end

ChebyshevBasisSecondKind{T}(n, ::Type{T} = Float64) = ChebyshevBasisSecondKind{T}(n)

instantiate{T}(::Type{ChebyshevBasisSecondKind}, n, ::Type{T}) = ChebyshevBasisSecondKind{T}(n)

promote_eltype{T,S}(b::ChebyshevBasisSecondKind{T}, ::Type{S}) = ChebyshevBasisSecondKind{promote_type(T,S)}(b.n)

resize(b::ChebyshevBasisSecondKind, n) = ChebyshevBasisSecondKind(n, eltype(b))

name(b::ChebyshevBasisSecondKind) = "Chebyshev series (second kind)"


left{T}(b::ChebyshevBasisSecondKind{T}) = -one(T)
left{T}(b::ChebyshevBasisSecondKind{T}, idx) = left(b)

right{T}(b::ChebyshevBasisSecondKind{T}) = one(T)
right{T}(b::ChebyshevBasisSecondKind{T}, idx) = right(b)

grid{T}(b::ChebyshevBasisSecondKind{T}) = ChebyshevNodeGrid{T}(b.n)


# The weight function
weight{T}(b::ChebyshevBasisSecondKind{T}, x) = sqrt(1-T(x)^2)

# Parameters alpha and beta of the corresponding Jacobi polynomial
jacobi_α(b::ChebyshevBasisSecondKind) = 1//2
jacobi_β(b::ChebyshevBasisSecondKind) = 1//2


# See DLMF, Table 18.9.1
# http://dlmf.nist.gov/18.9#i
rec_An(b::ChebyshevBasisSecondKind, n::Int) = 2

rec_Bn(b::ChebyshevBasisSecondKind, n::Int) = 0

rec_Cn(b::ChebyshevBasisSecondKind, n::Int) = 1
