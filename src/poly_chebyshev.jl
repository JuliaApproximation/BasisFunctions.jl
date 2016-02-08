# poly_chebyshev.jl


############################################
# Chebyshev polynomials of the first kind
############################################


"""
A basis of Chebyshev polynomials of the first kind on the interval [a,b].
"""
immutable ChebyshevBasis{T} <: OPS{T}
    n			::	Int

    ChebyshevBasis(n) = new(n)
end

typealias ChebyshevBasisFirstKind{T} ChebyshevBasis{T}


name(b::ChebyshevBasis) = "Chebyshev series (first kind)"

	
ChebyshevBasis{T}(n, ::Type{T} = Float64) = ChebyshevBasis{T}(n)

instantiate{T}(::Type{ChebyshevBasis}, n, ::Type{T}) = ChebyshevBasis{T}(n)
# convenience methods
ChebyshevBasis{T}(n, a::T, b::T) = rescale(ChebyshevBasis(n,T),a,b)
ChebyshevBasis{T,S}(n, a::T, b::T, ::Type{S}) = rescale(ChebyshevBasis(n,S),a,b)

similar{T}(b::ChebyshevBasis{T}, n) = ChebyshevBasis{T}(n)
similar{T}(b::ChebyshevBasis, ::Type{T}, n) = ChebyshevBasis{T}(n)
has_grid(b::ChebyshevBasis) = true
has_derivative(b::ChebyshevBasis) = true
has_antiderivative(b::ChebyshevBasis) = true
has_transform{G <: ChebyshevIIGrid}(b::ChebyshevBasis, d::DiscreteGridSpace{G}) = true
has_extension(b::ChebyshevBasis) = true

left(b::ChebyshevBasis) = -1
left(b::ChebyshevBasis, idx) = left(b)

right(b::ChebyshevBasis) = 1
right(b::ChebyshevBasis, idx) = right(b)

grid{T}(b::ChebyshevBasis{T}) = ChebyshevIIGrid(b.n,numtype(b))

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


# Map the point x in [a,b] to the corresponding point in [-1,1]

call_element(b::ChebyshevBasis, idx::Int, x) = cos((idx-1)*acos(x))

# TODO: do we need these two routines below? Are they different from the generic ones?
function apply!(op::Extension, dest::ChebyshevBasis, src::ChebyshevBasis, coef_dest, coef_src)
	@assert length(dest) > length(src)

	for i = 1:length(src)
		coef_dest[i] = coef_src[i]
	end
	for i = length(src)+1:length(dest)
		coef_dest[i] = 0
	end
end


function apply!(op::Restriction, dest::ChebyshevBasis, src::ChebyshevBasis, coef_dest, coef_src)
	@assert length(dest) < length(src)

	for i = 1:length(dest)
		coef_dest[i] = coef_src[i]
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
    result[1:n-order(op)]=tempr[1:n-order(op)]
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
end


abstract DiscreteChebyshevTransform{SRC,DEST} <: AbstractOperator{SRC,DEST}

abstract DiscreteChebyshevTransformFFTW{SRC,DEST} <: DiscreteChebyshevTransform{SRC,DEST}

# These types use FFTW and so they are (currently) limited to Float64.
# This may improve once the pure-julia implementation of FFT lands (#6193).

is_inplace{O <: DiscreteChebyshevTransformFFTW}(::Type{O}) = True


immutable FastChebyshevTransformFFTW{SRC,DEST,ELT} <: DiscreteChebyshevTransformFFTW{SRC,DEST}
	src		::	SRC
	dest	::	DEST
	plan!	::	Base.DFT.FFTW.DCTPlan

	FastChebyshevTransformFFTW(src, dest) = new(src, dest, plan_dct!(zeros(ELT,size(dest)), 1:dim(dest); flags= FFTW.ESTIMATE|FFTW.MEASURE|FFTW.PATIENT))
end

FastChebyshevTransformFFTW{SRC,DEST}(src::SRC, dest::DEST, T) = FastChebyshevTransformFFTW{SRC,DEST,T}(src, dest)

eltype{SRC,DEST,ELT}(::Type{FastChebyshevTransformFFTW{SRC,DEST,ELT}}) = ELT

immutable InverseFastChebyshevTransformFFTW{SRC,DEST,ELT} <: DiscreteChebyshevTransformFFTW{SRC,DEST}
	src		::	SRC
	dest	::	DEST
	plan!	::	Base.DFT.FFTW.DCTPlan

	InverseFastChebyshevTransformFFTW(src, dest) = new(src, dest, plan_idct!(zeros(ELT,size(src)), 1:dim(src); flags= FFTW.ESTIMATE|FFTW.MEASURE|FFTW.PATIENT))
end

InverseFastChebyshevTransformFFTW{SRC,DEST}(src::SRC, dest::DEST, T) = InverseFastChebyshevTransformFFTW{SRC,DEST, T}(src, dest)

eltype{SRC,DEST,ELT}(::Type{InverseFastChebyshevTransformFFTW{SRC,DEST,ELT}}) = ELT

# One implementation for forward and inverse transform in-place: call the plan. Added constant to undo the normalisation.
# apply!(op::DiscreteChebyshevTransformFFTW, dest, src, coef_srcdest) = sqrt(length(dest)/2^(dim(src)))*op.plan!*coef_srcdest
function apply!(op::FastChebyshevTransformFFTW, dest, src, coef_srcdest)
    op.plan!*coef_srcdest
end

function apply!(op::InverseFastChebyshevTransformFFTW, dest, src, coef_srcdest)
    op.plan!*coef_srcdest
end


immutable FastChebyshevTransform{SRC,DEST} <: DiscreteChebyshevTransform{SRC,DEST}
	src		::	SRC
	dest	::	DEST
end

# Our alternative for non-Float64 is to use ApproxFun's fft, at least for 1d.
# This allocates memory.
# We have to implement dct in terms of fft, which allocates more memory.
function dct(a::AbstractArray{Complex{BigFloat}})
	N = big(length(a))
    c = fft([a; flipdim(a,1)])
    d = c[1:N] .* exp(-im*big(pi)*(0:N-1)/(2*N))
    d[1] = d[1] / sqrt(big(2))
    d / sqrt(2*N)
end

dct(a::AbstractArray{BigFloat}) = real(dct(a+0im))

function idct(a::AbstractArray{Complex{BigFloat}})
    N = big(length(a))
    b = a * sqrt(2*N)
    b[1] = b[1] * sqrt(big(2))
    b = b .* exp(im*big(pi)*(0:N-1)/(2*N))
    b = [b; 0; conj(flipdim(b[2:end],1))]
    c = ifft(b)
    c[1:N]
end

idct(a::AbstractArray{BigFloat}) = real(idct(a+0im))

apply!(op::FastChebyshevTransform, dest, src, coef_dest, coef_src) = (coef_dest[:] = dct(coef_src))


immutable InverseFastChebyshevTransform{SRC,DEST} <: DiscreteChebyshevTransform{SRC,DEST}
	src		::	SRC
	dest	::	DEST
end

#again, why is this necessary?
## apply!(op::InverseFastChebyshevTransform, dest, src, coef_dest::Array{Complex{BigFloat}}, coef_src::Array{Complex{BigFloat}}) = (coef_dest[:] = idct(coef_src) * sqrt(length(dest))/2^(dim(src)))
apply!(op::InverseFastChebyshevTransform, dest, src, coef_dest, coef_src) = (coef_dest[:] = idct(coef_src))


ctranspose(op::FastChebyshevTransform) = InverseFastChebyshevTransform(dest(op), src(op))
ctranspose(op::FastChebyshevTransformFFTW) = InverseFastChebyshevTransformFFTW(dest(op), src(op))

ctranspose(op::InverseFastChebyshevTransform) = FastChebyshevTransform(dest(op), src(op))
ctranspose(op::InverseFastChebyshevTransformFFTW) = FastChebyshevTransformFFTW(dest(op), src(op))

inv(op::DiscreteChebyshevTransform) = ctranspose(op)

# TODO: restrict the grid of grid space here
transform_operator(src::DiscreteGridSpace, dest::ChebyshevBasis) =
	_forward_chebyshev_operator(src, dest, eltype(src,dest))

_forward_chebyshev_operator(src::DiscreteGridSpace, dest::ChebyshevBasis, ::Type{Float64}) =
	FastChebyshevTransformFFTW(src,dest, Float64)

_forward_chebyshev_operator(src::DiscreteGridSpace, dest::ChebyshevBasis, ::Type{Complex{Float64}}) =
	FastChebyshevTransformFFTW(src,dest, Complex{Float64})

_forward_chebyshev_operator{T <: Number}(src::DiscreteGridSpace, dest::ChebyshevBasis, ::Type{T}) =
	FastChebyshevTransform(src,dest)



transform_operator(src::ChebyshevBasis, dest::DiscreteGridSpace) =
	_backward_chebyshev_operator(src, dest, eltype(src,dest))

_backward_chebyshev_operator(src::ChebyshevBasis, dest::DiscreteGridSpace, ::Type{Float64}) =
	InverseFastChebyshevTransformFFTW(src,dest, Float64)

_backward_chebyshev_operator(src::ChebyshevBasis, dest::DiscreteGridSpace, ::Type{Complex{Float64}}) =
	InverseFastChebyshevTransformFFTW(src,dest,Complex{Float64})

_backward_chebyshev_operator{T <: Number}(src::ChebyshevBasis, dest::DiscreteGridSpace, ::Type{T}) =
	InverseFastChebyshevTransform(src, dest)



immutable ChebyshevNormalization{ELT,SRC} <: AbstractOperator{SRC,SRC}
    src     :: SRC
end

eltype{ELT,SRC}(::Type{ChebyshevNormalization{ELT,SRC}}) = ELT

transform_normalization_operator{T,ELT}(src::ChebyshevBasis{T}, ::Type{ELT} = T) =
	ScalingOperator(src,1/sqrt(T(length(src)/2)))*CoefficientScalingOperator(src,1,1/sqrt(T(2)))*UnevenSignFlipOperator(src)

function apply!(op::ChebyshevNormalization, dest, src, coef_srcdest)
	L = length(op.src)
	T = numtype(src)
	s = 1/sqrt(T(L)/2)
    coef_srcdest[1] /= sqrt(T(2))
end

dest(op::ChebyshevNormalization) = src(op)

is_inplace{OP <: ChebyshevNormalization}(::Type{OP}) = True

is_diagonal{OP <: ChebyshevNormalization}(::Type{OP}) = True

ctranspose(op::ChebyshevNormalization) = op



############################################
# Chebyshev polynomials of the second kind
############################################

"A basis of Chebyshev polynomials of the second kind (on the interval [-1,1])."
immutable ChebyshevBasisSecondKind{T} <: OPS{T}
    n			::	Int
end

ChebyshevBasisSecondKind{T}(n, ::Type{T} = Float64) = ChebyshevBasisSecondKind{T}(n)

instantiate{T}(::Type{ChebyshevBasisSecondKind}, n, ::Type{T}) = ChebyshevBasisSecondKind{T}(n)

name(b::ChebyshevBasisSecondKind) = "Chebyshev series (second kind)"


left{T}(b::ChebyshevBasisSecondKind{T}) = -one(T)
left{T}(b::ChebyshevBasisSecondKind{T}, idx) = left(b)

right{T}(b::ChebyshevBasisSecondKind{T}) = one(T)
right{T}(b::ChebyshevBasisSecondKind{T}, idx) = right(b)

grid{T}(b::ChebyshevBasisSecondKind{T}) = ChebyshevIIGrid{T}(b.n)


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



