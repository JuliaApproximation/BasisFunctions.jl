# fourierbasis.jl


# Fourier basis on the interval [a,b]
# EVEN is true if the length of the corresponding Fourier series is even.
immutable FourierBasis{EVEN,T <: AbstractFloat} <: AbstractBasis1d{T}
	n			::	Int
	a 			::	T
	b 			::	T

	FourierBasis(n, a, b) = (@assert iseven(n)==EVEN; new(n, a, b))
end

typealias FourierBasisEven{T} FourierBasis{true,T}
typealias FourierBasisOdd{T} FourierBasis{false,T}

name(b::FourierBasis) = "Fourier series"

isreal(b::FourierBasis) = False()
isreal{B <: FourierBasis}(::Type{B}) = False

iseven{EVEN}(::FourierBasis{EVEN}) = EVEN
iseven{EVEN,T}(::Type{FourierBasis{EVEN,T}}) = EVEN

isodd{EVEN}(::FourierBasis{EVEN}) = ~EVEN
isodd{EVEN,T}(::Type{FourierBasis{EVEN,T}}) = ~EVEN


FourierBasis{T}(n, a::T = -1.0, b::T = 1.0) = FourierBasis{iseven(n),T}(n, a, b)

# Typesafe method for constructing a Fourier series with even length
fourier_basis_even_length{T}(n, a::T = -1.0, b::T = 1.0) = FourierBasis{true,T}(n, a, b)

# Typesafe method for constructing a Fourier series with odd length
fourier_basis_odd_length{T}(n, a::T = -1.0, b::T = 1.0) = FourierBasis{false,T}(n, a, b)



length(b::FourierBasis) = b.n

left(b::FourierBasis) = b.a

left(b::FourierBasis, idx) = b.a

right(b::FourierBasis) = b.b

right(b::FourierBasis, idx) = b.b

period(b::FourierBasis) = b.b-b.a

grid(b::FourierBasis) = PeriodicEquispacedGrid(b.n, b.a, b.b)

nhalf(b::FourierBasis) = length(b)>>1


# Map the point x in [a,b] to the corresponding point in [0,1]
mapx(b::FourierBasis, x) = (x-b.a)/(b.b-b.a)

# The map between the index of a basis function and its Fourier frequency
idx2frequency(b::FourierBasisEven, idx::Int) = idx <= nhalf(b)+1 ? idx-1 : idx - 2*nhalf(b) - 1
idx2frequency(b::FourierBasisOdd, idx::Int) = idx <= nhalf(b)+1 ? idx-1 : idx - 2*nhalf(b) - 2

# The inverse map
frequency2idx(b::FourierBasis, freq::Int) = freq >= 0 ? freq+1 : length(b)-freq+1
#frequency2idx(b::FourierBasisOdd, freq::Int) = freq >= 0 ? freq+1 : length(b)-freq+1

# One has to be careful here not to match Floats and BigFloats by accident.
# Hence the conversions to T in the lines below.
call{T <: AbstractFloat}(b::FourierBasisOdd{T}, idx::Int, x::T) = exp(2 * T(pi) * 1im * mapx(b, x) * idx2frequency(b, idx))

call{T, S <: Number}(b::FourierBasisOdd{T}, idx::Int, x::S) = call(b, idx, T(x))

call{T <: AbstractFloat}(b::FourierBasisEven{T}, idx::Int, x::T) =
	(idx == nhalf(b)+1	? one(Complex{T}) * cos(2 * T(pi) * mapx(b, x) * idx2frequency(b,idx))
						: exp(2 * T(pi) * 1im * mapx(b, x) * idx2frequency(b,idx)))

call{T, S <: Number}(b::FourierBasisEven{T}, idx::Int, x::S) = call(b, idx, T(x))


function apply!{T}(op::Differentiation, dest::FourierBasisOdd{T}, src::FourierBasisOdd{T}, result, coef)
	@assert length(dest)==length(src)
#	@assert period(dest)==period(src)
	@assert op.var == 1

	nh = nhalf(src)
	p = period(src)
	i = order(op)

	for j = 0:nh
		result[j+1] = (2 * T(pi) * im * j / p)^i * coef[j+1]
	end
	for j = 1:nh
		result[nh+1+j] = (2 * T(pi) * im * (-nh-1+j) / p)^i * coef[nh+1+j]
	end
end


function apply!(op::Extension, dest::FourierBasisOdd, src::FourierBasisEven, coef_dest, coef_src)
	@assert length(dest) > length(src)

	nh = nhalf(src)

	for i=0:nh-1
		coef_dest[i+1] = coef_src[i+1]
	end
	for i=1:nh-1
		coef_dest[end-nh+i+1] = coef_src[nh+1+i]
	end
	coef_dest[nh+1] = coef_src[nh+1]/2
	coef_dest[end-nh+1] = coef_src[nh+1]/2
	for i = nh+2:length(coef_dest)-nh
		coef_dest[i] = 0
	end
end

function apply!(op::Extension, dest::FourierBasis, src::FourierBasisOdd, coef_dest, coef_src)
	@assert length(dest) > length(src)

	nh = nhalf(src)

	for i=0:nh
		coef_dest[i+1] = coef_src[i+1]
	end
	for i=1:nh
		coef_dest[end-nh+i] = coef_src[nh+1+i]
	end
	for i = nh+2:length(coef_dest)-nh
		coef_dest[i] = 0
	end
end


function apply!(op::Restriction, dest::FourierBasisOdd, src::FourierBasis, coef_dest, coef_src)
	@assert length(dest) < length(src)

	nh = nhalf(dest)
	for i=0:nh
		coef_dest[i+1] = coef_src[i+1]
	end
	for i=1:nh
		coef_dest[nh+1+i] = coef_src[end-nh+i]
	end
end



abstract DiscreteFourierTransform{SRC,DEST} <: AbstractOperator{SRC,DEST}

abstract DiscreteFourierTransformFFTW{SRC,DEST} <: DiscreteFourierTransform{SRC,DEST}

# These types use FFTW and so they are (currently) limited to Float64.
# This will improve once the pure-julia implementation of FFT lands (#6193).
# But, we can also borrow from ApproxFun so let's do that right away

is_inplace(op::DiscreteFourierTransformFFTW) = True()


immutable FastFourierTransformFFTW{SRC,DEST} <: DiscreteFourierTransformFFTW{SRC,DEST}
	src		::	SRC
	dest	::	DEST
	plan!	::	Base.DFT.FFTW.cFFTWPlan

	FastFourierTransformFFTW(src, dest) = new(src, dest, plan_fft!(zeros(eltype(dest),size(dest)), 1:dim(dest); flags= FFTW.ESTIMATE|FFTW.MEASURE|FFTW.PATIENT))
end

FastFourierTransformFFTW{SRC,DEST}(src::SRC, dest::DEST) = FastFourierTransformFFTW{SRC,DEST}(src, dest)

# Note that we choose to use bfft, an unscaled inverse fft.
immutable InverseFastFourierTransformFFTW{SRC,DEST} <: DiscreteFourierTransformFFTW{SRC,DEST}
	src		::	SRC
	dest	::	DEST
	plan!	::	Base.DFT.FFTW.cFFTWPlan

	InverseFastFourierTransformFFTW(src, dest) = new(src, dest, plan_bfft!(zeros(eltype(src),size(src)), 1:dim(src); flags= FFTW.ESTIMATE|FFTW.MEASURE|FFTW.PATIENT))
end

InverseFastFourierTransformFFTW{SRC,DEST}(src::SRC, dest::DEST) = InverseFastFourierTransformFFTW{SRC,DEST}(src, dest)

# One implementation for forward and inverse transform in-place: call the plan
apply!(op::DiscreteFourierTransformFFTW, dest, src, coef_srcdest) = op.plan!*coef_srcdest


immutable FastFourierTransform{SRC,DEST} <: DiscreteFourierTransform{SRC,DEST}
	src		::	SRC
	dest	::	DEST
end

# Our alternative for non-Float64 is to use ApproxFun's fft, at least for 1d.
# This allocates memory.
apply!(op::FastFourierTransform, dest, src, coef_dest, coef_src) = (coef_dest[:] = fft(coef_src))


immutable InverseFastFourierTransform{SRC,DEST} <: DiscreteFourierTransform{SRC,DEST}
	src		::	SRC
	dest	::	DEST
end

apply!(op::InverseFastFourierTransform, dest, src, coef_dest::Array{Complex{BigFloat}}, coef_src::Array{Complex{BigFloat}}) = (coef_dest[:] = ifft(coef_src) * length(coef_src))



# Default differentiation operator
differentiation_operator(b::FourierBasisOdd) = Differentiation(b, b)

function differentiation_operator(b::FourierBasisEven)
	b_odd = fourier_basis_odd_length(length(b)+1, left(b), right(b))
	differentiation_operator(b_odd) * Extension(b, b_odd)
end


# For the default Fourier transform, we have to distinguish (for the time being) between the version for Float64 and other types (like BigFloat).
# The discrete grid should actually be Periodic and equispaced!
AnyFourierBasis = Union{FourierBasis, TensorProductSet}
AnyDiscreteGridSpace = Union{DiscreteGridSpace, TensorProductSet}

transform_operator{N}(src::TensorProductSet,dest::TensorProductSet, srcsets::NTuple{N,FourierBasis},destsets::NTuple{N,DiscreteGridSpace}) = _backward_fourier_operator(src,dest,eltype(src,dest))
transform_operator{N}(src::TensorProductSet,dest::TensorProductSet, srcsets::NTuple{N,DiscreteGridSpace},destsets::NTuple{N,FourierBasis}) = _forward_fourier_operator(src,dest,eltype(src,dest))


transform_operator(src::DiscreteGridSpace, dest::FourierBasis) = _forward_fourier_operator(src, dest, eltype(src,dest))

_forward_fourier_operator(src::AnyDiscreteGridSpace, dest::AnyFourierBasis, ::Type{Complex{Float64}}) = FastFourierTransformFFTW(src,dest)

_forward_fourier_operator{T <: AbstractFloat}(src::AnyDiscreteGridSpace, dest::AnyFourierBasis, ::Type{Complex{T}}) = FastFourierTransform(src,dest)



transform_operator(src::FourierBasis, dest::DiscreteGridSpace) = _backward_fourier_operator(src, dest, eltype(src,dest))

_backward_fourier_operator(src::AnyFourierBasis, dest::AnyDiscreteGridSpace, ::Type{Complex{Float64}}) = InverseFastFourierTransformFFTW(src,dest)

_backward_fourier_operator{T <: AbstractFloat}(src::AnyFourierBasis, dest::AnyDiscreteGridSpace, ::Type{Complex{T}}) = InverseFastFourierTransform(src, dest)



# The default approximation operator for a Fourier series is the FFT.
approximation_operator(b::FourierBasis) = ScalingOperator(b, 1/length(b)) * transform_operator(grid(b), b)

evaluation_operator(b::FourierBasis) = transform_operator(b, grid(b))




