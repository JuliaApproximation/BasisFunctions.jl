# fourierbasis.jl


# Fourier basis on the interval [a,b]
# EVEN is true if the length of the corresponding Fourier series is even.
immutable FourierBasis{EVEN,T <: FloatingPoint} <: AbstractBasis1d{T}
	a 			::	T
	b 			::	T
	grid		::	PeriodicEquispacedGrid{T}

	FourierBasis(n, a, b) = (@assert iseven(n)==EVEN; new(a, b, PeriodicEquispacedGrid(n, a, b)))
end

typealias FourierBasisEven{T} FourierBasis{true,T}
typealias FourierBasisOdd{T} FourierBasis{false,T}

typealias FourierBasisNd{EVEN,G,N,T} TensorProductBasis{FourierBasis{EVEN,T},G,N,T}

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



length(b::FourierBasis) = length(b.grid)

left(b::FourierBasis) = b.a

left(b::FourierBasis, idx) = b.a

right(b::FourierBasis) = b.b

right(b::FourierBasis, idx) = b.b

period(b::FourierBasis) = b.b-b.a

grid(b::FourierBasis) = b.grid


function frequency(b::FourierBasisEven, idx::Int)
	checkbounds(b, idx)
	nh = length(b)>>1
	idx <= nh+1 ? idx-1 : idx - 2*nh - 1
end

function frequency(b::FourierBasisOdd, idx::Int)
	nh = (length(b)-1)>>1
	idx <= nh+1 ? idx-1 : idx - 2*nh - 2
end

mapx(b::FourierBasis, x) = (x-b.a)/(b.b-b.a)

# One has to be careful here not to match Floats and BigFloats by accident.
# Hence the conversions to T in the lines below.
call{T <: FloatingPoint}(b::FourierBasisOdd{T}, idx::Int, x::T) = exp(T(2) * pi * 1im * mapx(b, x) * frequency(b, idx))

call{T, S <: Number}(b::FourierBasisOdd{T}, idx::Int, x::S) = call(b, idx, T(x))

call{T <: FloatingPoint}(b::FourierBasisEven{T}, idx::Int, x::T) = (idx == length(b)/2+1 ? one(Complex{T}) * cos(T(2) * pi * mapx(b, x) * frequency(b, idx)) : exp(T(2) * pi * 1im * mapx(b, x) * frequency(b, idx)))

call{T, S <: Number}(b::FourierBasisEven{T}, idx::Int, x::S) = call(b, idx, T(x))



function differentiate!(dest::FourierBasisOdd, src::FourierBasisOdd, result, coef, i)
	@assert i>=0
	@assert length(dest)==length(src)

	nh = (length(b)-1)>>1
	p = period(b)

	for j = 0:nh
		result[j+1] = (2 * pi * im * j / p)^i * coef[j+1]
	end
	for j = 1:nh
		result[nh+1+j] = (2 * pi * im * (-nh-1+j) / p)^i * coef[nh+1+j]
	end
end


abstract DiscreteFourierTransform{SRC,DEST} <: AbstractDiscreteTransform{SRC,DEST}

abstract DiscreteFourierTransformFFTW{SRC,DEST} <: DiscreteFourierTransform{SRC,DEST}

# These types use FFTW and so they are (currently) limited to Float64.
# This will improve once the pure-julia implementation of FFT lands (#6193).
# But, we can also borrow from ApproxFun so let's do that right away

is_inplace(op::DiscreteFourierTransformFFTW) = True()



immutable FastFourierTransformFFTW{SRC,DEST} <: DiscreteFourierTransformFFTW{SRC,DEST}
	src		::	SRC
	dest	::	DEST
	plan!	::	Function

	FastFourierTransformFFTW(src, dest) = new(src, dest, plan_fft!(zeros(eltype(dest),size(dest)), 1:dim(dest), FFTW.ESTIMATE|FFTW.MEASURE|FFTW.PATIENT))
end

FastFourierTransformFFTW{SRC,DEST}(src::SRC, dest::DEST) = FastFourierTransformFFTW{SRC,DEST}(src, dest)

# Note that we choose to use bfft, an unscaled inverse fft.
immutable InverseFastFourierTransformFFTW{SRC,DEST} <: DiscreteFourierTransformFFTW{SRC,DEST}
	src		::	SRC
	dest	::	DEST
	plan!	::	Function

	InverseFastFourierTransformFFTW(src, dest) = new(src, dest, plan_bfft!(zeros(eltype(src),size(src)), 1:dim(src), FFTW.ESTIMATE|FFTW.MEASURE|FFTW.PATIENT))
end

InverseFastFourierTransformFFTW{SRC,DEST}(src::SRC, dest::DEST) = InverseFastFourierTransformFFTW{SRC,DEST}(src, dest)

# One implementation for forward and inverse transform in-place: call the plan
apply!(op::DiscreteFourierTransformFFTW, dest, src, coef_srcdest) = op.plan!(coef_srcdest)


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



transform_operator(src::TimeDomain, dest::FourierBasis) = _transform_operator(src, dest, eltype(src,dest))

_transform_operator(src::TimeDomain, dest::FourierBasis, ::Type{Complex{Float64}}) = FastFourierTransformFFTW(src,dest)

_transform_operator{T <: FloatingPoint}(src::TimeDomain, dest::FourierBasis, ::Type{Complex{T}}) = FastFourierTransform(src,dest)

transform_operator(src::FourierBasis, dest::TimeDomain) = _transform_operator(src, dest, eltype(src,dest))

_transform_operator(src::FourierBasis, dest::TimeDomain, ::Type{Complex{Float64}}) = InverseFastFourierTransformFFTW(src,dest)

_transform_operator{T <: FloatingPoint}(src::FourierBasis, dest::TimeDomain, ::Type{Complex{T}}) = InverseFastFourierTransform(src, dest)



