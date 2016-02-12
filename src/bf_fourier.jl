# bf_fourier.jl


"""
A Fourier basis on the interval [a,b].
EVEN is true if the length of the corresponding Fourier series is even.
"""
immutable FourierBasis{EVEN,T} <: AbstractBasis1d{T}
	n			::	Int

	FourierBasis(n) = (@assert iseven(n)==EVEN; new(n))
end

typealias FourierBasisEven{T} FourierBasis{true,T}
typealias FourierBasisOdd{T} FourierBasis{false,T}

name(b::FourierBasis) = "Fourier series"

# The Element Type of a Fourier Basis is complex by definition. Real types are complexified.
FourierBasis{T}(n, ::Type{T} = Complex{Float64}) = FourierBasis{iseven(n),complexify(T)}(n)
# convenience methods
FourierBasis{T}(n, a::T, b::T) = rescale(FourierBasis(n,complexify(T)),a,b)
FourierBasis{T,S}(n, a::T, b::T, ::Type{S}) = rescale(FourierBasis(n,S),a,b)
# Typesafe methods for constructing a Fourier series with even length
fourier_basis_even{T}(n, ::Type{T}) = FourierBasis{true,T}(n)

# Typesafe method for constructing a Fourier series with odd length
fourier_basis_odd{T}(n, ::Type{T}) = FourierBasis{false,T}(n)


instantiate{T}(::Type{FourierBasis}, n, ::Type{T}) = FourierBasis(n, T)

similar(b::FourierBasisEven, T, n::Int) = FourierBasis{iseven(n),T}(n)
similar(b::FourierBasisOdd, T, n::Int) = FourierBasis{iseven(n),T}(n)

# Traits

isreal{B <: FourierBasis}(::Type{B}) = False

iseven{EVEN,T}(::Type{FourierBasis{EVEN,T}}) = EVEN
iseven(b::FourierBasis) = iseven(typeof(b))

isodd{EVEN,T}(::Type{FourierBasis{EVEN,T}}) = ~EVEN
isodd(b::FourierBasis) = isodd(typeof(b))

is_orthogonal{B <: FourierBasis}(::Type{B}) = True
is_biorthogonal{B <: FourierBasis}(::Type{B}) = True


# Methods for purposes of testing functionality.
has_grid(b::FourierBasis) = true
has_derivative(b::FourierBasis) = true
# Until adapted for DC coefficient
has_antiderivative(b::FourierBasis) = false
has_transform{G <: PeriodicEquispacedGrid}(b::FourierBasis, d::DiscreteGridSpace{G}) = true
has_extension(b::FourierBasis) = true


length(b::FourierBasis) = b.n

left(b::FourierBasis) = -1

left(b::FourierBasis, idx) = -1

right(b::FourierBasis) = 1

right(b::FourierBasis, idx) = 1

period(b::FourierBasis) = 2

grid(b::FourierBasis) = PeriodicEquispacedGrid(b.n, numtype(b))

nhalf(b::FourierBasis) = length(b)>>1


# Map the point x in [a,b] to the corresponding point in [0,1]
mapx(b::FourierBasis, x) = (x+1.0)/(2.0)

# Natural index of an even Fourier basis ranges from -N+1 to N.
natural_index(b::FourierBasisEven, idx) = idx <= nhalf(b)+1 ? idx-1 : idx - 2*nhalf(b) - 1

# Natural index of an odd Fourier basis ranges from -N to N.
natural_index(b::FourierBasisOdd, idx::Int) = idx <= nhalf(b)+1 ? idx-1 : idx - 2*nhalf(b) - 2

logical_index(b::FourierBasis, freq) = freq >= 0 ? freq+1 : length(b)+freq+1

idx2frequency(b::FourierBasis, idx::Int) = natural_index(b, idx)
frequency2idx(b::FourierBasis, freq::Int) = logical_index(b, freq)

# One has to be careful here not to match Floats and BigFloats by accident.
# Hence the conversions to T in the lines below.
call_element{T, S <: Number}(b::FourierBasisOdd{T}, idx::Int, x::S) = exp(mapx(b, x) * 2 * T(pi) * 1im  * idx2frequency(b, idx))

call_element{T, S <: Number}(b::FourierBasisEven{T}, idx::Int, x::S) =
	(idx == nhalf(b)+1	?  cos(mapx(b, x) * 2 * T(pi) * idx2frequency(b,idx))
						: exp(mapx(b, x) * 2 * T(pi) * 1im * idx2frequency(b,idx)))


function apply!{T}(op::Differentiation, dest::FourierBasisOdd{T}, src::FourierBasisOdd{T}, result, coef)
	@assert length(dest)==length(src)
#	@assert period(dest)==period(src)

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

function apply!{T}(op::AntiDifferentiation, dest::FourierBasisOdd{T}, src::FourierBasisOdd{T}, result, coef)
	@assert length(dest)==length(src)
#	@assert period(dest)==period(src)

	nh = nhalf(src)
	p = period(src)
	i = -1*order(op)

        result[1] = 0
	for j = 1:nh
		result[j+1] = (2 * T(pi) * im * j / p)^i * coef[j+1]
	end
	for j = 1:nh
		result[nh+1+j] = (2 * T(pi) * im * (-nh-1+j) / p)^i * coef[nh+1+j]
	end
end

extension_size(b::FourierBasisEven) = 2*length(b)
extension_size(b::FourierBasisOdd) = 2*length(b)+1

approx_length(b::FourierBasisEven, n::Int) = iseven(n) ? n : n+1
approx_length(b::FourierBasisOdd, n::Int) = isodd(n) ? n : n+1


function apply!(op::Extension, dest::FourierBasis, src::FourierBasisEven, coef_dest, coef_src)
	## @assert length(dest) > length(src)

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
	## @assert length(dest) > length(src)

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
	## @assert length(dest) < length(src)

	nh = nhalf(dest)
	for i=0:nh
		coef_dest[i+1] = coef_src[i+1]
	end
	for i=1:nh
		coef_dest[nh+1+i] = coef_src[end-nh+i]
	end
end

function apply!(op::Restriction, dest::FourierBasisEven, src::FourierBasis, coef_dest, coef_src)
	## @assert length(dest) < length(src)

	nh = nhalf(dest)
	for i=0:nh-1
		coef_dest[i+1] = coef_src[i+1]
	end
	for i=1:nh-1
		coef_dest[nh+1+i] = coef_src[end-nh+i+1]
	end
	coef_dest[nh+1] = coef_src[nh+1] + coef_src[end-nh+1]
end
# We extend the even basis both for derivation and antiderivation, regardless of order
for op in (:derivative_space, :antiderivative_space)
    @eval $op(b::FourierBasisEven, order::Int) = fourier_basis_odd(length(b)+1,eltype(b))
end

for op in (:differentiation_operator, :antidifferentiation_operator)
    @eval function $op(b::FourierBasisEven, b_odd::FourierBasisOdd, order::Int)
        $op(b_odd, order) * extension_operator(b, b_odd)
    end
end


abstract DiscreteFourierTransform{SRC,DEST} <: AbstractOperator{SRC,DEST}

abstract DiscreteFourierTransformFFTW{SRC,DEST} <: DiscreteFourierTransform{SRC,DEST}

# These types use FFTW and so they are (currently) limited to Float64.
# This will improve once the pure-julia implementation of FFT lands (#6193).
# But, we can also borrow from ApproxFun so let's do that right away

is_inplace{O <: DiscreteFourierTransformFFTW}(::Type{O}) = True


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

function apply!(op::FastFourierTransformFFTW, dest, src, coef_srcdest)
    op.plan!*coef_srcdest
    for i=1:length(coef_srcdest)
        coef_srcdest[i]/=sqrt(length(coef_srcdest))
    end
end

function apply!(op::InverseFastFourierTransformFFTW, dest, src, coef_srcdest)
    op.plan!*coef_srcdest
    for i=1:length(coef_srcdest)
        coef_srcdest[i]/=sqrt(length(coef_srcdest))
    end
end


immutable FastFourierTransform{SRC,DEST} <: DiscreteFourierTransform{SRC,DEST}
	src		::	SRC
	dest	::	DEST
end

# Our alternative for non-Float64 is to use ApproxFun's fft, at least for 1d.
# This allocates memory.
apply!(op::FastFourierTransform, dest, src, coef_dest, coef_src) = (coef_dest[:] =
	fft(coef_src)/sqrt(convert(eltype(coef_src),length(coef_src))))


immutable InverseFastFourierTransform{SRC,DEST} <: DiscreteFourierTransform{SRC,DEST}
	src		::	SRC
	dest	::	DEST
end

# Why was the below line necessary?
## apply!(op::InverseFastFourierTransform, dest, src, coef_dest::Array{Complex{BigFloat}}, coef_src::Array{Complex{BigFloat}}) = (coef_dest[:] = ifft(coef_src) )
apply!(op::InverseFastFourierTransform, dest, src, coef_dest, coef_src) =
	coef_dest[:] = ifft(coef_src) * sqrt(convert(eltype(coef_src),length(coef_src)))

ctranspose(op::FastFourierTransform) = InverseFastFourierTransform(dest(op), src(op))
ctranspose(op::FastFourierTransformFFTW) = InverseFastFourierTransformFFTW(dest(op), src(op))

ctranspose(op::InverseFastFourierTransform) = FastFourierTransform(dest(op), src(op))
ctranspose(op::InverseFastFourierTransformFFTW) = FastFourierTransformFFTW(dest(op), src(op))

inverse(op::DiscreteFourierTransform) = ctranspose(op)


transform_operator{G <: PeriodicEquispacedGrid}(src::DiscreteGridSpace{G}, dest::FourierBasis) =
	_forward_fourier_operator(src, dest, eltype(src,dest))

_forward_fourier_operator(src::DiscreteGridSpace, dest::FourierBasis, ::Type{Complex{Float64}}) =
	FastFourierTransformFFTW(src,dest)

_forward_fourier_operator{T <: AbstractFloat}(src::DiscreteGridSpace, dest::FourierBasis, ::Type{Complex{T}}) =
	FastFourierTransform(src,dest)


transform_operator{G <: PeriodicEquispacedGrid}(src::FourierBasis, dest::DiscreteGridSpace{G}) =
	_backward_fourier_operator(src, dest, eltype(src,dest))

_backward_fourier_operator(src::FourierBasis, dest::DiscreteGridSpace, ::Type{Complex{Float64}}) =
	InverseFastFourierTransformFFTW(src,dest)

_backward_fourier_operator{T <: AbstractFloat}(src::FourierBasis, dest::DiscreteGridSpace, ::Type{Complex{T}}) =
	InverseFastFourierTransform(src, dest)



evaluation_operator(b::FourierBasis) = transform_operator(b, grid(b))

function transform_normalization_operator(src::FourierBasis)
	L = length(src)
	ELT = eltype(src)
	ScalingOperator(src, 1/sqrt(ELT(L)))
end

