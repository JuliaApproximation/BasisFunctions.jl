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



isreal(b::FourierBasis) = False
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

natural_grid(b::FourierBasis) = b.grid


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

call{T <: Number}(b::FourierBasisOdd, idx::Int, x::T) = exp(2 * pi * 1im * mapx(b, x) * frequency(b, idx))

call{T,S <: Number}(b::FourierBasisEven{T}, idx::Int, x::S) = (idx == length(b)/2+1 ? one(Complex{T}) * cos(2 * pi * mapx(b, x) * frequency(b, idx)) : exp(2 * pi * 1im * mapx(b, x) * frequency(b, idx)))



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



immutable FastFourierTransform{SRC,DEST} <: AbstractDiscreteTransform{SRC,DEST}
	src		::	SRC
	dest	::	DEST
end


immutable InverseFastFourierTransform{SRC,DEST} <: AbstractDiscreteTransform{SRC,DEST}
	src		::	SRC
	dest	::	DEST
end


function apply!{SRC,DEST <: FourierBasis}(op::FastFourierTransform{SRC,DEST}, coef_dest, coef_src)
	n = length(dest(op))
	coef_dest[:] = coef_src[:]/n
	fft!(coef_dest)
end


function apply!{SRC,DEST <: FourierBasisNd}(op::FastFourierTransform{SRC,DEST}, coef_dest, coef_src)
	n = length(dest(op))
	coef_dest[:] = coef_src[:]/n
	coef_dest = reshape(coef_dest, size(dest(op)))
	fft!(coef_dest)
end


function apply!{SRC <: FourierBasis,DEST}(op::InverseFastFourierTransform{SRC,DEST}, coef_dest, coef_src)
	coef_dest[:] = coef_src[:]
	bfft!(coef_dest) 	# bfft is an unscaled inverse fft, which is what we need here
end


#function apply!{SRC <: FourierBasisNd, DEST}(op::InverseFastFourierTransform{SRC,DEST}, coef_dest, coef_src)
function apply!{SRC, DEST}(op::InverseFastFourierTransform{SRC,DEST}, coef_dest, coef_src)
	coef_dest[:] = coef_src[:]
	coef_dest = reshape(coef_dest, size(src(op)))
	bfft!(coef_dest)
end


