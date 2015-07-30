# chebyshevbasis.jl


# Fourier basis on the interval [a,b]
immutable ChebyshevBasis{T <: FloatingPoint} <: AbstractBasis1d{T}
    n                   ::      Int
    a 			::	T
    b 			::	T
    # I don't see why grid is necessary for a basis..
	# grid		::	PeriodicEquispacedGrid{T}
end

name(b::ChebyshevBasis) = "Chebyshev series"

isreal(b::ChebyshevBasis) = True()
isreal{B <: ChebyshevBasis}(::Type{B}) = True()

	
ChebyshevBasis{T}(n, a::T = -1.0, b::T = 1.0) = ChebyshevBasis{T}(n, a, b)

length(b::ChebyshevBasis) = n

left(b::ChebyshevBasis) = b.a

left(b::ChebyshevBasis, idx) = b.a

right(b::ChebyshevBasis) = b.b

right(b::ChebyshevBasis, idx) = b.b

grid{T}(b::ChebyshevBasis{T}) = ChebyshevIIGrid(n,T)


# Map the point x in [a,b] to the corresponding point in [-1,1]
mapx(b::ChebyshevBasis, x) = (x-b.a)/(b.b-b.a)*2-1

# One has to be careful here not to match Floats and BigFloats by accident.
# Hence the conversions to T in the lines below.
call{T <: FloatingPoint}(b::ChebyshevBasis{T}, idx::Int, x::T) = cos(one(T)*idx*acos(mapx(x)))

call{T, S <: Number}(b::ChebyshevBasis{T}, idx::Int, x::S) = call(b, idx, T(x))

function apply!(op::Extension, dest::ChebyshevBasis, src::ChebyshevBasis, coef_dest, coef_src)
	@assert length(dest) > length(src)

	for i=1:length(src)
		coef_dest[i] = coef_src[i]
	end
	for i=length(src)+1:length(dest)
		coef_dest[i] = 0
	end
end


function apply!(op::Restriction, dest::ChebyshevBasisOdd, src::ChebyshevBasis, coef_dest, coef_src)
	@assert length(dest) < length(src)

	for i=1:length(dest)
		coef_dest[i] = coef_src[i]
	end
end



abstract DiscreteChebyshevTransform{SRC,DEST} <: AbstractOperator{SRC,DEST}

abstract DiscreteChebyshevTransformFFTW{SRC,DEST} <: DiscreteChebyshevTransform{SRC,DEST}

# These types use FFTW and so they are (currently) limited to Float64.
# This will improve once the pure-julia implementation of FFT lands (#6193).
# But, we can also borrow from ApproxFun so let's do that right away

is_inplace(op::DiscreteChebyshevTransformFFTW) = True()


immutable FastChebyshevTransformFFTW{SRC,DEST} <: DiscreteChebyshevTransformFFTW{SRC,DEST}
	src		::	SRC
	dest	::	DEST
	plan!	::	Base.DFT.FFTW.cFFTWPlan

	FastChebyshevTransformFFTW(src, dest) = new(src, dest, plan_dct!(zeros(eltype(dest),size(dest)), 1:dim(dest); flags= FFTW.ESTIMATE|FFTW.MEASURE|FFTW.PATIENT))
end

FastChebyshevTransformFFTW{SRC,DEST}(src::SRC, dest::DEST) = FastChebyshevTransformFFTW{SRC,DEST}(src, dest)

immutable InverseFastChebyshevTransformFFTW{SRC,DEST} <: DiscreteChebyshevTransformFFTW{SRC,DEST}
	src		::	SRC
	dest	::	DEST
	plan!	::	Base.DFT.FFTW.cFFTWPlan

	InverseFastChebyshevTransformFFTW(src, dest) = new(src, dest, plan_idct!(zeros(eltype(src),size(src)), 1:dim(src); flags= FFTW.ESTIMATE|FFTW.MEASURE|FFTW.PATIENT))
end

InverseFastChebyshevTransformFFTW{SRC,DEST}(src::SRC, dest::DEST) = InverseFastChebyshevTransformFFTW{SRC,DEST}(src, dest)

# One implementation for forward and inverse transform in-place: call the plan
apply!(op::DiscreteChebyshevTransformFFTW, dest, src, coef_srcdest) = op.plan!*coef_srcdest


immutable FastChebyshevTransform{SRC,DEST} <: DiscreteChebyshevTransform{SRC,DEST}
	src		::	SRC
	dest	::	DEST
end

# Our alternative for non-Float64 is to use ApproxFun's fft, at least for 1d.
# This allocates memory.
apply!(op::FastChebyshevTransform, dest, src, coef_dest, coef_src) = (coef_dest[:] = dct(coef_src))


immutable InverseFastChebyshevTransform{SRC,DEST} <: DiscreteChebyshevTransform{SRC,DEST}
	src		::	SRC
	dest	::	DEST
end

apply!(op::InverseFastChebyshevTransform, dest, src, coef_dest::Array{Complex{BigFloat}}, coef_src::Array{Complex{BigFloat}}) = (coef_dest[:] = idct(coef_src) * length(coef_src))

AnyFourierBasis = Union{ChebyshevBasis, TensorProductSet}
AnyDiscreteGridSpace = Union{DiscreteGridSpace, TensorProductSet}
# Defined in fourierbasis
#transform_operator(src::TensorProductSet,dest::TensorProductSet) = transform_operator(src,dest,sets(src),sets(dest))
transform_operator{N}(src::TensorProductSet,dest::TensorProductSet, srcsets::NTuple{N,ChebyshevBasis},destsets::NTuple{N,DiscreteGridSpace}) = _backward_chebyshev_operator(src,dest,eltype(src,dest))
transform_operator{N}(src::TensorProductSet,dest::TensorProductSet, srcsets::NTuple{N,DiscreteGridSpace},destsets::NTuple{N,ChebyshevBasis}) = _forward_chebyshev_operator(src,dest,eltype(src,dest))

# For the default Chebyshev transform, we have to distinguish (for the time being) between the version for Float64 and other types (like BigFloat)
transform_operator(src::DiscreteGridSpace, dest::ChebyshevBasis) = _forward_chebyshev_operator(src, dest, eltype(src,dest))

_transform_operator(src::DiscreteGridSpace, dest::ChebyshevBasis, ::Type{Complex{Float64}}) = FastChebyshevTransformFFTW(src,dest)

_forward_chebyshev_operator{T <: FloatingPoint}(src::DiscreteGridSpace, dest::ChebyshevBasis, ::Type{Complex{T}}) = FastChebyshevTransform(src,dest)



transform_operator(src::ChebyshevBasis, dest::DiscreteGridSpace) = _backward_chebyshev_operator(src, dest, eltype(src,dest))

_backward_chebyshev_operator(src::ChebyshevBasis, dest::DiscreteGridSpace, ::Type{Complex{Float64}}) = InverseFastChebyshevTransformFFTW(src,dest)

_backward_chebyshev_operator{T <: FloatingPoint}(src::ChebyshevBasis, dest::DiscreteGridSpace, ::Type{Complex{T}}) = InverseFastChebyshevTransform(src, dest)






