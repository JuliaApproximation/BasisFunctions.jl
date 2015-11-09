# chebyshevbasis.jl


############################################
# Chebyshev polynomials of the first kind
############################################


# A basis of Chebyshev polynomials of the first kind on the interval [a,b]
immutable ChebyshevBasis{T <: AbstractFloat} <: OPS{T}
    n			::	Int
    a 			::	T
    b 			::	T
end

typealias ChebyshevBasisFirstKind{T} ChebyshevBasis{T}


name(b::ChebyshevBasis) = "Chebyshev series (first kind)"

isreal(b::ChebyshevBasis) = True()
isreal{B <: ChebyshevBasis}(::Type{B}) = True()

	
ChebyshevBasis{T}(n, a::T = -1.0, b::T = 1.0) = ChebyshevBasis{T}(n, a, b)

left(b::ChebyshevBasis) = b.a
left(b::ChebyshevBasis, idx) = left(b)

right(b::ChebyshevBasis) = b.b
right(b::ChebyshevBasis, idx) = right(b)

grid{T}(b::ChebyshevBasis{T}) = LinearMappedGrid(ChebyshevIIGrid{T}(b.n), left(b), right(b))

# The weight function
weight(b::ChebyshevBasis, x) = 1/sqrt(1-x^2)

# Parameters alpha and beta of the corresponding Jacobi polynomial
jacobi_alpha{T}(b::ChebyshevBasis{T}) = -one(T)/2
jacobi_beta{T}(b::ChebyshevBasis{T}) = -one(T)/2


# See DLMF, Table 18.9.1
# http://dlmf.nist.gov/18.9#i
rec_An(b::ChebyshevBasis, n::Int) = n==0 ? 1 : 2

rec_Bn(b::ChebyshevBasis, n::Int) = 0

rec_Cn(b::ChebyshevBasis, n::Int) = 1


# Map the point x in [a,b] to the corresponding point in [-1,1]
mapx(b::ChebyshevBasis, x) = (x-b.a)/(b.b-b.a)*2-1

call{T <: AbstractFloat}(b::ChebyshevBasis{T}, idx::Int, x::T) = cos((idx-1)*acos(mapx(b,x)))
call{T <: AbstractFloat}(b::ChebyshevBasis{T}, idx::Int, x::Complex{T}) = cos((idx-1)*acos(mapx(b,x)))


function apply!(op::Extension, dest::ChebyshevBasis, src::ChebyshevBasis, coef_dest, coef_src)
	@assert length(dest) > length(src)

	for i=1:length(src)
		coef_dest[i] = coef_src[i]
	end
	for i=length(src)+1:length(dest)
		coef_dest[i] = 0
	end
end


function apply!(op::Restriction, dest::ChebyshevBasis, src::ChebyshevBasis, coef_dest, coef_src)
	@assert length(dest) < length(src)

	for i=1:length(dest)
		coef_dest[i] = coef_src[i]
	end
end


function differentiation_matrix{T}(src::ChebyshevBasis{T})
	n = length(src)
	N = n-1
	D = zeros(T, N+1, N+1)
	A = zeros(1, N+1)
	B = zeros(1, N+1)
	B[N+1] = 2*N
	D[N,:] = B
	for k = N-1:-1:1
		C = A
		C[k+1] = 2*k
		D[k,:] = C
		A = B
		B = C
	end
	D[1,:] = D[1,:]/2
	D
end

function apply!{T}(op::Differentiation, dest::ChebyshevBasis{T}, src::ChebyshevBasis{T}, coef_dest, coef_src)
	D = differentiation_matrix(src)
	coef_dest[:] = D*coef_src
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
	plan!	::	Base.DFT.FFTW.DCTPlan

	FastChebyshevTransformFFTW(src, dest) = new(src, dest, plan_dct!(zeros(eltype(dest),size(dest)), 1:dim(dest); flags= FFTW.ESTIMATE|FFTW.MEASURE|FFTW.PATIENT))
end

FastChebyshevTransformFFTW{SRC,DEST}(src::SRC, dest::DEST) = FastChebyshevTransformFFTW{SRC,DEST}(src, dest)

immutable InverseFastChebyshevTransformFFTW{SRC,DEST} <: DiscreteChebyshevTransformFFTW{SRC,DEST}
	src		::	SRC
	dest	::	DEST
	plan!	::	Base.DFT.FFTW.DCTPlan

	InverseFastChebyshevTransformFFTW(src, dest) = new(src, dest, plan_idct!(zeros(eltype(src),size(src)), 1:dim(src); flags= FFTW.ESTIMATE|FFTW.MEASURE|FFTW.PATIENT))
end

InverseFastChebyshevTransformFFTW{SRC,DEST}(src::SRC, dest::DEST) = InverseFastChebyshevTransformFFTW{SRC,DEST}(src, dest)

# One implementation for forward and inverse transform in-place: call the plan. Added constant to undo the normalisation.
apply!(op::DiscreteChebyshevTransformFFTW, dest, src, coef_srcdest) = sqrt(length(dest)/2^(dim(src)))*op.plan!*coef_srcdest


immutable FastChebyshevTransform{SRC,DEST} <: DiscreteChebyshevTransform{SRC,DEST}
	src		::	SRC
	dest	::	DEST
end

# Our alternative for non-Float64 is to use ApproxFun's fft, at least for 1d.
# This allocates memory.
apply!(op::FastChebyshevTransform, dest, src, coef_dest, coef_src) = (coef_dest[:] = dct(coef_src)*sqrt(length(dest))/2^(dim(src)))


immutable InverseFastChebyshevTransform{SRC,DEST} <: DiscreteChebyshevTransform{SRC,DEST}
	src		::	SRC
	dest	::	DEST
end

apply!(op::InverseFastChebyshevTransform, dest, src, coef_dest::Array{Complex{BigFloat}}, coef_src::Array{Complex{BigFloat}}) = (coef_dest[:] = idct(coef_src) * sqrt(length(dest))/2^(dim(src)))

AnyFourierBasis = Union{ChebyshevBasis, TensorProductSet}
AnyDiscreteGridSpace = Union{DiscreteGridSpace, TensorProductSet}
# Defined in fourierbasis
#transform_operator(src::TensorProductSet,dest::TensorProductSet) = transform_operator(src,dest,sets(src),sets(dest))
transform_operator{N}(src::TensorProductSet,dest::TensorProductSet, srcsets::NTuple{N,ChebyshevBasis},destsets::NTuple{N,DiscreteGridSpace}) = _backward_chebyshev_operator(src,dest,eltype(src,dest))
transform_operator{N}(src::TensorProductSet,dest::TensorProductSet, srcsets::NTuple{N,DiscreteGridSpace},destsets::NTuple{N,ChebyshevBasis}) = _forward_chebyshev_operator(src,dest,eltype(src,dest))

# For the default Chebyshev transform, we have to distinguish (for the time being) between the version for Float64 and other types (like BigFloat)
transform_operator(src::DiscreteGridSpace, dest::ChebyshevBasis) = _forward_chebyshev_operator(src, dest, eltype(src,dest))

_forward_chebyshev_operator(src::AnyDiscreteGridSpace, dest::AnyChebyshevBasis, ::Type{Complex{Float64}}) = FastChebyshevTransformFFTW(src,dest)

_forward_chebyshev_operator{T <: AbstractFloat}(src::AnyDiscreteGridSpace, dest::AnyChebyshevBasis, ::Type{Complex{T}}) = FastChebyshevTransform(src,dest)



transform_operator(src::ChebyshevBasis, dest::DiscreteGridSpace) = _backward_chebyshev_operator(src, dest, eltype(src,dest))

_backward_chebyshev_operator(src::AnyChebyshevBasis, dest::AnyDiscreteGridSpace, ::Type{Complex{Float64}}) = InverseFastChebyshevTransformFFTW(src,dest)

_backward_chebyshev_operator{T <: AbstractFloat}(src::AnyChebyshevBasis, dest::AnyDiscreteGridSpace, ::Type{Complex{T}}) = InverseFastChebyshevTransform(src, dest)



############################################
# Chebyshev polynomials of the second kind
############################################

# A basis of Chebyshev polynomials of the second kind (on the interval [-1,1])
immutable ChebyshevBasisSecondKind{T <: AbstractFloat} <: OPS{T}
    n			::	Int

    ChebyshevBasisSecondKind(n) = new(n)
end

ChebyshevBasisSecondKind(n) = ChebyshevBasisSecondKind{Float64}(n)

name(b::ChebyshevBasisSecondKind) = "Chebyshev series (second kind)"

isreal(b::ChebyshevBasisSecondKind) = True()
isreal{B <: ChebyshevBasisSecondKind}(::Type{B}) = True()


left{T}(b::ChebyshevBasisSecondKind{T}) = -one(T)
left{T}(b::ChebyshevBasisSecondKind{T}, idx) = left(b)

right{T}(b::ChebyshevBasisSecondKind{T}) = one(T)
right{T}(b::ChebyshevBasisSecondKind{T}, idx) = right(b)

grid{T}(b::ChebyshevBasisSecondKind{T}) = ChebyshevIIGrid{T}(b.n)


# The weight function
weight(b::ChebyshevBasisSecondKind, x) = sqrt(1-x^2)

# Parameters alpha and beta of the corresponding Jacobi polynomial
jacobi_alpha{T}(b::ChebyshevBasisSecondKind{T}) = one(T)/2
jacobi_beta{T}(b::ChebyshevBasisSecondKind{T}) = one(T)/2


# See DLMF, Table 18.9.1
# http://dlmf.nist.gov/18.9#i
rec_An(b::ChebyshevBasisSecondKind, n::Int) = 2

rec_Bn(b::ChebyshevBasisSecondKind, n::Int) = 0

rec_Cn(b::ChebyshevBasisSecondKind, n::Int) = 1



