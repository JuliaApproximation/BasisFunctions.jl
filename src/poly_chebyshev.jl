# poly_chebyshev.jl


############################################
# Chebyshev polynomials of the first kind
############################################


"""
A basis of Chebyshev polynomials of the first kind on the interval [a,b].
"""
immutable ChebyshevBasis{T <: AbstractFloat} <: OPS{T}
    n			::	Int
    a 			::	T
    b 			::	T

    ChebyshevBasis(n, a = -one(T), b = one(T)) = new(n, a, b)
end

typealias ChebyshevBasisFirstKind{T} ChebyshevBasis{T}


name(b::ChebyshevBasis) = "Chebyshev series (first kind)"

	
ChebyshevBasis{T}(n, a::T, b::T) = ChebyshevBasis{T}(n, a, b)

ChebyshevBasis{T}(n, ::Type{T} = Float64) = ChebyshevBasis{T}(n)

instantiate{T}(::Type{ChebyshevBasis}, n, ::Type{T}) = ChebyshevBasis{T}(n)

has_grid(b::ChebyshevBasis) = true

has_derivative(b::ChebyshevBasis) = true

left(b::ChebyshevBasis) = b.a
left(b::ChebyshevBasis, idx) = left(b)

right(b::ChebyshevBasis) = b.b
right(b::ChebyshevBasis, idx) = right(b)

grid{T}(b::ChebyshevBasis{T}) = LinearMappedGrid(ChebyshevIIGrid{T}(b.n), left(b), right(b))

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
mapx(b::ChebyshevBasis, x) = (x-b.a)/(b.b-b.a)*2-1

call{T <: AbstractFloat}(b::ChebyshevBasis{T}, idx::Int, x::T) = cos((idx-1)*acos(mapx(b,x)))
call{T <: AbstractFloat}(b::ChebyshevBasis{T}, idx::Int, x::Complex{T}) = cos((idx-1)*acos(mapx(b,x)))


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

	for i=1:length(dest)
		coef_dest[i] = coef_src[i]
	end
end

# TODO: this matrix does not take into account a and b -> either remove a and b (in favour of mapped basis) or update this function
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

# TODO: update in order to avoid memory allocation in constructing the differentiation_matrix
# Would be better to write differentiation_matrix in terms of apply! (can be generic), rather than the other way around
function apply!{T}(op::Differentiation, dest::ChebyshevBasis{T}, src::ChebyshevBasis{T}, coef_dest, coef_src)
	D = differentiation_matrix(src)
	coef_dest[:] = D*coef_src
end

has_derivative(b::ChebyshevBasis) = true

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
# apply!(op::DiscreteChebyshevTransformFFTW, dest, src, coef_srcdest) = sqrt(length(dest)/2^(dim(src)))*op.plan!*coef_srcdest
function apply!(op::FastChebyshevTransformFFTW, dest, src, coef_srcdest)
    op.plan!*coef_srcdest
    ## for i=1:Integer(round(length(coef_srcdest)/2))
    ##     temp = coef_srcdest[i]
    ##     coef_srcdest[i]=coef_srcdest[end-i+1]
    ##     coef_srcdest[end-i+1]=temp
    ## end
end

function apply!(op::InverseFastChebyshevTransformFFTW, dest, src, coef_srcdest)
    op.plan!*coef_srcdest
    ## for i=1:length(coef_srcdest)
    ##     coef_srcdest[i]/=length(coef_srcdest)
    ## end
end


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



# TODO: restrict the grid of grid space here
transform_operator(src::DiscreteGridSpace, dest::ChebyshevBasis) = _forward_chebyshev_operator(src, dest, eltype(src,dest))

_forward_chebyshev_operator(src::DiscreteGridSpace, dest::ChebyshevBasis, ::Type{Float64}) = FastChebyshevTransformFFTW(src,dest)

_forward_chebyshev_operator{T <: AbstractFloat}(src::DiscreteGridSpace, dest::ChebyshevBasis, ::Type{T}) = FastChebyshevTransform(src,dest)



transform_operator(src::ChebyshevBasis, dest::DiscreteGridSpace) = _backward_chebyshev_operator(src, dest, eltype(src,dest))

_backward_chebyshev_operator(src::ChebyshevBasis, dest::DiscreteGridSpace, ::Type{Float64}) = InverseFastChebyshevTransformFFTW(src,dest)

_backward_chebyshev_operator{T <: AbstractFloat}(src::ChebyshevBasis, dest::DiscreteGridSpace, ::Type{T}) = InverseFastChebyshevTransform(src, dest)


# TODO: experimentally, this composite operator looks like it does the trick, up to an alternating flip.
# Below let's implement a custom type to do the alternating flip. But this should be fixed.
#approximation_operator(b::ChebyshevBasis) = CoefficientScalingOperator(b, 1, 1/sqrt(2)) * ScalingOperator(b, 1/sqrt(length(b)/2)) * transform_operator(b, grid(b))

immutable ChebyshevEvaluation{SRC,DEST,OP} <: AbstractOperator{SRC,DEST}
    src     ::  SRC
    dest    ::  DEST
    op      ::  OP
end

function apply!(op::ChebyshevEvaluation, dest, src, coef_dest, coef_src)
    apply!(op.op, coef_dest, coef_src)
    for i = 1:length(coef_dest)
        coef_dest[i] = (-1)^(i+1) * coef_dest[i]
    end
end

function approximation_operator(b::ChebyshevBasis)
    g = grid(b)
    op = CoefficientScalingOperator(b, 1, 1/sqrt(2)) * ScalingOperator(b, 1/sqrt(length(b)/2)) * transform_operator(b, grid(b))
    ChebyshevEvaluation(b, g, op)
end




############################################
# Chebyshev polynomials of the second kind
############################################

"A basis of Chebyshev polynomials of the second kind (on the interval [-1,1])."
immutable ChebyshevBasisSecondKind{T <: AbstractFloat} <: OPS{T}
    n			::	Int
end

ChebyshevBasisSecondKind{T}(n, ::Type{T} = Float64) = ChebyshevBasisSecondKind{T}(n)

instantiate{T}(::Type{ChebyshevBasisSecondKind}, n, ::Type{T}) = ChebyshevBasisSecondKind{T}(n)

name(b::ChebyshevBasisSecondKind) = "Chebyshev series (second kind)"

isreal(b::ChebyshevBasisSecondKind) = True()
isreal{B <: ChebyshevBasisSecondKind}(::Type{B}) = True


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



