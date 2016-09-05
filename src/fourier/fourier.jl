# fourier.jl

"""
A Fourier basis on the interval [0,1]. The precise basis functions are:
exp(2 π i k)
with k ranging from -N to N for Fourier series of odd length 2N+1.

The basis functions are ordered the way they are expected by a typical FFT
implementation. The frequencies k are in the following order:
0 1 2 3 ... N -N -N+1 ... -2 -1

Parameter EVEN is true if the length of the corresponding Fourier series is
even. In that case, the largest frequency function in the set is a cosine.
"""
immutable FourierBasis{EVEN,T} <: FunctionSet1d{T}
	n			::	Int

	FourierBasis(n) = (@assert iseven(n)==EVEN; new(n))
end

typealias FourierBasisEven{T} FourierBasis{true,T}
typealias FourierBasisOdd{T} FourierBasis{false,T}

name(b::FourierBasis) = "Fourier series"

# The Element Type of a Fourier Basis is complex by definition. Real types are complexified.
FourierBasis{T}(n, ::Type{T} = Float64) = FourierBasis{iseven(n),complexify(floatify(T))}(n)

FourierBasis{T}(n, a, b, ::Type{T} = promote_type(typeof(a),typeof(b))) = rescale(FourierBasis(n, T), a, b)

# Typesafe methods for constructing a Fourier series with even length
fourier_basis_even{T}(n, ::Type{T}) = FourierBasis{true,T}(n)

# Typesafe method for constructing a Fourier series with odd length
fourier_basis_odd{T}(n, ::Type{T}) = FourierBasis{false,T}(n)


instantiate{T}(::Type{FourierBasis}, n, ::Type{T}) = FourierBasis(n, T)

promote_eltype{EVEN,T,S}(b::FourierBasis{EVEN,T}, ::Type{S}) = FourierBasis{EVEN,promote_type(T,S)}(b.n)

resize(b::FourierBasis, n) = FourierBasis(n, eltype(b))


# Properties

isreal(b::FourierBasis) = false

iseven{EVEN}(b::FourierBasis{EVEN}) = EVEN
isodd(b::FourierBasis) = ~iseven(b)

is_basis(b::FourierBasis) = true
is_orthogonal(b::FourierBasis) = true


# Methods for purposes of testing functionality.
has_grid(b::FourierBasis) = true
has_derivative(b::FourierBasis) = true
# Until adapted for DC coefficient
has_antiderivative(b::FourierBasis) = false
has_transform{G <: PeriodicEquispacedGrid}(b::FourierBasis, d::DiscreteGridSpace{G}) = true
has_extension(b::FourierBasis) = true


length(b::FourierBasis) = b.n

left(b::FourierBasis) = 0

left(b::FourierBasis, idx) = left(b)

right(b::FourierBasis) = 1

right(b::FourierBasis, idx) = right(b)

period(b::FourierBasis) = 1

grid(b::FourierBasis) = PeriodicEquispacedGrid(b.n, 0, 1, numtype(b))

nhalf(b::FourierBasis) = length(b)>>1


# The frequency of an even Fourier basis ranges from -N+1 to N.
idx2frequency(b::FourierBasisEven, idx) = idx <= nhalf(b)+1 ? idx-1 : idx - 2*nhalf(b) - 1

# The frequency of an odd Fourier basis ranges from -N to N.
idx2frequency(b::FourierBasisOdd, idx::Int) = idx <= nhalf(b)+1 ? idx-1 : idx - 2*nhalf(b) - 2

frequency2idx(b::FourierBasis, freq) = freq >= 0 ? freq+1 : length(b)+freq+1

# The native index of a FourierBasis is the frequency. Since that is an integer,
# it is wrapped in a different type.
immutable FourierFrequency <: NativeIndex
	index	::	Int
end
native_index(b::FourierBasis, idx::Int) = FourierFrequency(idx2frequency(b, idx))
linear_index(b::FourierBasis, idxn::NativeIndex) = frequency2idx(b, index(idxn))

# One has to be careful here not to match Floats and BigFloats by accident.
# Hence the conversions to T in the lines below.
call_element{T, S <: Number}(b::FourierBasisOdd{T}, idx::Int, x::S) = exp(x * 2 * T(pi) * 1im  * idx2frequency(b, idx))

# Note that the function below is typesafe because T(pi) converts pi to a complex number, hence the cosine returns a complex number
call_element{T, S <: Number}(b::FourierBasisEven{T}, idx::Int, x::S) =
	(idx == nhalf(b)+1	?  cos(x * 2 * T(pi) * idx2frequency(b,idx))
						: exp(x * 2 * T(pi) * 1im * idx2frequency(b,idx)))

moment{EVEN,T}(b::FourierBasis{EVEN,T}, idx) = idx == 1 ? T(2) : T(0)

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
	result
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
	result
end

extension_size(b::FourierBasisEven) = 2*length(b)
extension_size(b::FourierBasisOdd) = 2*length(b)+1

approx_length(b::FourierBasisEven, n::Int) = iseven(n) ? n : n+1
approx_length(b::FourierBasisOdd, n::Int) = isodd(n) ? n : n+1


function apply!(op::Extension, dest::FourierBasis, src::FourierBasisEven, coef_dest, coef_src)
	## @assert length(dest) > length(src)

	nh = nhalf(src)

	for i = 0:nh-1
		coef_dest[i+1] = coef_src[i+1]
	end
	for i = 1:nh-1
		coef_dest[end-nh+i+1] = coef_src[nh+1+i]
	end
	coef_dest[nh+1] = coef_src[nh+1]/2
	coef_dest[end-nh+1] = coef_src[nh+1]/2
	for i = nh+2:length(coef_dest)-nh
		coef_dest[i] = 0
	end
	coef_dest
end

function apply!(op::Extension, dest::FourierBasis, src::FourierBasisOdd, coef_dest, coef_src)
	## @assert length(dest) > length(src)

	nh = nhalf(src)

	for i = 0:nh
		coef_dest[i+1] = coef_src[i+1]
	end
	for i = 1:nh
		coef_dest[end-nh+i] = coef_src[nh+1+i]
	end
	for i = nh+2:length(coef_dest)-nh
		coef_dest[i] = 0
	end
	coef_dest
end


function apply!(op::Restriction, dest::FourierBasisOdd, src::FourierBasis, coef_dest, coef_src)
	## @assert length(dest) < length(src)

	nh = nhalf(dest)
	for i = 0:nh
		coef_dest[i+1] = coef_src[i+1]
	end
	for i = 1:nh
		coef_dest[nh+1+i] = coef_src[end-nh+i]
	end
	coef_dest
end

function apply!(op::Restriction, dest::FourierBasisEven, src::FourierBasis, coef_dest, coef_src)
	## @assert length(dest) < length(src)

	nh = nhalf(dest)
	for i = 0:nh-1
		coef_dest[i+1] = coef_src[i+1]
	end
	for i = 1:nh-1
		coef_dest[nh+1+i] = coef_src[end-nh+i+1]
	end
	coef_dest[nh+1] = coef_src[nh+1] + coef_src[end-nh+1]
	coef_dest
end

derivative_set(b::FourierBasisOdd, order) = b

# We extend the even basis both for derivation and antiderivation, regardless of order
for op in (:derivative_set, :antiderivative_set)
    @eval $op(b::FourierBasisEven, order::Int; options...) = fourier_basis_odd(length(b)+1,eltype(b))
end

for op in (:differentiation_operator, :antidifferentiation_operator)
    @eval function $op(b::FourierBasisEven, b_odd::FourierBasisOdd, order::Int; options...)
        $op(b_odd, b_odd, order; options...) * extension_operator(b, b_odd; options...)
    end
end

# Both differentiation and antidifferentiation are diagonal operations
diff_scaling_function{T}(b::FourierBasisOdd{T}, idx, order) = (2 * T(pi) * im * idx2frequency(b,idx))^order
function differentiation_operator{T}(b1::FourierBasisOdd{T}, b2::FourierBasisOdd{T}, order::Int; options...)
	@assert length(b1) == length(b2)
	DiagonalOperator(b1, [diff_scaling_function(b1, idx, order) for idx in eachindex(b1)])
end

antidiff_scaling_function{T}(b::FourierBasisOdd{T}, idx, order) = idx==0 ? T(0) : 1 / (i* 2 * T(pi) * im)^order
function antidifferentiation_operator(b1::FourierBasisOdd, b2::FourierBasisOdd, order::Int; options...)
	@assert length(b1) == length(b2)
	DiagonalOperator(b1, [antidiff_scaling_function(b1, idx, order) for idx in eachindex(b1)])
end


transform_operator{G <: PeriodicEquispacedGrid}(src::DiscreteGridSpace{G}, dest::FourierBasis; options...) =
	_forward_fourier_operator(src, dest, eltype(src, dest); options...)

_forward_fourier_operator(src, dest, ::Type{Complex{Float64}}; options...) =
	FastFourierTransformFFTW(src, dest; options...)

_forward_fourier_operator{T <: AbstractFloat}(src, dest, ::Type{Complex{T}}; options...) =
	FastFourierTransform(src, dest)


transform_operator{G <: PeriodicEquispacedGrid}(src::FourierBasis, dest::DiscreteGridSpace{G}; options...) =
	_backward_fourier_operator(src, dest, eltype(src, dest); options...)

_backward_fourier_operator(src, dest, ::Type{Complex{Float64}}; options...) =
	InverseFastFourierTransformFFTW(src, dest; options...)

_backward_fourier_operator{T <: AbstractFloat}(src, dest, ::Type{Complex{T}}; options...) =
	InverseFastFourierTransform(src, dest)

# Catch 2D and 3D fft's automatically
transform_operator_tensor{G <: PeriodicEquispacedGrid}(src, dest,
	src_set1::DiscreteGridSpace{G}, src_set2::DiscreteGridSpace{G},
	dest_set1::FourierBasis, dest_set2::FourierBasis; options...) =
		_forward_fourier_operator(src, dest, eltype(src, dest); options...)

transform_operator_tensor{G <: PeriodicEquispacedGrid}(src, dest,
	src_set1::FourierBasis, src_set2::FourierBasis,
	dest_set1::DiscreteGridSpace{G}, dest_set2::DiscreteGridSpace{G}; options...) =
		_backward_fourier_operator(src, dest, eltype(src, dest); options...)

transform_operator_tensor{G <: PeriodicEquispacedGrid}(src, dest,
	src_set1::DiscreteGridSpace{G}, src_set2::DiscreteGridSpace{G}, src_set3::DiscreteGridSpace{G},
	dest_set1::FourierBasis, dest_set2::FourierBasis, dest_set3::FourierBasis; options...) =
		_forward_fourier_operator(src, dest, eltype(src, dest); options...)

transform_operator_tensor{G <: PeriodicEquispacedGrid}(src, dest,
	src_set1::FourierBasis, src_set2::FourierBasis, src_set3::FourierBasis,
	dest_set1::DiscreteGridSpace{G}, dest_set2::DiscreteGridSpace{G}, dest_set3::DiscreteGridSpace{G}; options...) =
		_backward_fourier_operator(src, dest, eltype(src, dest); options...)


function transform_normalization_operator(src::FourierBasis; options...)
    L = length(src)
    ELT = eltype(src)
    ScalingOperator(src, 1/sqrt(ELT(L)))
end

is_compatible(s1::FourierBasis, s2::FourierBasis) = true
# Multiplication of Fourier Series
function (*)(src1::FourierBasisOdd, src2::FourierBasisEven, coef_src1, coef_src2)
    dsrc2 = resize(src2,length(src2)+1)
    (*)(src1,dsrc2,coef_src1,extension_operator(src2,dsrc2)*coef_src2)
end

function (*)(src1::FourierBasisEven, src2::FourierBasisOdd, coef_src1, coef_src2)
    dsrc1 = resize(src1,length(src1)+1)
    (*)(dsrc1,src2,extension_operator(src1,dsrc1)*coef_src1,coef_src2)
end

function (*)(src1::FourierBasisEven, src2::FourierBasisEven, coef_src1, coef_src2)
    dsrc1 = resize(src1,length(src1)+1)
    dsrc2 = resize(src2,length(src2)+1)
    (*)(dsrc1,dsrc2,extension_operator(src1,dsrc1)*coef_src1,extension_operator(src2,dsrc2)*coef_src2)
end

function (*)(src1::FourierBasisOdd, src2::FourierBasisOdd, coef_src1, coef_src2)
    dest = FourierBasis(length(src1)+length(src2)-1,eltype(src1,src2))
    coef_src1 = [coef_src1[(nhalf(src1))+2:end]; coef_src1[1:nhalf(src1)+1]]
    coef_src2 = [coef_src2[(nhalf(src2))+2:end]; coef_src2[1:nhalf(src2)+1]]
    coef_dest = conv(coef_src1,coef_src2)
    coef_dest = [coef_dest[(nhalf(dest)+1):end]; coef_dest[1:(nhalf(dest))]]
    (dest,coef_dest)
end
