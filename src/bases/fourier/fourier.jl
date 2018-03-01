# fourier.jl

"""
A Fourier basis on the interval `[0,1]`. The precise basis functions are:
`exp(2 π i k)`
with `k` ranging from `-N` to `N` for Fourier series of odd length `2N+1`.

The basis functions are ordered the way they are expected by a typical FFT
implementation. The frequencies k are in the following order:
0 1 2 3 ... N -N -N+1 ... -2 -1

Parameter EVEN is true if the length of the corresponding Fourier series is
even. In that case, the largest frequency function in the set is a cosine.
"""
struct FourierBasis{T <: Real} <: FunctionSet{T}
	n			::	Int
end

const FourierSpan{A,F <: FourierBasis} = Span{A,F}

name(b::FourierBasis) = "Fourier series"

# The default numeric type is Float64
FourierBasis(n::Int) = FourierBasis{Float64}(n)

# Convenience constructor: map the Fourier series to the interval [a,b]
FourierBasis{T}(n, a::Number, b::Number) where {T} = rescale(FourierBasis{T}(n), a, b)

# We can deduce a candidate for T from a and b
function FourierBasis(n, a::Number, b::Number)
	T = float(promote_type(typeof(a),typeof(b)))
	FourierBasis{T}(n, a, b)
end


instantiate(::Type{FourierBasis}, n, ::Type{T}) where {T} = FourierBasis{T}(n)

set_promote_domaintype(b::FourierBasis{T}, ::Type{S}) where {T,S} = FourierBasis{S}(b.n)

resize(b::FourierBasis{T}, n) where {T} = FourierBasis{T}(n)

# The rangetype of a Fourier series is complex
rangetype(::Type{FourierBasis{T}}) where {T} = Complex{T}


# Properties

isreal(b::FourierBasis) = false

is_basis(b::FourierBasis) = true
is_orthogonal(b::FourierBasis) = true
is_orthonormal(b::FourierBasis) = oddlength(b)
is_biorthogonal(b::FourierBasis) = true

# Methods for purposes of testing functionality.
has_grid(b::FourierBasis) = true
has_derivative(b::FourierBasis) = true
# Until adapted for DC coefficient
has_antiderivative(b::FourierBasis) = false
has_extension(b::FourierBasis) = true

# For has_transform we introduce some more functionality:
# - Check whether the given periodic equispaced grid is compatible with the FFT operators
# 1+ because 0!≅eps()
compatible_grid(set::FourierBasis, grid::PeriodicEquispacedGrid) =
	(1+(left(set) - leftendpoint(grid))≈1) && (1+(right(set) - rightendpoint(grid))≈1) && (length(set)==length(grid))
# - Any non-periodic grid is not compatible
compatible_grid(set::FourierBasis, grid::AbstractGrid) = false
# - We have a transform if the grid is compatible
has_grid_transform(b::FourierBasis, gs, grid) = compatible_grid(b, grid)


length(b::FourierBasis) = b.n

oddlength(b::FourierBasis) = isodd(length(b))
evenlength(b::FourierBasis) = iseven(length(b))

left(b::FourierBasis) = zero(domaintype(b))
left(b::FourierBasis, idx) = left(b)

right(b::FourierBasis) = one(domaintype(b))
right(b::FourierBasis, idx) = right(b)

period(b::FourierBasis{T}) where {T} = T(1)

grid(b::FourierBasis) = PeriodicEquispacedGrid(b.n, left(b), right(b), domaintype(b))

nhalf(b::FourierBasis) = length(b)>>1


# The frequency of an even Fourier basis ranges from -N+1 to N.
# The frequency of an odd Fourier basis ranges from -N to N.
# This makes for a small difference in the ordering.
function idx2frequency(b::FourierBasis, idx)
	nh = nhalf(b)
	if idx <= nhalf(b)+1
		idx-1
	else
		if evenlength(b)
			idx-2*nh-1
		else
			idx-2*nh-2
		end
	end
end

frequency2idx(b::FourierBasis, freq) = freq >= 0 ? freq+1 : length(b)+freq+1

# The native index of a FourierBasis is the frequency. Since that is an integer,
# it is wrapped in a different type.
struct FourierFrequency <: NativeIndex
	index	::	Int
end
native_index(b::FourierBasis, idx::Int) = FourierFrequency(idx2frequency(b, idx))
linear_index(b::FourierBasis, idxn::NativeIndex) = frequency2idx(b, index(idxn))

# One has to be careful here not to match Floats and BigFloats by accident.
# Hence the conversions to T in the lines below.
function eval_element(b::FourierBasis, idx::Int, x)
	T = domaintype(b)
	k = idx2frequency(b, idx)
	if oddlength(b) || idx != nhalf(b)+1
		# Even-length Fourier series have a cosine at the maximal frequency
		exp(x * 2 * T(pi) * 1im  * k)
	else
		# We convert to a complex number for type safety here, because the
		# exponentials above are complex-valued but the cosine is real
		Complex{T}(cos(x * 2 * T(pi) * k))
	end
end

function eval_element_derivative(b::FourierBasis, idx::Int, x)
	# The structure for this reason is similar to the eval_element routine above
	T = domaintype(b)
	k = idx2frequency(b, idx)
	if oddlength(b) || idx != nhalf(b)+1
		arg = 2*T(pi)*1im*k
		arg * exp(arg * x)
	else
		arg = 2*T(pi)*k
		Complex{T}(-arg * sin(arg*x))
	end
end

function moment(b::FourierBasis, idx)
	T = rangetype(b)
	idx == 1 ? T(2) : T(0)
end

extension_size(b::FourierBasis) = evenlength(b) ? 2*length(b) : 2*length(b)+1

approx_length(b::FourierBasis, n::Int) = n

"Shift an expansion to the right by delta."
function shift(b::FourierBasis, coefficients, delta)
	# Only works for odd-length Fourier series for now
	@assert oddlength(b)
	T = domaintype(b)
	coef2 = copy(coefficients)
	for i in eachindex(coefficients)
		coef2[i] *= exp(2 * T(pi) * im * idx2frequency(b, i) * delta)
	end
	coef2
end

function apply!(op::Extension, dest::FourierBasis, src::FourierBasis, coef_dest, coef_src)
	## @assert length(dest) > length(src)

	nh = nhalf(src)
	# We have to distinguish between even and odd length Fourier series
	# because the memory layout is slightly different
	if evenlength(src)
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
	else
		for i = 0:nh
			coef_dest[i+1] = coef_src[i+1]
		end
		for i = 1:nh
			coef_dest[end-nh+i] = coef_src[nh+1+i]
		end
		for i = nh+2:length(coef_dest)-nh
			coef_dest[i] = 0
		end
	end
	coef_dest
end


function apply!(op::Restriction, dest::FourierBasis, src::FourierBasis, coef_dest, coef_src)
	## @assert length(dest) < length(src)

	nh = nhalf(dest)
	if oddlength(dest)
		for i = 0:nh
			coef_dest[i+1] = coef_src[i+1]
		end
		for i = 1:nh
			coef_dest[nh+1+i] = coef_src[end-nh+i]
		end
	else
		for i = 0:nh-1
			coef_dest[i+1] = coef_src[i+1]
		end
		for i = 1:nh-1
			coef_dest[nh+1+i] = coef_src[end-nh+i+1]
		end
		coef_dest[nh+1] = coef_src[nh+1] + coef_src[end-nh+1]
	end
	coef_dest
end

function derivative_space(s::FourierSpan, order; options...)
	A = coeftype(s)
	basis = set(s)
	if oddlength(basis)
		s
	else
		T = domaintype(basis)
		basis2 = FourierBasis{T}(length(basis)+1)
		Span(basis2, A)
	end
end

# Both differentiation and antidifferentiation are diagonal operations
function diff_scaling_function(b::FourierBasis, idx, order)
	T = domaintype(b)
	k = idx2frequency(b, idx)
	(2 * T(pi) * im * k)^order
end

function antidiff_scaling_function(b::FourierBasis, idx, order)
	T = domaintype(b)
	k = idx2frequency(b, idx)
	k==0 ? Complex{T}(0) : 1 / (k * 2 * T(pi) * im)^order
end


function differentiation_operator(s1::FourierSpan{A}, s2::FourierSpan{A}, order::Int; options...) where {A}
	if isodd(length(s1))
		@assert s1 == s2
		DiagonalOperator(s1, [diff_scaling_function(set(s1), idx, order) for idx in eachindex(set(s1))])
	else
		differentiation_operator(s2, s2, order; options...) * extension_operator(s1, s2; options...)
	end
end


function transform_from_grid(src, dest::FourierSpan, grid; options...)
	@assert compatible_grid(set(dest), grid)
	forward_fourier_operator(src, dest, coeftype(dest); options...)
end

function transform_to_grid(src::FourierSpan, dest, grid; options...)
	@assert compatible_grid(set(src), grid)
	backward_fourier_operator(src, dest, coeftype(src); options...)
end

function transform_to_grid_tensor(::Type{F}, ::Type{G}, s1, s2, grid; options...) where {F <: FourierSpan,G <: PeriodicEquispacedGrid}
	@assert reduce(&, map(compatible_grid, elements(s1), elements(grid)))
	backward_fourier_operator(s1, s2, coeftype(s1); options...)
end

function transform_from_grid_tensor(::Type{F}, ::Type{G}, s1, s2, grid; options...) where {F <: FourierSpan,G <: PeriodicEquispacedGrid}
	@assert reduce(&, map(compatible_grid, elements(s2), elements(grid)))
	forward_fourier_operator(s1, s2, coeftype(s2); options...)
end



function transform_from_grid_post(src, dest::FourierSpan, grid; options...)
	@assert compatible_grid(set(dest), grid)
    L = convert(coeftype(dest), length(src))
    ScalingOperator(dest, 1/sqrt(L))
end

function transform_to_grid_pre(src::FourierSpan, dest, grid; options...)
	@assert compatible_grid(set(src), grid)
	inv(transform_from_grid_post(dest, src, grid; options...))
end


# Try to efficiently evaluate a Fourier series on a regular equispaced grid
# The case of a periodic grid is handled generically in generic/evaluation, because
# it is the associated grid of the function set.
function grid_evaluation_operator(span::FourierSpan, dgs::DiscreteGridSpace, grid::EquispacedGrid; options...)
	fs = set(span)
	a = leftendpoint(grid)
	b = rightendpoint(grid)
	# We can use the fft if the equispaced grid is a subset of the periodic grid
	if (a > 0) || (b < 1)
		# We are dealing with a subgrid. The main question is: if we extend it
		# to the full support, is it compatible with a periodic grid?
		h = stepsize(grid)
		nleft = a/h
		nright = (1-b)/h
		if (nleft ≈ round(nleft)) && (nright ≈ round(nright))
			nleft_int = round(Int, nleft)
			nright_int = round(Int, nright)
			ntot = length(grid) + nleft_int + nright_int - 1
			T = domaintype(grid)
			super_grid = PeriodicEquispacedGrid(ntot, T(0), T(1))
			super_dgs = gridspace(span, super_grid)
			E = evaluation_operator(span, super_dgs; options...)
			R = IndexRestrictionOperator(super_dgs, dgs, nleft_int+1:nleft_int+length(grid))
			R*E
		else
			default_evaluation_operator(span, dgs; options...)
		end
	elseif a ≈ left(fs) && b ≈ right(fs)
		# TODO: cover the case where the EquispacedGrid is like a PeriodicEquispacedGrid
		# but with the right endpoint added
		default_evaluation_operator(span, dgs; options...)
	else
		default_evaluation_operator(span, dgs; options...)
	end
end

is_compatible(s1::FourierBasis, s2::FourierBasis) = true

# Multiplication of Fourier Series
function (*)(src1::FourierBasis, src2::FourierBasis, coef_src1, coef_src2)
	if oddlength(src1) && evenlength(src2)
	    dsrc2 = resize(src2, length(src2)+1)
	    (*)(src1, dsrc2, coef_src1, extension_operator(span(src2, eltype(coef_src2)), span(dsrc2, eltype(coef_src2)))*coef_src2)
	elseif evenlength(src1) && oddlength(src2)
		dsrc1 = resize(src1, length(src1)+1)
	    (*)(dsrc1, src2, extension_operator(span(src1, eltype(coef_sr1)), span(dsrc1, eltype(coef_src1)))*coef_src1,coef_src2)
	elseif evenlength(src1) && evenlength(src2)
		dsrc1 = resize(src1, length(src1)+1)
	    dsrc2 = resize(src2, length(src2)+1)
		T1 = eltype(coef_src1)
		T2 = eltype(coef_src2)
	    (*)(dsrc1,dsrc2,extension_operator(span(src1, T1), span(dsrc1, T1))*coef_src1, extension_operator(span(src2, T2), span(dsrc2, T2))*coef_src2)
	else # they are both odd
		@assert domaintype(src1) == domaintype(src2)
	    dest = FourierBasis{domaintype(src1)}(length(src1)+length(src2)-1)
	    coef_src1 = [coef_src1[(nhalf(src1))+2:end]; coef_src1[1:nhalf(src1)+1]]
	    coef_src2 = [coef_src2[(nhalf(src2))+2:end]; coef_src2[1:nhalf(src2)+1]]
	    coef_dest = conv(coef_src1,coef_src2)
	    coef_dest = [coef_dest[(nhalf(dest)+1):end]; coef_dest[1:(nhalf(dest))]]
	    (dest,coef_dest)
	end
end


dot(set::FourierBasis, f1::Function, f2::Function, nodes::Array=native_nodes(set); options...) =
    dot(x->conj(f1(x))*f2(x), nodes; options...)

function Gram(s::FourierSpan; options...)
	if iseven(length(s))
		CoefficientScalingOperator(s, s, (length(s)>>1)+1, one(coeftype(s))/2)
	else
		IdentityOperator(s, s)
	end
end

UnNormalizedGram(b::FourierSpan, oversampling) = ScalingOperator(b, b, length_oversampled_grid(set(b), oversampling))
