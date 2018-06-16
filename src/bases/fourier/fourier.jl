# fourier.jl

"""
A Fourier basis on the interval `[0,1]`. The basis functions are given by
`exp(2 π i k)`, with `k` ranging from `-N` to `N` for Fourier series of odd
length (`2N+1`) and from `-N+1` to `N` for even length. In the latter case, the
highest frequency basis function corresponding to frequency `N` is a cosine.

The basis functions are ordered the way they are expected by a typical FFT
implementation. The frequencies k are in the following order:
```
0 1 2 3 ... N -N -N+1 ... -2 -1,
```
for odd length and
```
0 1 2 3 ... N -N+1 ... -2 -1,
```
for even length.
"""
struct FourierBasis{T <: Real} <: Dictionary{T,Complex{T}}
	n	::	Int
end

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

length(b::FourierBasis) = b.n
oddlength(b::FourierBasis) = isodd(length(b))
evenlength(b::FourierBasis) = iseven(length(b))

instantiate(::Type{FourierBasis}, n, ::Type{T}) where {T} = FourierBasis{T}(n)

dict_promote_domaintype(b::FourierBasis{T}, ::Type{S}) where {T,S} = FourierBasis{promote_type(S,T)}(b.n)
dict_promote_coeftype(b::FourierBasis{T}, ::Type{S}) where {T,S<:Real} = error("FourierBasis with real coefficients not implemented")
dict_promote_coeftype(b::FourierBasis{T}, ::Type{S}) where {T,S<:Complex} = FourierBasis{promote_type(T,real(S))}(b.n)
resize(b::FourierBasis{T}, n) where {T} = FourierBasis{T}(n)



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
compatible_grid(b::FourierBasis, grid::PeriodicEquispacedGrid) =
	has_grid_equal_span(b,grid) && (length(b)==length(grid))
# - Any non-periodic grid is not compatible
compatible_grid(b::FourierBasis, grid::AbstractGrid) = false
# - We have a transform if the grid is compatible
has_grid_transform(b::FourierBasis, gb, grid) = compatible_grid(b, grid)


grid(b::FourierBasis) = PeriodicEquispacedGrid(b.n, support(b), domaintype(b))


##################
# Native indices
##################

const FourierFrequency = NativeIndex{:fourier}

frequency(idxn::FourierFrequency) = value(idxn)

Base.show(io::IO, idx::BasisFunctions.NativeIndex{:fourier}) =
	print(io, "Fourier frequency: $(value(idx))")

"""
`FFTIndexList` defines the map from native indices to linear indices
for a finite Fourier basis, when the indices are ordered in the way they
are expected in the FFT routine.
"""
struct FFTIndexList <: IndexList{FourierFrequency}
	n	::	Int
end

length(list::FFTIndexList) = list.n
size(list::FFTIndexList) = (list.n,)

# The frequency of an even Fourier basis ranges from -N+1 to N.
# The frequency of an odd Fourier basis ranges from -N to N.
# This makes for a small difference in the ordering.
function getindex(m::FFTIndexList, idx::Int)
	n = length(m)
	nhalf = n >> 1
	if idx <= nhalf+1
		FourierFrequency(idx-1)
	else
		if iseven(n)
			FourierFrequency(idx-2*nhalf-1)
		else
			FourierFrequency(idx-2*nhalf-2)
		end
	end
end

function getindex(list::FFTIndexList, idxn::FourierFrequency)
	k = value(idxn)
	k >= 0 ? k+1 : length(list)+k+1
end

ordering(b::FourierBasis) = FFTIndexList(length(b))

# Shorthand: compute the linear index based on the size and element type
# of an array only
linear_index(idxn::FourierFrequency, size::Tuple{Int}, T) = FFTIndexList(size[1])[idxn]

# Convenience: compute with integer frequencies, rather than FourierFrequency types
idx2frequency(b::FourierBasis, idx) = frequency(native_index(b, idx))
frequency2idx(b::FourierBasis, k) = linear_index(b, FourierFrequency(k))

nhalf(b::FourierBasis) = length(b)>>1

maxfrequency(b::FourierBasis) = nhalf(b)
minfrequency(b::FourierBasis) = oddlength(b) ? -nhalf(b) : -nhalf(b)+1




#############
# Evaluation
#############

support(b::FourierBasis) = UnitInterval{domaintype(b)}()

period(b::FourierBasis{T}) where {T} = T(1)


# One has to be careful here not to match Floats and BigFloats by accident.
# Hence the conversions to S in the lines below.
function unsafe_eval_element(b::FourierBasis, idxn::FourierFrequency, x)
	S = domaintype(b)
	k = frequency(idxn)

	# Even-length Fourier series have a cosine at the maximal frequency.
	# We do a test to distinguish between the complex exponentials and the cosine
	if oddlength(b) || k != maxfrequency(b)
		exp(x * 2 * S(pi) * 1im  * k)
	else
		# We convert to a complex number for type safety here, because the
		# exponentials above are complex-valued but the cosine is real
		complex(cospi(2*S(x)*k))
	end
end

function unsafe_eval_element_derivative(b::FourierBasis, idxn::FourierFrequency, x)
	# The structure for this reason is similar to the unsafe_eval_element routine above
	S = domaintype(b)
	k = frequency(idxn)
	if oddlength(b) || k != maxfrequency(b)
		arg = 2*S(pi)*1im*k
		arg * exp(arg * x)
	else
		arg = 2*S(pi)*k
		complex(-arg * sin(arg*x))
	end
end

function unsafe_moment(b::FourierBasis, idxn::FourierFrequency)
	T = codomaintype(b)
	frequency(idx) == 0 ? T(1) : T(0)
end



extension_size(b::FourierBasis) = evenlength(b) ? 2*length(b) : 2*length(b)+1

approx_length(b::FourierBasis, n::Int) = n



"Shift an expansion to the right by delta."
function shift(b::FourierBasis, coefficients, delta)
	# Only works for odd-length Fourier series for now
	@assert oddlength(b)
	S = domaintype(b)
	coef2 = copy(coefficients)
	for i in eachindex(coefficients)
		coef2[i] *= exp(2 * S(pi) * im * idx2frequency(b, i) * delta)
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

function derivative_space(s::FourierBasis, order; options...)
	if oddlength(s)
		s
	else
		T = domaintype(s)
		basis2 = FourierBasis{T}(length(s)+1)
		basis2
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


function differentiation_operator(s1::FourierBasis{T}, s2::FourierBasis{T}, order::Int; options...) where {T}
	if isodd(length(s1))
		@assert s1 == s2
		DiagonalOperator(s1, [diff_scaling_function(s1, idx, order) for idx in eachindex(s1)])
	else
		differentiation_operator(s2, s2, order; options...) * extension_operator(s1, s2; options...)
	end
end


function transform_from_grid(src, dest::FourierBasis, grid; options...)
	@assert compatible_grid(dest, grid)
	forward_fourier_operator(src, dest, coeftype(dest); options...)
end

function transform_to_grid(src::FourierBasis, dest, grid; options...)
	@assert compatible_grid(src, grid)
	backward_fourier_operator(src, dest, coeftype(src); options...)
end

function transform_to_grid_tensor(::Type{F}, ::Type{G}, s1, s2, grid; options...) where {F <: FourierBasis,G <: PeriodicEquispacedGrid}
	#@assert reduce(&, map(compatible_grid, elements(s1), elements(grid)))
	backward_fourier_operator(s1, s2, coeftype(s1); options...)
end

function transform_from_grid_tensor(::Type{F}, ::Type{G}, s1, s2, grid; options...) where {F <: FourierBasis,G <: PeriodicEquispacedGrid}
	#@assert reduce(&, map(compatible_grid, elements(s2), elements(grid)))
	forward_fourier_operator(s1, s2, coeftype(s2); options...)
end

## function transform_from_grid_post(src, dest::FourierSeries, grid; options...)
## 	@assert compatible_grid(dictionary(dest), grid)
##     L = convert(coeftype(dest), length(src))
##     ScalingOperator(dest, 1/L)
## end

## function transform_to_grid_pre(src::FourierSeries, dest, grid; options...)
## 	@assert compatible_grid(dictionary(src), grid)
## 	inv(transform_from_grid_post(dest, src, grid; options...))
## end




# Try to efficiently evaluate a Fourier series on a regular equispaced grid
# The case of a periodic grid is handled generically in generic/evaluation, because
# it is the associated grid of the function set.
function grid_evaluation_operator(fs::FourierBasis, dgs::GridBasis, grid::EquispacedGrid; options...)
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
			super_dgs = gridspace(fs, super_grid)
			E = evaluation_operator(fs, super_dgs; options...)
			R = IndexRestrictionOperator(super_dgs, dgs, nleft_int+1:nleft_int+length(grid))
			R*E
		else
			default_evaluation_operator(fs, dgs; options...)
		end
	elseif a ≈ infimum(support(fs)) && b ≈ supremum(support(fs))
		# TODO: cover the case where the EquispacedGrid is like a PeriodicEquispacedGrid
		# but with the right endpoint added
		default_evaluation_operator(fs, dgs; options...)
	else
		default_evaluation_operator(fs, dgs; options...)
	end
end

is_compatible(s1::FourierBasis, s2::FourierBasis) = true

# Multiplication of Fourier Series
function (*)(src1::FourierBasis, src2::FourierBasis, coef_src1, coef_src2)
	if oddlength(src1) && evenlength(src2)
	    dsrc2 = resize(src2, length(src2)+1)
	    (*)(src1, dsrc2, coef_src1, extension_operator(src2, dsrc2)*coef_src2)
	elseif evenlength(src1) && oddlength(src2)
		dsrc1 = resize(src1, length(src1)+1)
	    (*)(dsrc1, src2, extension_operator(src1,dsrc1)*coef_src1,coef_src2)
	elseif evenlength(src1) && evenlength(src2)
		dsrc1 = resize(src1, length(src1)+1)
	    dsrc2 = resize(src2, length(src2)+1)
		T1 = eltype(coef_src1)
		T2 = eltype(coef_src2)
	    (*)(dsrc1,dsrc2,extension_operator(src1, dsrc1)*coef_src1, extension_operator(src2, dsrc2)*coef_src2)
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


dot(b::FourierBasis, f1::Function, f2::Function, nodes::Array=native_nodes(b); options...) =
    dot(x->conj(f1(x))*f2(x), nodes; options...)

function Gram(s::FourierBasis; options...)
	if iseven(length(s))
		CoefficientScalingOperator(s, s, (length(s)>>1)+1, one(coeftype(s))/2)
	else
		IdentityOperator(s, s)
	end
end

UnNormalizedGram(b::FourierBasis, oversampling) = ScalingOperator(b, b, length_oversampled_grid(b, oversampling))


##################
# Platform
##################

"A doubling sequence that produces odd values, i.e., the value after `n` is `2n+1`."
struct OddDoublingSequence <: DimensionSequence
    initial ::  Int

	# Default constructor to guarantee that the initial value is odd
	OddDoublingSequence(initial::Int) = (@assert isodd(initial); new(initial))
end

initial(s::OddDoublingSequence) = s.initial

OddDoublingSequence() = OddDoublingSequence(1)

getindex(s::OddDoublingSequence, idx::Int) = initial(s) * 2<<(idx-1) - 1


fourier_platform() = fourier_platform(Float64)

fourier_platform(n::Int) = fourier_platform(Float64, n)

fourier_platform(::Type{T}) where {T} = fourier_platform(T, 1)

function fourier_platform(::Type{T}, n::Int) where {T}
	primal = FourierBasis{T}
	dual = FourierBasis{T}
	sampler = n -> GridSamplingOperator(gridbasis(PeriodicEquispacedGrid(n, UnitInterval{T}()), T))
	params = isodd(n) ? OddDoublingSequence(n) : DoublingSequence(n)
	GenericPlatform(primal = primal, dual = dual, sampler = sampler,
		params = params, name = "Fourier series")
end
