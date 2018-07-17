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
const FFreq = FourierFrequency

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

nhalf(b::FourierBasis) = nhalf(length(b))
nhalf(n::Int) = n>>1

maxfrequency(b::FourierBasis) = nhalf(b)
minfrequency(b::FourierBasis) = oddlength(b) ? -nhalf(b) : -nhalf(b)+1




#############
# Evaluation
#############

support(b::FourierBasis{T}) where T = UnitInterval{T}()

measure(b::FourierBasis{T}) where T = FourierMeasure{T}()

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

############################
# Extension and restriction
############################

# We make special-purpose operators for the extension of a Fourier series,
# since we have to add zeros in the middle of the given Fourier coefficients.
# This can not be achieved with a single IndexExtensionOperator.
# It could be a composition of two, but this special case is widespread and hence
# we make it more efficient.
struct FourierIndexExtensionOperator{T} <: DictionaryOperator{T}
	src		::	Dictionary
	dest	::	Dictionary
	n1		::	Int
	n2		::	Int
end

FourierIndexExtensionOperator(src, dest, n1 = length(src), n2 = length(dest)) =
	FourierIndexExtensionOperator{op_eltype(src,dest)}(src, dest, n1, n2)

string(op::FourierIndexExtensionOperator) = "Fourier series extension from length $(op.n1) to length $(op.n2)"

wrap_operator(src, dest, op::FourierIndexExtensionOperator{T}) where T =
	FourierIndexExtensionOperator{T}(src, dest, op.n1, op.n2)

function extension_operator(b1::FourierBasis, b2::FourierBasis; options...)
	@assert length(b1) <= length(b2)
	FourierIndexExtensionOperator(b1, b2)
end

function apply!(op::FourierIndexExtensionOperator, coef_dest, coef_src)
	## @assert length(dest) > length(src)

	nh = nhalf(length(coef_src))
	# We have to distinguish between even and odd length Fourier series
	# because the memory layout is slightly different
	if iseven(length(coef_src))
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


struct FourierIndexRestrictionOperator{T} <: DictionaryOperator{T}
	src		::	Dictionary
	dest	::	Dictionary
	n1		::	Int
	n2		::	Int
end

FourierIndexRestrictionOperator(src, dest, n1 = length(src), n2 = length(dest)) =
	FourierIndexRestrictionOperator{op_eltype(src,dest)}(src, dest, n1, n2)

string(op::FourierIndexRestrictionOperator) = "Fourier series restriction from length $(op.n1) to length $(op.n2)"

wrap_operator(src, dest, op::FourierIndexRestrictionOperator{T}) where T =
	FourierIndexRestrictionOperator{T}(src, dest, op.n1, op.n2)

function restriction_operator(b1::FourierBasis, b2::FourierBasis; options...)
	@assert length(b1) >= length(b2)
	FourierIndexRestrictionOperator(b1, b2)
end


function apply!(op::FourierIndexRestrictionOperator, coef_dest, coef_src)
	## @assert length(dest) < length(src)

	nh = nhalf(length(coef_dest))
	if isodd(length(coef_dest))
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

is_diagonal(::FourierIndexExtensionOperator) = true
is_diagonal(::FourierIndexRestrictionOperator) = true

ctranspose(op::FourierIndexExtensionOperator{T}) where {T} =
	FourierIndexRestrictionOperator{T}(dest(op), src(op), op.n2, op.n1)

ctranspose(op::FourierIndexRestrictionOperator{T}) where {T} =
	FourierIndexExtensionOperator{T}(dest(op), src(op), op.n2, op.n1)


function derivative_dict(s::FourierBasis, order; options...)
	if oddlength(s)
		s
	else
		T = domaintype(s)
		basis2 = FourierBasis{T}(length(s)+1)
		basis2
	end
end

# Both differentiation and antidifferentiation are diagonal operations
function diff_scaling_function(b::FourierBasis, idx, symbol)
	T = domaintype(b)
	k = idx2frequency(b,idx)
	symbol(2 * T(pi) * im * k)
end

diff_scaling_function(b::FourierBasis, idx, order::Int) = diff_scaling_function(b,idx,x->x^order)

function antidiff_scaling_function(b::FourierBasis, idx, order::Int)
	T = domaintype(b)
	k = idx2frequency(b, idx)
	k==0 ? Complex{T}(0) : 1 / (k * 2 * T(pi) * im)^order
end

differentiation_operator(s1::FourierBasis{T}, s2::FourierBasis{T}, order::Int; options...) where {T} = pseudodifferential_operator(s1,s2,x->x^order;options...)

pseudodifferential_operator(s::FourierBasis, symbol::Function; options...) = pseudodifferential_operator(s,s,symbol; options...)

function pseudodifferential_operator(s1::FourierBasis{T},s2::FourierBasis{T}, symbol::Function; options...) where {T}
	if isodd(length(s1))
		@assert s1 == s2
		_pseudodifferential_operator(s2, symbol; options...)
	else # The internal representation of Fourier is not closed under differentiation unless it is odd order
		_pseudodifferential_operator(s2, symbol; options...) * extension_operator(s1, s2; options...)
	end
end

_pseudodifferential_operator(s::FourierBasis{T}, symbol::Function; options...) where {T} = DiagonalOperator(s, [diff_scaling_function(s, idx, symbol) for idx in eachindex(s)])

pseudodifferential_operator(s::TensorProductDict,symbol::Function; options...) = pseudodifferential_operator(s,s,symbol; options...)

function pseudodifferential_operator(s1::TensorProductDict,s2::TensorProductDict,symb::Function; options...)
	#@assert length(first(methods(symbol)).sig.parameters) = dimension(s1) + 1
	@assert s1 == s2 # There is currently no support for s1 != s2
	# Build a vector of the first order differential operators in each spatial direction:
	Diffs = map(differentiation_operator,elements(s1))
	@assert is_diagonal(Diffs[1]) #should probably also check others too. This is a temp hack.
	# Build the diagonal from the symbol applied to the diagonals of these (diagonal) operators:
	N = prod(size(s1))
	diag = zeros(eltype(Diffs[1]),N)
	for k = 1:N
		vec = [diagonal(Diffs[i],native_index(s1, k)[i]) for i in 1:dimension(s1)]
		diag[k] = symb(vec)
	end
	DiagonalOperator(s1,diag)
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
			super_dgs = gridbasis(fs, super_grid)
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

iscosine(b::FourierBasis, i::FourierFrequency) = iseven(length(b)) && (i==length(b)>>1)


# Evaluate the inner product of two Fourier basis functions on the full domain
function innerproduct_fourier_full(b1::FourierBasis, i::FFreq, b2::FourierBasis, j::FFreq)
	T = codomaintype(b1)
	# The outcome can be zero, one or 1/2.
	# The latter happens when integrating a cosine with another cosine, or with
	# a complex exponential that contains a cosine with the same frequency.
	# Apart from that special case, the integral is zero when the indices differ
	# and one when they are equal.
	if (iscosine(b1, i) && abs(j) == i) || (iscosine(b2, j) && abs(i) == j)
		one(T)/2
	elseif i != j
		zero(T)
	else
		one(T)
	end
end

# Evaluate the inner product on the interval [a,b]
function innerproduct_fourier_part(b1::FourierBasis, i::FFreq, b2::FourierBasis, j::FFreq, a, b)
	S = domaintype(b1)
	T = codomaintype(b1)
	# convert 2π to type S and 2πi to type T
	twopi = 2*S(pi)
	tpi = 2im*T(pi)
	if iscosine(b1, i)
		if iscosine(b2, j)
			if i == j
				T(-cos(twopi*i*a)*sin(twopi*i*a)+cos(twopi*i*b)*sin(twopi*i*b)-twopi*i*(a-b))/(2*twopi*i)
			else
				T((-i-j)*sin(twopi*(i-j)*a)+(i+j)*sin(twopi*(i-j)*b)-(sin(twopi*(i+j)*a)-sin(twopi*(i+j)*b))*(i-j))/(2*i^2*twopi-2*j^2*twopi)
			end
		else
			if i == j
				((-cos(twopi*i*a)-sin(twopi*i*a))*exp(tpi*a)+exp(tpi*b)*(sin(twopi*i*b)+cos(twopi*i*b)))/(2*twopi*i)
			else
				(-T(im)*j*exp(tpi*j*a)*cos(twopi*i*a)+T(im)*j*exp(tpi*j*b)*cos(twopi*i*b)-i*exp(tpi*j*a)*sin(twopi*i*a)+i*exp(tpi*j*b)*sin(twopi*i*b))/((2*i^2-2*j^2)*S(pi))
			end
		end
	else
		if iscosine(b2, j)
			if i == j
				(-T(im)*cos(twopi*i*a)^2-cos(twopi*i*a)*sin(twopi*i*a)+T(im)*cos(twopi*i*b)^2+cos(twopi*i*b)*sin(twopi*i*b)-twopi*i*(a-b))/(2*twopi*i)
			else
				(-T(im)*i*exp(-tpi*i*a)*cos(twopi*j*a)+T(im)*i*exp(-tpi*i*b)*cos(twopi*j*b)+j*exp(-tpi*i*a)*sin(twopi*j*a)-j*exp(-tpi*i*b)*sin(twopi*j*b))/((2*i^2-2*j^2)*pi)
			end
		else
			if i == j
				T(b-a)
			else
				-T(1im)/(twopi*(j-i)) * (exp(tpi*b*(j-i)) - exp(tpi*a*(j-i)))
			end
		end
	end
end


# For the uniform measure on [0,1], invoke innerproduct_fourier_full
innerproduct(b1::FourierBasis, i::FFreq, b2::FourierBasis, j::FFreq, m::FourierMeasure) =
	innerproduct_fourier_full(b1, i, b2, j)

function innerproduct(b1::FourierBasis, i::FFreq, b2::FourierBasis, j::FFreq, m::LebesgueMeasure)
	d = support(m)
	if typeof(d) <: AbstractInterval
		if leftendpoint(d) == 0 && rightendpoint(d) == 1
			innerproduct_fourier_full(b1, i, b2, j)
		else
			innerproduct_fourier_part(b1, i, b2, j, leftendpoint(d), rightendpoint(d))
		end
	else
		default_innerproduct(b1, i, b2, j, m)
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


fourier_platform(;options...) = fourier_platform(Float64; options...)

fourier_platform(n::Int; options...) = fourier_platform(Float64, n; options...)

fourier_platform(::Type{T};options...) where {T} = fourier_platform(T, 1; options...)

function fourier_platform(::Type{T}, n::Int; oversampling=1) where {T}
	primal = FourierBasis{T}
	dual = FourierBasis{T}
        sampler = n -> GridSamplingOperator(gridbasis(PeriodicEquispacedGrid(round(Int,oversampling*n), UnitInterval{T}()), T))
        dual_sampler = n->(1/length(dest(sampler(n))))*sampler(n)
	params = isodd(n) ? OddDoublingSequence(n) : DoublingSequence(n)
	GenericPlatform(primal = primal, dual = dual, sampler = sampler, dual_sampler=dual_sampler,
		params = params, name = "Fourier series")
end
