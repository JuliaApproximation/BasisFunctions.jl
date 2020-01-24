
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
struct Fourier{T <: Real} <: Dictionary{T,Complex{T}}
	n	::	Int
end

name(b::Fourier) = "Fourier series"

# The default numeric type is Float64
Fourier(n::Int) = Fourier{Float64}(n)

# Convenience constructor: map the Fourier series to the interval [a,b]
Fourier{T}(n, a::Number, b::Number) where {T} = rescale(Fourier{T}(n), a, b)

# We can deduce a candidate for T from a and b
function Fourier(n, a::Number, b::Number)
	T = float(promote_type(typeof(a),typeof(b)))
	Fourier{T}(n, a, b)
end

size(b::Fourier) = (b.n,)

oddlength(b::Fourier) = isodd(length(b))
evenlength(b::Fourier) = iseven(length(b))

similar(b::Fourier, ::Type{T}, n::Int) where {T} = Fourier{T}(n)

# Properties

isreal(b::Fourier) = false

isbasis(b::Fourier) = true
isorthogonal(b::Fourier, ::FourierMeasure) = true
isorthogonal(b::Fourier, measure::DiracCombMeasure) = islooselycompatible(b, grid(measure))
isorthogonal(b::Fourier, measure::DiracCombProbabilityMeasure) = islooselycompatible(b, grid(measure))


isorthonormal(b::Fourier, ::FourierMeasure) = oddlength(b)
isorthonormal(b::Fourier, measure::DiracCombProbabilityMeasure) = iscompatible(b, grid(measure)) || islooselycompatible(b, grid(measure)) && oddlength(b)
isbiorthogonal(b::Fourier) = true

# Methods for purposes of testing functionality.
hasinterpolationgrid(b::Fourier) = true
hasderivative(b::Fourier) = true
# Until adapted for DC coefficient
hasantiderivative(b::Fourier) = false
hasextension(b::Fourier) = isodd(length(b))

# For hastransform we introduce some more functionality:
# - Check whether the given periodic equispaced grid is compatible with the FFT operators
# 1+ because 0!≅eps()
iscompatible(dict::Fourier, grid::AbstractEquispacedGrid) =
	islooselycompatible(dict, grid) && (length(dict)==length(grid))
# - Fourier grids are of course okay
iscompatible(dict::Fourier, grid::FourierGrid) = length(dict)==length(grid)
# - Any non-periodic grid is not compatible
iscompatible(dict::Fourier, grid::AbstractGrid) = false
# - We have a transform if the grid is compatible
hasgrid_transform(dict::Fourier, gb, grid) = iscompatible(dict, grid)

islooselycompatible(dict::Fourier, grid::AbstractEquispacedGrid) =
	isperiodic(grid) && support(dict) ≈ support(grid)


interpolation_grid(b::Fourier{T}) where {T} = FourierGrid{T}(length(b))


##################
# Native indices
##################

const FourierFrequency = NativeIndex{:fourier}
const FFreq = FourierFrequency

frequency(idxn::FourierFrequency) = value(idxn)

Base.show(io::IO, idx::NativeIndex{:fourier}) =
	print(io, "Fourier frequency $(value(idx))")

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

ordering(b::Fourier) = FFTIndexList(length(b))

# Shorthand: compute the linear index based on the size and element type
# of an array only
linear_index(idxn::FourierFrequency, size::Tuple{Int}, T) = FFTIndexList(size[1])[idxn]

# Convenience: compute with integer frequencies, rather than FourierFrequency types
idx2frequency(b::Fourier, idx) = frequency(native_index(b, idx))
frequency2idx(b::Fourier, k) = linear_index(b, FourierFrequency(k))

nhalf(b::Fourier) = nhalf(length(b))
nhalf(n::Int) = n>>1

maxfrequency(b::Fourier) = nhalf(b)
minfrequency(b::Fourier) = oddlength(b) ? -nhalf(b) : -nhalf(b)+1




#############
# Evaluation
#############

support(b::Fourier{T}) where T = UnitInterval{T}()

hasmeasure(b::Fourier) = true
measure(b::Fourier{T}) where T = FourierMeasure{T}()

period(b::Fourier{T}) where {T} = T(1)


# One has to be careful here not to match Floats and BigFloats by accident.
# Hence the conversions to S in the lines below.
function unsafe_eval_element(b::Fourier, idxn::FourierFrequency, x)
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

function unsafe_eval_element_derivative(b::Fourier, idxn::FourierFrequency, x)
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

function unsafe_moment(b::Fourier, idxn::FourierFrequency)
	T = codomaintype(b)
	frequency(idx) == 0 ? T(1) : T(0)
end



# By default, we preserve the odd/even property of the size when extending
extension_size(b::Fourier) = evenlength(b) ? 2*length(b) : max(3,2*length(b)-1)

approx_length(b::Fourier, n::Int) = n



"Shift an expansion to the right by delta."
function shift(b::Fourier, coefficients, delta)
	# Only works for odd-length Fourier series for now
	@assert oddlength(b)
	S = domaintype(b)
	coef2 = copy(coefficients)
	for i in eachindex(coefficients)
		coef2[i] *= exp(2 * S(pi) * im * idx2frequency(b, i) * delta)
	end
	coef2
end


function transform_from_grid(src::GridBasis, dest::Fourier, grid; T = op_eltype(src,dest), options...)
	@assert iscompatible(dest, grid)
	forward_fourier_operator(src, dest, T; options...)
end

function transform_to_grid(src::Fourier, dest::GridBasis, grid; T = op_eltype(src,dest), options...)
	@assert iscompatible(src, grid)
	inverse_fourier_operator(src, dest, T; options...)
end

grid_evaluation_operator(dict::Fourier, gb::GridBasis, grid::FourierGrid; options...) =
	resize_and_transform(dict, gb, grid; options...)


function grid_evaluation_operator(dict::Fourier, gb::GridBasis,
			grid::PeriodicEquispacedGrid; options...)
	if support(grid)≈support(dict)
		resize_and_transform(dict, gb, grid; options...)
	else
		@debug "Periodic grid mismatch with Fourier basis"
		dense_evaluation_operator(dict, gb; options...)
	end
end

to_periodic_grid(dict::Fourier, grid::AbstractGrid) = nothing
to_periodic_grid(dict::Fourier, grid::PeriodicEquispacedGrid{T}) where {T} =
	iscompatible(dict, grid) ? FourierGrid{T}(length(grid)) : nothing

function grid_evaluation_operator(dict::Fourier, gb::GridBasis, grid;
			options...)
	grid2 = to_periodic_grid(dict, grid)
	if grid2 != nothing
		gb2 = GridBasis{coefficienttype(gb)}(grid2)
		evaluation_operator(dict, gb2, grid2; options...) * gridconversion(gb, gb2; options...)
	else
		@debug "Evaluation: could not convert $(string(grid)) to periodic grid"
		dense_evaluation_operator(dict, gb; options...)
	end
end

# Try to efficiently evaluate a Fourier series on a regular equispaced grid
function grid_evaluation_operator(fs::Fourier, dgs::GridBasis, grid::EquispacedGrid; T=op_eltype(fs, dgs), options...)

	# We can use the fft if the equispaced grid is a subset of the periodic grid
	if support(grid) ≈ support(fs)
		# TODO: cover the case where the EquispacedGrid is like a PeriodicEquispacedGrid
		# but with the right endpoint added
		return dense_evaluation_operator(fs, dgs; T=T, options...)
	elseif support(grid) ∈ support(fs)
		a, b = endpoints(support(grid))
		# We are dealing with a subgrid. The main question is: if we extend it
		# to the full support, is it compatible with a periodic grid?
		h = step(grid)
		nleft = a/h
		nright = (1-b)/h
		if (nleft ≈ round(nleft)) && (nright ≈ round(nright))
			nleft_int = round(Int, nleft)
			nright_int = round(Int, nright)
			ntot = length(grid) + nleft_int + nright_int - 1
			super_grid = FourierGrid(ntot)
			super_dgs = GridBasis(fs, super_grid)
			E = evaluation_operator(fs, super_dgs; T=T, options...)
			R = IndexRestrictionOperator(super_dgs, dgs, nleft_int+1:nleft_int+length(grid); T=T)
			R*E
		else
			dense_evaluation_operator(fs, dgs; T=T, options...)
		end
	else
		dense_evaluation_operator(fs, dgs; T=T, options...)
	end
end

function grid_evaluation_operator(dict::Fourier, gb::GridBasis, grid::MidpointEquispacedGrid;
			T=op_eltype(dict, gb), options...)
	if isodd(length(grid)) && support(grid)≈support(dict)
		if length(grid) == length(dict)
			A = evaluation_operator(dict, FourierGrid{domaintype(dict)}(length(dict)); T=T, options...)
			diag = zeros(T, length(dict))
			delta = step(grid)/2
			for i in 1:length(dict)
				diag[i] = exp(2 * T(pi) * im * idx2frequency(dict, i) * delta)
			end
			D = DiagonalOperator(dict, diag)
			wrap_operator(dict, gb, A*D)
		else
			dict2 = resize(dict, length(grid))
			evaluation_operator(dict2, grid) * extension_operator(dict, dict2)
		end
	else
		dense_evaluation_operator(dict, gb; T=T, options...)
	end
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

FourierIndexExtensionOperator(src, dest, n1 = length(src), n2 = length(dest);
			T=op_eltype(src,dest)) =
	FourierIndexExtensionOperator{T}(src, dest, n1, n2)

string(op::FourierIndexExtensionOperator) = "Fourier series extension from length $(op.n1) to length $(op.n2)"

wrap_operator(src, dest, op::FourierIndexExtensionOperator{T}) where {T} =
	FourierIndexExtensionOperator{T}(src, dest, op.n1, op.n2)

function extension_operator(b1::Fourier, b2::Fourier; T = op_eltype(b1,b2), options...)
	@assert length(b1) <= length(b2)
	FourierIndexExtensionOperator(b1, b2; T=T)
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

FourierIndexRestrictionOperator(src, dest, n1 = length(src), n2 = length(dest); T = op_eltype(src,dest), options...) =
	FourierIndexRestrictionOperator{T}(src, dest, n1, n2)

string(op::FourierIndexRestrictionOperator) = "Fourier series restriction from length $(op.n1) to length $(op.n2)"

wrap_operator(src, dest, op::FourierIndexRestrictionOperator{T}) where T =
	FourierIndexRestrictionOperator{T}(src, dest, op.n1, op.n2)

function restriction_operator(b1::Fourier, b2::Fourier; T=op_eltype(b1,b2), options...)
	@assert length(b1) >= length(b2)
	FourierIndexRestrictionOperator(b1, b2; T=T)
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
		# if isodd(length(coef_src))
			# coef_dest[nh+1] = (coef_src[nh+1] + coef_src[end-nh+1])
		# else
			coef_dest[nh+1] = (coef_src[nh+1] + coef_src[end-nh+1]) / 2 # Added  / 2 to get adjoint correct
		# end
	end
	coef_dest
end

isdiag(::FourierIndexExtensionOperator) = true
isdiag(::FourierIndexRestrictionOperator) = true

adjoint(op::FourierIndexExtensionOperator{T}) where {T} =
	FourierIndexRestrictionOperator{T}(dest(op), src(op), op.n2, op.n1)

adjoint(op::FourierIndexRestrictionOperator{T}) where {T} =
	FourierIndexExtensionOperator{T}(dest(op), src(op), op.n2, op.n1)::DictionaryOperator

function derivative_dict(s::Fourier, order; options...)
	if oddlength(s)
		s
	else
		T = domaintype(s)
		basis2 = Fourier{T}(length(s)+1)
		basis2
	end
end

# Both differentiation and antidifferentiation are diagonal operations
function diff_scaling_function(b::Fourier, idx, symbol)
	T = domaintype(b)
	k = idx2frequency(b,idx)
	symbol(2 * T(pi) * im * k)
end

diff_scaling_function(b::Fourier, idx, order::Int) = diff_scaling_function(b,idx,x->x^order)

function antidiff_scaling_function(b::Fourier, idx, order::Int)
	T = domaintype(b)
	k = idx2frequency(b, idx)
	k==0 ? Complex{T}(0) : 1 / (k * 2 * T(pi) * im)^order
end

differentiation_operator(s1::Fourier{T}, s2::Fourier{T}, order = 1; options...) where {T} = pseudodifferential_operator(s1,s2,x->x^order;options...)

pseudodifferential_operator(s::Fourier, symbol::Function; options...) = pseudodifferential_operator(s,s,symbol; options...)

function pseudodifferential_operator(s1::Fourier{T},s2::Fourier{T}, symbol::Function; options...) where {T}
	if isodd(length(s1))
		@assert s1 == s2
		_pseudodifferential_operator(s2, symbol; options...)
	else # The internal representation of Fourier is not closed under differentiation unless it is odd order
		_pseudodifferential_operator(s2, symbol; options...) * extension_operator(s1, s2; options...)
	end
end

_pseudodifferential_operator(s::Fourier, symbol::Function; T=coefficienttype(s), options...) =
	DiagonalOperator(s, [diff_scaling_function(s, idx, symbol) for idx in eachindex(s)]; T=T)

pseudodifferential_operator(s::TensorProductDict,symbol::Function; options...) = pseudodifferential_operator(s,s,symbol; options...)

function pseudodifferential_operator(s1::TensorProductDict,s2::TensorProductDict,symbol::Function; T=op_eltype(s1,s2), options...)
	#@assert length(first(methods(symbol)).sig.parameters) = dimension(s1) + 1
	@assert s1 == s2 # There is currently no support for s1 != s2
	# Build a vector of the first order differential operators in each spatial direction:
	Diffs = map(x->differentiation_operator(x; T=T, options...),elements(s1))
	@assert isdiag(Diffs[1]) #should probably also check others too. This is a temp hack.
	# Build the diagonal from the symbol applied to the diagonals of these (diagonal) operators:
	N = prod(size(s1))
	diag = zeros(N)
	for k = 1:N
		vec = [diag(Diffs[i],native_index(s1, k)[i]) for i in 1:dimension(s1)]
		diag[k] = symbol(vec)
	end
	DiagonalOperator(s1,diag; T=T)
end




# Try to efficiently evaluate a Fourier series on a regular equispaced grid
# The case of a periodic grid is handled generically in generic/evaluation, because
# it is the associated grid of the function set.
function evaluation_operator(fs::Fourier, gb::GridBasis, grid::EquispacedGrid; T=op_eltype(fs,gb), options...)
	# We can use the fft if the equispaced grid is a subset of the periodic grid
	if support(grid) ∈ support(fs)
		# We are dealing with a subgrid. The main question is: if we extend it
		# to the full support, is it compatible with a periodic grid?
		h = step(grid)
		nleft = a/h
		nright = (1-b)/h
		if (nleft ≈ round(nleft)) && (nright ≈ round(nright))
			nleft_int = round(Int, nleft)
			nright_int = round(Int, nright)
			ntot = length(grid) + nleft_int + nright_int - 1
			T = domaintype(grid)
			super_grid = FourierGrid(ntot)
			super_gb = GridBasis(fs, super_grid)
			E = evaluation_operator(fs, super_gb; T=T, options...)
			R = IndexRestrictionOperator(super_gb, gb, nleft_int+1:nleft_int+length(grid); T=T)
			R*E
		else
			dense_evaluation_operator(fs, gb; T=T, options...)
		end
	elseif a ≈ infimum(support(fs)) && b ≈ supremum(support(fs))
		# TODO: cover the case where the EquispacedGrid is like a PeriodicEquispacedGrid
		# but with the right endpoint added
		dense_evaluation_operator(fs, gb; T=T, options...)
	else
		dense_evaluation_operator(fs, gb; T=T, options...)
	end
end

# Multiplication of Fourier Series
function (*)(src1::Fourier, src2::Fourier, coef_src1, coef_src2)
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
	    dest = Fourier{domaintype(src1)}(length(src1)+length(src2)-1)
	    coef_src1 = [coef_src1[(nhalf(src1))+2:end]; coef_src1[1:nhalf(src1)+1]]
	    coef_src2 = [coef_src2[(nhalf(src2))+2:end]; coef_src2[1:nhalf(src2)+1]]
	    coef_dest = conv(coef_src1,coef_src2)
	    coef_dest = [coef_dest[(nhalf(dest)+1):end]; coef_dest[1:(nhalf(dest))]]
	    (dest,coef_dest)
	end
end

iscosine(b::Fourier, i::FourierFrequency) = iseven(length(b)) && (i==length(b)>>1)


## Inner products

# Evaluate the inner product of two Fourier basis functions on the full domain
function innerproduct_fourier_full(b1::Fourier, i::FFreq, b2::Fourier, j::FFreq)
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
function innerproduct_fourier_part(b1::Fourier, i::FFreq, b2::Fourier, j::FFreq, a, b)
	S = domaintype(b1)
	T = codomaintype(b1)
	# convert 2π to type S and 2πi to type T
	twopi = 2*S(pi)
	tpi = 2im*T(pi)
	if iscosine(b1, i)
		if iscosine(b2, j)
			if abs(i) == abs(j)
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
			if abs(i) == abs(j)
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
innerproduct_native(b1::Fourier, i::FFreq, b2::Fourier, j::FFreq, m::FourierMeasure; options...) =
	innerproduct_fourier_full(b1, i, b2, j)

function innerproduct_native(b1::Fourier, i::FFreq, b2::Fourier, j::FFreq, m::LebesgueMeasure; options...)
	d = support(m)
	if d isa AbstractInterval
		if infimum(d) == 0 && supremum(d) == 1
			innerproduct_fourier_full(b1, i, b2, j)
		else
			innerproduct_fourier_part(b1, i, b2, j, infimum(d), supremum(d))
		end
	else
		default_dict_innerproduct(b1, i, b2, j, m)
	end
end

function gramoperator(dict::Fourier, measure::FourierMeasure; T = coefficienttype(dict), options...)
	@assert isorthogonal(dict, measure) # some robustness.
	if iseven(length(dict))
		CoefficientScalingOperator{T}(dict, (length(dict)>>1)+1, one(T)/2)
	else
		@assert isorthonormal(dict, measure)
		IdentityOperator{T}(dict, dict)
	end
end

function gramoperator(dict::Fourier, measure::DiscreteMeasure, grid::AbstractEquispacedGrid, weights::FillArrays.AbstractFill;
	T = promote_type(subdomaintype(measure), coefficienttype(dict)), options...)
	if support(grid) ≈ support(dict) && isperiodic(grid)
		if isorthonormal(dict, measure)
			IdentityOperator{T}(dict)
		elseif isorthogonal(dict, measure)
			if isodd(length(dict)) || (length(dict)==length(grid))
				ScalingOperator(dict, unsafe_discrete_weight(measure, 1)*length(grid); T=T)
			else
				CoefficientScalingOperator{T}(dict, (length(dict)>>1)+1, one(T)/2)*ScalingOperator(dict, weights[1]*length(grid); T=T)
			end
		end
	else
		default_mixedgramoperator_discretemeasure(dict, dict, measure, grid, weights; T=T, options...)
	end
end
