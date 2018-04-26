# bf_splines.jl

SplineDegree{K} = Val{K}

spline_eval(::Type{SplineDegree{0}}, i, x, a, b, h) = (x >= a+i*h) && (x < a + (i+1)*h) ? 1 : 0

spline_eval(::Type{SplineDegree{K}}, i, x, a, b, h) where {K} =
	(x - (a+i*h)) / (K*h) * spline_eval(SplineDegree{K-1}, i, x, a, b, h) +
	(a+(i+K+1)*h - x)/(K*h) * spline_eval(SplineDegree{K-1}, i+1, x, a, b, h)


# Splines of degree K (with equispaced knots only...)
abstract type SplineBasis{K,T} <: Dictionary{T,T}
end

is_biorthogonal(::SplineBasis) = true

"The degree of the splines."
degree(b::SplineBasis{K}) where {K} = K

left(b::SplineBasis) = b.a
right(b::SplineBasis) = b.b

"Return the index of the interval between two knots in which x lies, starting from index 0."
interval(b::SplineBasis, x) = round(Int, floor( (x-left(b))/stepsize(b) ))

is_basis(b::SplineBasis) = true

# All splines have compact support
has_compact_support(::Type{B}) where {B <: SplineBasis} = True

"Return the i-th knot of the spline, using native indices."
knot(b::SplineBasis, idxn) = left(b) + idxn*stepsize(b)



"""
The full space of piecewise polynomials of degree K on n subintervals of [a,b].
"""
struct FullSplineBasis{K,T} <: SplineBasis{K,T}
	n		::	Int
	a		::	T
	b		::	T

	FullSplineBasis{K,T}(n, a = -one(T), b = one(T)) where {K,T} = new(n, a, b)
end

name(b::FullSplineBasis) = "Full splines of degree $(degree(b))"


FullSplineBasis{K,T}(n, a::T, b::T, ::Type{SplineDegree{K}}) = FullSplineBasis{K,T}(n, a, b)

FullSplineBasis{T}(n, a::T, b::T, k::Int = 3) = FullSplineBasis(n, a, b, SplineDegree{k})

FullSplineBasis{K,T}(n, ::Type{SplineDegree{K}} = SplineDegree{3}, ::Type{T} = Float64) = FullSplineBasis{K,T}(n)

instantiate{T}(::Type{FullSplineBasis}, n, ::Type{T}) = FullSplineBasis{3,T}(n-3)

# Full splines do not have an interpolation grid
#has_grid(b::FullSplineBasis) = true
# ASK added -1, correct?
length(b::FullSplineBasis{K}) where {K} = b.n+K-1

# Indices of splines naturally range from -K to n-1.
const FSplineIndex = ShiftedIndex
ordering(b::FullSplineBasis{K}) where {K} = ShiftedIndexList{K+1}(length(b))

left(b::FullSplineBasis, idxn::FSplineIndex) = max(b.a, knot(b, value(idxn)))
right(b::FullSplineBasis{K}, idxn::FSplineIndex) where {K} =
	min(b.b, knot(b, value(idxn) + K+1))

waypoints(b::FullSplineBasis{K}, idx) where {K} =
	unique([min(max(knot(b, native_index(b, idx)+i), b.a), b.b) for i = 0:K+1])

stepsize(b::FullSplineBasis) = stepsize(grid(b))

grid(b::FullSplineBasis) = EquispacedGrid(b.n, b.a, b.b)

function active_indices(b::FullSplineBasis{K}, x) where {K}
	i = interval(b, x)
	[i+j for j=1:min(K+1,length(b)-i)]
end



unsafe_eval_element(b::FullSplineBasis{K,T}, idxn::FSplineIndex, x) where {K,T} =
	spline_eval(SplineDegree{K}, value(idxn), x, b.a, b.b, stepsize(b))


# Natural splines of degree K
struct NaturalSplineBasis{K,T} <: SplineBasis{K,T}
	n		::	Int
	a		::	T
	b		::	T

	NaturalSplineBasis{K,T}(n, a = -one(T), b = one(T)) where {K,T} = new(n, a, b)
end

name(b::NaturalSplineBasis) = "Natural splines of degree $(degree(b))"

NaturalSplineBasis{K,T}(n, a::T, b::T, ::Type{SplineDegree{K}}) = NaturalSplineBasis{K,T}(n, a, b)

NaturalSplineBasis{T}(n, a::T, b::T, k::Int = 3) = NaturalSplineBasis(n, a, b, SplineDegree{k})

NaturalSplineBasis{K,T}(n, ::Type{SplineDegree{K}} = SplineDegree{3}, ::Type{T} = Float64) = NaturalSplineBasis{K,T}(n)

instantiate{T}(::Type{NaturalSplineBasis}, n, ::Type{T}) = NaturalSplineBasis{3,T}(n)

has_grid(b::NaturalSplineBasis) = true


length(b::NaturalSplineBasis) = b.n+1

# Indices of natural splines naturally range from 0 to n
const NSplineIndex = ShiftedIndex{1}
ordering(b::NaturalSplineBasis) = ShiftedIndexList{1}(length(b))

left(b::NaturalSplineBasis, idxn::NSplineIndex) = max(b.a, b.a + (value(idxn)-1)*b.h)
right(b::NaturalSplineBasis, idxn::NSplineIndex) = min(b.b, b.a + (value(idxn)+1)*b.h)

stepsize(b::NaturalSplineBasis) = stepsize(b.grid)

grid(b::NaturalSplineBasis) = EquispacedGrid(b.n, b.a, b.b)

unsafe_eval_element(b::NaturalSplineBasis{K,T}, idx::Int, x) where {K,T} =
	error("Natural splines not implemented yet. Sorry. Carry on.")


#######################
# Periodic splines
#######################


"""
Periodic splines of degree K.
"""
struct PeriodicSplineBasis{K,T} <: SplineBasis{K,T}
	n		::	Int
	a		::	T
	b		::	T

	PeriodicSplineBasis{K,T}(n, a = -one(T), b = one(T)) where {K,T} = new{K,T}(n, a, b)
end

name(b::PeriodicSplineBasis) = "Periodic splines of degree $(degree(b))"

## CONSTRUCTORS

# If no type parameter is given, assume K = 3
PeriodicSplineBasis(n::Int, ab...) = PeriodicSplineBasis{3}(n, ab...)

# If K is given but T is not, assume T=Float64
PeriodicSplineBasis{K}(n::Int) where {K} = PeriodicSplineBasis{K,Float64}(n)

# If a and b are given, get T from them, but make sure it is a floating point type
PeriodicSplineBasis{K}(n::Int, a::Number, b::Number) where {K} =
	PeriodicSplineBasis{K}(n, promote(a,b)...)
PeriodicSplineBasis{K}(n::Int, a::T, b::T) where {K,T <: Number} =
	PeriodicSplineBasis{K,float(T)}(n, a, b)

instantiate(::Type{PeriodicSplineBasis}, n, T) = PeriodicSplineBasis{3,T}(n)

dict_promote_domaintype(b::PeriodicSplineBasis{K,T}, ::Type{S}) where {K,T,S} =
	PeriodicSplineBasis{K,S}(b.n, b.a, b.b)

## Properties

length(b::PeriodicSplineBasis) = b.n

resize(b::PeriodicSplineBasis{K,T}, n) where {K,T} = PeriodicSplineBasis{K,T}(n, b.a, b.b)

rescale(s::PeriodicSplineBasis{K,T}, a, b) where {K,T} = PeriodicSplineBasis{K,T}(s.n, a, b)

has_grid(b::PeriodicSplineBasis) = true

grid(b::PeriodicSplineBasis) = PeriodicEquispacedGrid(b.n, b.a, b.b)

stepsize(b::PeriodicSplineBasis) = (b.b-b.a)/b.n

period(b::PeriodicSplineBasis) = b.b - b.a


## Indexing and evaluation

const PSplineIndex = ShiftedIndex{1}
ordering(b::PeriodicSplineBasis) = ShiftedIndexList{1}(length(b))

left(b::PeriodicSplineBasis) = b.a

right(b::PeriodicSplineBasis) = b.b

domain(b::PeriodicSplineBasis) = interval(b.a,b.b)

# There is something non-standard about these left and right routines: they currently
# return points outside the interval [a,b] for the first few and last few
# splines. This is due to periodicity: the first spline actually has its support near
# a and near b, i.e., its support restricted to [a,b] consists of two pieces. It is
# easier to use periodicity, and return a single interval near a or b, possibly outside [a,b].
function left(b::PeriodicSplineBasis{K}, idxn::PSplineIndex) where {K}
	j = linear_index(b, idxn)
	b.a + (j - 1 - ((K+1) >> 1) ) * stepsize(b)
end

function right(b::PeriodicSplineBasis{K}, idxn::PSplineIndex) where {K}
	j = linear_index(b, idxn)
	b.a + (j - ((K+1) >> 1) + K) * stepsize(b)
end

# We only return true when x is actually inside the interval, in spite of periodicity
function in_support(b::PeriodicSplineBasis{K}, idxn::PSplineIndex, x) where {K}
	idx = linear_index(b, idxn)
	period = right(b) - left(b)

	A = left(b) <= x <= right(b)
	B = (left(b, idx) <= x <= right(b, idx)) || (left(b, idx) <= x-period <= right(b, idx)) ||
		(left(b, idx) <= x+period <= right(b, idx))
	A && B
end

function unsafe_eval_element(b::PeriodicSplineBasis{K,T}, idxn::PSplineIndex, x) where {K,T}
	idx_n = value(idxn)
	while x < left(b)
		x = x + period(b)
	end
	while x > right(b)
		x = x - period(b)
	end

	h = stepsize(b)
	L1 = (K-1) >> 1
	L2 = K-L1
	if idx_n <= L1
		z = spline_eval(SplineDegree{K}, idx_n-L1-1, x, b.a, b.b, h) +
			spline_eval(SplineDegree{K}, idx_n-L1-1, x-period(b), b.a, b.b, h)
	elseif idxn > b.n - L2
		z = spline_eval(SplineDegree{K}, idx_n-L1-1, x, b.a, b.b, h) +
			spline_eval(SplineDegree{K}, idx_n-L1-1, x+period(b), b.a, b.b, h)
	else
		z = spline_eval(SplineDegree{K}, idx_n-L1-1, x, b.a, b.b, h)
	end
	z
end


function eval_expansion(b::PeriodicSplineBasis{K}, coef, x::T) where {K,T <: Number}
	i = interval(b, x)
	n = length(b)

	L1 = (K-1) >> 1
	L2 = K-L1

	z = zero(T)
	for idxn = i-L1-1:i+L2
		idx = linear_index(b, PSplineIndex(mod(idxn,n)))
		z = z + coef[idx] * unsafe_eval_element(b, idx, x)
	end
	z
end
