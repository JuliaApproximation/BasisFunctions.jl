# bf_splines.jl


SplineDegree{K} = Val{K}


spline_eval(::Type{SplineDegree{0}}, i, x, a, b, h) = (x >= a+i*h) && (x < a + (i+1)*h) ? 1 : 0

spline_eval{K}(::Type{SplineDegree{K}}, i, x, a, b, h) = (x - (a+i*h)) / (K*h) * spline_eval(SplineDegree{K-1}, i, x, a, b, h) + (a+(i+K+1)*h - x)/(K*h) * spline_eval(SplineDegree{K-1}, i+1, x, a, b, h)


# Splines of degree K (with equispaced knots only...)
abstract type SplineBasis{K,T} <: FunctionSet{T}
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

# Full splines to not have an interpolation grid
#has_grid(b::FullSplineBasis) = true
# ASK added -1, correct?
length{K}(b::FullSplineBasis{K}) = b.n+K-1

# Indices of splines naturally range from -K to n-1.
native_index{K}(b::FullSplineBasis{K}, idx::Int) = idx-K-1
linear_index{K}(b::FullSplineBasis{K}, idxn::Int) = idxn+K+1

left(b::FullSplineBasis, idx) = max(b.a, knot(b, native_index(b, idx)))

right{K}(b::FullSplineBasis{K}, idx) = min(b.b, knot(b, native_index(b, idx) + K+1))

waypoints{K}(b::FullSplineBasis{K}, idx) = unique([min(max(knot(b, native_index(b, idx)+i), b.a), b.b) for i = 0:K+1])

stepsize(b::FullSplineBasis) = stepsize(grid(b))

grid(b::FullSplineBasis) = EquispacedGrid(b.n, b.a, b.b)

function active_indices{K}(b::FullSplineBasis{K}, x)
	i = interval(b, x)
	[i+j for j=1:min(K+1,length(b)-i)]
end



function eval_element{K,T}(b::FullSplineBasis{K,T}, idx::Int, x)
	x < left(b) && throw(BoundsError())
	x > right(b) && throw(BoundsError())
	spline_eval(SplineDegree{K}, native_index(b, idx), x, b.a, b.b, stepsize(b))
end



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
native_index(b::NaturalSplineBasis, idx::Int) = idx-1
linear_index(b::NaturalSplineBasis, idxn::Int) = idxn+1

left(b::NaturalSplineBasis, idx) = max(b.a, b.a + (native_index(b, idx)-1)*b.h)

right(b::NaturalSplineBasis, idx) = min(b.b, b.a + (native_index(b, idx)+1)*b.h)

stepsize(b::NaturalSplineBasis) = stepsize(b.grid)

grid(b::NaturalSplineBasis) = EquispacedGrid(b.n, b.a, b.b)

eval_element{K,T}(b::NaturalSplineBasis{K,T}, idx::Int, x) = error("Natural splines not implemented yet. Sorry. Carry on.")


"""
Periodic splines of degree K.
"""
struct PeriodicSplineBasis{K,T} <: SplineBasis{K,T}
	n		::	Int
	a		::	T
	b		::	T

	PeriodicSplineBasis{K,T}(n, a = -one(T), b = one(T)) where {K,T} = new(n, a, b)
end

name(b::PeriodicSplineBasis) = "Periodic splines of degree $(degree(b))"

# Type-unsafe constructor
PeriodicSplineBasis(n, k::Int, a...) = PeriodicSplineBasis(n, SplineDegree{k}, a...)

# This one is type-safe
PeriodicSplineBasis{K,T}(n, ::Type{SplineDegree{K}}, a::T, b::T) = PeriodicSplineBasis{K,T}(n, a, b)

PeriodicSplineBasis{K,T}(n, ::Type{SplineDegree{K}}, ::Type{T} = Float64) = PeriodicSplineBasis{K,T}(n)

instantiate{T}(::Type{PeriodicSplineBasis}, n, ::Type{T}) = PeriodicSplineBasis{3,T}(n)

set_promote_eltype{K,T,S}(b::PeriodicSplineBasis{K,T}, ::Type{S}) = PeriodicSplineBasis{K,S}(b.n, b.a, b.b)

resize{K,T}(b::PeriodicSplineBasis{K,T}, n) = PeriodicSplineBasis{K,T}(n, b.a, b.b)

has_grid(b::PeriodicSplineBasis) = true


length(b::PeriodicSplineBasis) = b.n

grid(b::PeriodicSplineBasis) = PeriodicEquispacedGrid(b.n, b.a, b.b)

# Indices of periodic splines naturally range from 0 to n-1
native_index(b::PeriodicSplineBasis, idx::Int) = idx-1
linear_index{K}(b::PeriodicSplineBasis{K}, idxn::Int) = idxn+1

stepsize(b::PeriodicSplineBasis) = (b.b-b.a)/b.n

period(b::PeriodicSplineBasis) = b.b - b.a

left(b::PeriodicSplineBasis) = b.a

right(b::PeriodicSplineBasis) = b.b

# There is something non-standard about these left and right routines: they currently
# return points outside the interval [a,b] for the first few and last few
# splines. This is due to periodicity: the first spline actually has its support near
# a and near b, i.e., its support restricted to [a,b] consists of two pieces. It is
# easier to use periodicity, and return a single interval near a or b, possibly outside [a,b].
left{K}(b::PeriodicSplineBasis{K}, j::Int) = b.a + (j - 1 - ((K+1) >> 1) ) * stepsize(b)

right{K}(b::PeriodicSplineBasis{K}, j::Int) = b.a + (j - ((K+1) >> 1) + K) * stepsize(b)

# We only return true when x is actually inside the interval, in spite of periodicity
function in_support{K}(b::PeriodicSplineBasis{K}, idx, x)
	period = right(b) - left(b)

	A = left(b) <= x <= right(b)
	B = (left(b, idx) <= x <= right(b, idx)) || (left(b, idx) <= x-period <= right(b, idx)) ||
		(left(b, idx) <= x+period <= right(b, idx))
	A && B
end

rescale{K,T}(s::PeriodicSplineBasis{K,T}, a, b) = PeriodicSplineBasis{K,T}(s.n, a, b)

function eval_element{K,T}(b::PeriodicSplineBasis{K,T}, idx::Int, x)
	while x < left(b)
		x = x + period(b)
	end
	while x > right(b)
		x = x - period(b)
	end
	idxn = native_index(b, idx)

	h = stepsize(b)
	L1 = (K-1) >> 1
	L2 = K-L1
	if idxn <= L1
		z = spline_eval(SplineDegree{K}, idxn-L1-1, x, b.a, b.b, h) + spline_eval(SplineDegree{K}, idxn-L1-1, x-period(b), b.a, b.b, h)
	elseif idxn > b.n - L2
		z = spline_eval(SplineDegree{K}, idxn-L1-1, x, b.a, b.b, h) + spline_eval(SplineDegree{K}, idxn-L1-1, x+period(b), b.a, b.b, h)
	else
		z = spline_eval(SplineDegree{K}, idxn-L1-1, x, b.a, b.b, h)
	end
	z
end


function eval_expansion{K,T <: Number}(b::PeriodicSplineBasis{K}, coef, x::T)
	i = interval(b, x)
	n = length(b)

	L1 = (K-1) >> 1
	L2 = K-L1

	z = zero(T)
	for idxn = i-L1-1:i+L2
		idx = linear_index(b, mod(idxn,n))
		z = z + coef[idx] * eval_element(b, idx, x)
	end

	z
end
