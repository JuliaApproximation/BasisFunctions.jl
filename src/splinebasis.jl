# splinebasis.jl


export SplineBasis, FullSplineBasis, PeriodicSplineBasis, NaturalSplineBasis, SplineDegree

export degree, interval


immutable SplineDegree{K}
end


spline_eval(::Type{SplineDegree{0}}, i, x, a, b, h) = (x >= a+i*h) && (x < a + (i+1)*h) ? 1 : 0

spline_eval{K}(::Type{SplineDegree{K}}, i, x, a, b, h) = (x - (a+i*h)) / (K*h) * spline_eval(SplineDegree{K-1}, i, x, a, b, h) + (a+(i+K+1)*h - x)/(K*h) * spline_eval(SplineDegree{K-1}, i+1, x, a, b, h)


# Splines of degree K (with equispaced knots)
abstract SplineBasis{K,T} <: AbstractBasis1d{T}

# The degree of the splines
degree{K}(b::SplineBasis{K}) = K

left(b::SplineBasis) = b.a

right(b::SplineBasis) = b.b

# Return the index of the interval between two knots in which x lies, starting from index 0
interval(b::SplineBasis, x) = int(floor( (x-left(b))/stepsize(b) ))

# All splines have compact support
has_compact_support{B <: SplineBasis}(::Type{B}) = True

# Return the i-th knot of the spline, using natural indices
knot(b::SplineBasis, idxn) = left(b) + idxn*stepsize(b)

# The natural grid associated with splines
grid(b::SplineBasis) = b.grid


# The full space of piecewise polynomials of degree K on n subintervals of [a,b]
immutable FullSplineBasis{K,T} <: SplineBasis{K,T}
	a		::	T
	b		::	T
	n		::	Int
	grid	::	EquispacedGrid{T}	# this grid is not suitable for interpolation

	FullSplineBasis(a, b, n) = new(a, b, n, EquispacedGrid(n, a, b))
end

FullSplineBasis{T}(a::T, b::T, n, k) = FullSplineBasis{k,T}(a, b, n)

FullSplineBasis{K,T}(a::T, b::T, n, ::Type{SplineDegree{K}}) = FullSplineBasis{K,T}(a, b, n)

length{K}(b::FullSplineBasis{K}) = b.n+K

# Indices of splines naturally range from -K to n-1
# Functions assuming natural indices take idxn as argument, rather than idx
natural_index{K}(b::FullSplineBasis{K}, idx) = idx-K-1

# convert back from natural index to general index
general_index{K}(b::FullSplineBasis{K}, idxn) = idxn+K+1

left(b::FullSplineBasis, idx) = max(b.a, knot(b, natural_index(b,idx)))

right{K}(b::FullSplineBasis{K}, idx) = min(b.b, knot(b, natural_index(b,idx)+K+1))

waypoints{K}(b::FullSplineBasis{K}, idx) = unique([min(max(knot(b, natural_index(b,idx)+i),b.a),b.b) for i = 0:K+1])

stepsize(b::FullSplineBasis) = stepsize(b.grid)


function active_indices{K}(b::FullSplineBasis{K}, x)
	i = interval(b, x)
	[i+j for j=1:min(K+1,length(b)-i)]
end



function call{K,T}(b::FullSplineBasis{K,T}, idx, x)
	x < left(b) && throw(BoundsError())
	x > right(b) && throw(BoundsError())
	spline_eval(SplineDegree{K}, natural_index(b,idx), x, b.a, b.b, stepsize(b))
end


# Natural splines of degree K
immutable NaturalSplineBasis{K,T} <: SplineBasis{K,T}
	a		::	T
	b		::	T
	n		::	Int
	grid	::	EquispacedGrid{T}

	NaturalSplineBasis(a, b, n) = new(a, b, n, EquispacedGrid(n, a, b))
end

NaturalSplineBasis{T}(a::T, b::T, n, k) = NaturalSplineBasis{k,T}(a, b, n)

NaturalSplineBasis{K,T}(a::T, b::T, n, ::Type{SplineDegree{K}}) = NaturalSplineBasis{K,T}(a, b, n)


length(b::NaturalSplineBasis) = b.n+1

# Indices of natural splines naturally range from 0 to n
natural_index(b::NaturalSplineBasis, idx) = idx-1

left(b::NaturalSplineBasis, idx) = max(b.a, b.a + (natural_index(b, idx)-1)*b.h)

right(b::NaturalSplineBasis, idx) = min(b.b, b.a + (natural_index(b, idx)+1)*b.h)

stepsize(b::NaturalSplineBasis) = stepsize(b.grid)


call{K,T}(b::NaturalSplineBasis{K,T}, idx, x) = error("Natural splines not implemented yet. Sorry. Carry on.")

# Periodic splines of degree K
immutable PeriodicSplineBasis{K,T} <: SplineBasis{K,T}
	a		::	T
	b		::	T
	n		::	Int
	grid	::	PeriodicEquispacedGrid{T}

	PeriodicSplineBasis(a, b, n) = new(a, b, n, PeriodicEquispacedGrid(n, a, b))
end

PeriodicSplineBasis{T}(a::T, b::T, n, k) = PeriodicSplineBasis{k,T}(a,b,n)

PeriodicSplineBasis{K,T}(a::T, b::T, n, ::Type{SplineDegree{K}}) = PeriodicSplineBasis{K,T}(a,b,n)

length(b::PeriodicSplineBasis) = b.n

# Indices of periodic splines naturally range from 0 to n-1
natural_index(b::PeriodicSplineBasis, idx) = idx-1

# convert back from natural index to general index
general_index{K}(b::PeriodicSplineBasis{K}, idxn) = idxn+1

stepsize(b::PeriodicSplineBasis) = stepsize(b.grid)

period(b::PeriodicSplineBasis) = b.b - b.a


function call{K,T}(b::PeriodicSplineBasis{K,T}, idx, x)
	checkbounds(b, idx)
	while x < left(b)
		x = x + period(b)
	end
	while x > right(b)
		x = x - period(b)
	end
	idxn = natural_index(b, idx)

	h = stepsize(b)
	L1 = int(floor((K-1)/2))
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


function call{K,T <: Number}(e::SetExpansion{PeriodicSplineBasis{K}}, x::T)
	i = interval(e.set, x)
	n = length(e.set)

	L1 = int(floor((K-1)/2))
	L2 = K-L1

	z = zero(T)
	for idxn = i-L1-1:i+L2
		idx = general_index(e.set, mod(idxn,n))
		z = z + e.coef[idx] * call(e.set, idx, x)
	end

	z	
end





