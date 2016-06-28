# bf_splines.jl


typealias SplineDegree{K} Val{K}


spline_eval(::Type{SplineDegree{0}}, i, x, a, b, h) = (x >= a+i*h) && (x < a + (i+1)*h) ? 1 : 0

spline_eval{K}(::Type{SplineDegree{K}}, i, x, a, b, h) = (x - (a+i*h)) / (K*h) * spline_eval(SplineDegree{K-1}, i, x, a, b, h) + (a+(i+K+1)*h - x)/(K*h) * spline_eval(SplineDegree{K-1}, i+1, x, a, b, h)


# Splines of degree K (with equispaced knots only...)
abstract SplineBasis{K,T} <: AbstractBasis1d{T}

"The degree of the splines."
degree{K}(b::SplineBasis{K}) = K

left(b::SplineBasis) = b.a

right(b::SplineBasis) = b.b

"Return the index of the interval between two knots in which x lies, starting from index 0."
interval(b::SplineBasis, x) = round(Int, floor( (x-left(b))/stepsize(b) ))

# All splines have compact support
has_compact_support{B <: SplineBasis}(::Type{B}) = True

"Return the i-th knot of the spline, using natural indices."
knot(b::SplineBasis, idxn) = left(b) + idxn*stepsize(b)



"""
The full space of piecewise polynomials of degree K on n subintervals of [a,b].
"""
immutable FullSplineBasis{K,T} <: SplineBasis{K,T}
	n		::	Int
	a		::	T
	b		::	T

	FullSplineBasis(n, a = -one(T), b = one(T)) = new(n, a, b)
end

@compat (b::FullSplineBasis)(x...) = call_set(b, x...)

name(b::FullSplineBasis) = "Full splines of degree $(degree(b))"


FullSplineBasis{K,T}(n, a::T, b::T, ::Type{SplineDegree{K}}) = FullSplineBasis{K,T}(n, a, b)

FullSplineBasis{T}(n, a::T, b::T, k::Int = 3) = FullSplineBasis(n, a, b, SplineDegree{k})

FullSplineBasis{K,T}(n, ::Type{SplineDegree{K}} = SplineDegree{3}, ::Type{T} = Float64) = FullSplineBasis{K,T}(n)

instantiate{T}(::Type{FullSplineBasis}, n, ::Type{T}) = FullSplineBasis{3,T}(n-3)

# Full splines to not have an interpolation grid
#has_grid(b::FullSplineBasis) = true

length{K}(b::FullSplineBasis{K}) = b.n+K

# Indices of splines naturally range from -K to n-1.
natural_index{K}(b::FullSplineBasis{K}, idx) = idx-K-1
logical_index{K}(b::FullSplineBasis{K}, idxn) = idxn+K+1

left(b::FullSplineBasis, idx) = max(b.a, knot(b, natural_index(b, idx)))

right{K}(b::FullSplineBasis{K}, idx) = min(b.b, knot(b, natural_index(b, idx) + K+1))

waypoints{K}(b::FullSplineBasis{K}, idx) = unique([min(max(knot(b, natural_index(b, idx)+i), b.a), b.b) for i = 0:K+1])

stepsize(b::FullSplineBasis) = stepsize(grid(b))

grid(b::FullSplineBasis) = EquispacedGrid(b.n, b.a, b.b)

function active_indices{K}(b::FullSplineBasis{K}, x)
	i = interval(b, x)
	[i+j for j=1:min(K+1,length(b)-i)]
end



function call_element{K,T}(b::FullSplineBasis{K,T}, idx::Int, x)
	x < left(b) && throw(BoundsError())
	x > right(b) && throw(BoundsError())
	spline_eval(SplineDegree{K}, natural_index(b, idx), x, b.a, b.b, stepsize(b))
end



# Natural splines of degree K
immutable NaturalSplineBasis{K,T} <: SplineBasis{K,T}
	n		::	Int
	a		::	T
	b		::	T

	NaturalSplineBasis(n, a = -one(T), b = one(T)) = new(n, a, b)
end

@compat (b::NaturalSplineBasis)(x...) = call_set(b, x...)

name(b::NaturalSplineBasis) = "Natural splines of degree $(degree(b))"


NaturalSplineBasis{K,T}(n, a::T, b::T, ::Type{SplineDegree{K}}) = NaturalSplineBasis{K,T}(n, a, b)

NaturalSplineBasis{T}(n, a::T, b::T, k::Int = 3) = NaturalSplineBasis(n, a, b, SplineDegree{k})

NaturalSplineBasis{K,T}(n, ::Type{SplineDegree{K}} = SplineDegree{3}, ::Type{T} = Float64) = NaturalSplineBasis{K,T}(n)

instantiate{T}(::Type{NaturalSplineBasis}, n, ::Type{T}) = NaturalSplineBasis{3,T}(n)

has_grid(b::NaturalSplineBasis) = true



length(b::NaturalSplineBasis) = b.n+1

# Indices of natural splines naturally range from 0 to n
natural_index(b::NaturalSplineBasis, idx) = idx-1
logical_index(b::NaturalSplineBasis, idxn) = idxn+1

left(b::NaturalSplineBasis, idx) = max(b.a, b.a + (natural_index(b, idx)-1)*b.h)

right(b::NaturalSplineBasis, idx) = min(b.b, b.a + (natural_index(b, idx)+1)*b.h)

stepsize(b::NaturalSplineBasis) = stepsize(b.grid)

grid(b::NaturalSplineBasis) = EquispacedGrid(b.n, b.a, b.b)

call_element{K,T}(b::NaturalSplineBasis{K,T}, idx::Int, x) = error("Natural splines not implemented yet. Sorry. Carry on.")


"""
Periodic splines of degree K.
"""
immutable PeriodicSplineBasis{K,T} <: SplineBasis{K,T}
	n		::	Int
	a		::	T
	b		::	T

	PeriodicSplineBasis(n, a = -one(T), b = one(T)) = new(n, a, b)
end

@compat (b::PeriodicSplineBasis)(x...) = call_set(b, x...)

name(b::PeriodicSplineBasis) = "Periodic splines of degree $(degree(b))"

# Type-unsafe constructor
PeriodicSplineBasis(n, k::Int, a...) = PeriodicSplineBasis(n, SplineDegree{k}, a...)

# This one is type-safe
PeriodicSplineBasis{K,T}(n, ::Type{SplineDegree{K}}, a::T, b::T) = PeriodicSplineBasis{K,T}(n, a, b)

PeriodicSplineBasis{K,T}(n, ::Type{SplineDegree{K}}, ::Type{T} = Float64) = PeriodicSplineBasis{K,T}(n)

instantiate{T}(::Type{PeriodicSplineBasis}, n, ::Type{T}) = PeriodicSplineBasis{3,T}(n)

promote_eltype{K,T,S}(b::PeriodicSplineBasis{K,T}, ::Type{S}) = PeriodicSplineBasis{K,promote_type(T,S)}(b.n, b.a, b.b)

resize{K,T}(b::PeriodicSplineBasis{K,T}, n) = PeriodicSplineBasis{K,T}(n, b.a, b.b)

has_grid(b::PeriodicSplineBasis) = true


length(b::PeriodicSplineBasis) = b.n

grid(b::PeriodicSplineBasis) = PeriodicEquispacedGrid(b.n, b.a, b.b)

# Indices of periodic splines naturally range from 0 to n-1
natural_index(b::PeriodicSplineBasis, idx) = idx-1
logical_index{K}(b::PeriodicSplineBasis{K}, idxn) = idxn+1

stepsize(b::PeriodicSplineBasis) = (b.b-b.a)/b.n

period(b::PeriodicSplineBasis) = b.b - b.a

left(b::PeriodicSplineBasis) = b.a

right(b::PeriodicSplineBasis) = b.b

left{K}(b::PeriodicSplineBasis{K}, j::Int) = b.a + (j - 1 - ((K+1) >> 1) ) * stepsize(b)

right{K}(b::PeriodicSplineBasis{K}, j::Int) = b.a + (j - ((K+1) >> 1) + K) * stepsize(b)

rescale{K,T}(s::PeriodicSplineBasis{K,T}, a, b) = PeriodicSplineBasis{K,T}(s.n, a, b)

function call_element{K,T}(b::PeriodicSplineBasis{K,T}, idx::Int, x)
	checkbounds(b, idx)
	while x < left(b)
		x = x + period(b)
	end
	while x > right(b)
		x = x - period(b)
	end
	idxn = natural_index(b, idx)

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


function call_expansion{K,T <: Number}(b::PeriodicSplineBasis{K}, coef, x::T)
	i = interval(b, x)
	n = length(b)

	L1 = (K-1) >> 1
	L2 = K-L1

	z = zero(T)
	for idxn = i-L1-1:i+L2
		idx = logical_index(b, mod(idxn,n))
		z = z + coef[idx] * call_set(b, idx, x)
	end

	z
end
