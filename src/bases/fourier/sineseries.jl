
############################################
# Sine series
############################################


"""
Sine series on the interval `[0,1]`.
"""
struct SineSeries{T} <: Dictionary{T,T}
    n   ::  Int
end

name(b::SineSeries) = "Sine series"

SineSeries(n::Int) = SineSeries{Float64}(n)

similar(b::SineSeries, ::Type{T}, n::Int) where {T} = SineSeries{T}(n)

isbasis(b::SineSeries) = true
isorthogonal(b::SineSeries, ::FourierMeasure) = true


hasinterpolationgrid(b::SineSeries) = false
hasderivative(b::SineSeries) = false #for now
hasantiderivative(b::SineSeries) = false #for now
hastransform(b::SineSeries, d::GridBasis{T,G}) where {T,G <: PeriodicEquispacedGrid} = false #for now
hasextension(b::SineSeries) = true

extension(::Type{T}, src::SineSeries, dest::SineSeries; options...) where {T} = IndexExtension{T}(src, dest, 1:length(src))
restriction(::Type{T}, src::SineSeries, dest::SineSeries; options...) where {T} = IndexRestriction{T}(src, dest, 1:length(dest))

size(b::SineSeries) = (b.n,)

period(b::SineSeries{T}, idx) where {T} = T(2)

interpolation_grid(b::SineSeries{T}) where {T} = EquispacedGrid(b.n, T(0), T(1))

##################
# Native indices
##################

const SineFrequency = NativeIndex{:sine}

frequency(idxn::SineFrequency) = value(idxn)

Base.show(io::IO, idx::SineFrequency) =
	print(io, "Sine frequency $(value(idx))")

ordering(b::SineSeries) = NativeIndexList{:sine}(length(b))


##################
# Evaluation
##################

support(b::SineSeries{T}) where {T} = UnitInterval{T}()

unsafe_eval_element(b::SineSeries{T}, idx::SineFrequency, x) where {T} =
    sinpi(T(x) * frequency(idx))

function unsafe_eval_element_derivative(b::SineSeries{T}, idx::SineFrequency, x) where {T}
    arg = T(pi) * frequency(idx)
    arg * cos(arg * x)
end

##################
# Differentiation
##################

derivative_dict(Φ::CosineSeries{T}, order::Int) where {T} =
	iseven(order) ? Φ : SineSeries{T}(length(Φ)-1)

diff_scaling_function(Φ::CosineSeries{T}, idx::Int, symbol) where {T} = symbol(T(π)*idx)

function differentiation(::Type{T}, src::CosineSeries, dest::CosineSeries, order::Int; options...) where {T}
	if orderiszero(order)
		@assert src==dest
		IdentityOperator{T}(src)
	else
		@assert iseven(order)
		sign = (-1)^(order>>1)
		_pseudodifferential_operator(T, src, dest, x->sign*x^order; options...)
	end
end

function differentiation(::Type{T}, src::CosineSeries, dest::SineSeries, order::Int; options...) where {T}
	@assert isodd(order)
	sign = (-1)^((order-1)>>1)
	_pseudodifferential_operator(T, src, dest, x->sign*x^order; options...)
end



## Inner products

hasmeasure(dict::SineSeries) = true

measure(dict::SineSeries{T}) where {T} = FourierMeasure{T}()

gram(::Type{T}, dict::SineSeries, ::FourierMeasure; options...) where {T} = ScalingOperator(one(T)/2, dict)

function innerproduct_native(b1::SineSeries, i::SineFrequency, b2::SineSeries, j::SineFrequency, m::FourierMeasure;
			T = coefficienttype(b1), quad = :analytic, options...)
	if quad == :analytic
		innerproduct_sine_full(i, j, T)
	else
		innerproduct1(b1, i, b2, j, m)
	end
end

function innerproduct_sine_full(i, j, T)
	if i == j
		one(T)/2
	else
		zero(T)
	end
end
