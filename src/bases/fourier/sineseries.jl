
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

SineSeries{T}(n::Int, a::Number, b::Number) where {T} =
    rescale(SineSeries{T}(n), a, b)

function SineSeries(n::Int, a::Number, b::Number)
    T = float(promote_type(typeof(a),typeof(b)))
    SineSeries{T}(n, a, b)
end

similar(b::SineSeries, ::Type{T}, n::Int) where {T} = SineSeries{T}(n)

isbasis(b::SineSeries) = true
isorthogonal(b::SineSeries, ::FourierMeasure) = true


hasinterpolationgrid(b::SineSeries) = false
hasderivative(b::SineSeries) = false #for now
hasantiderivative(b::SineSeries) = false #for now
hastransform(b::SineSeries, d::GridBasis{T,G}) where {T,G <: PeriodicEquispacedGrid} = false #for now
hasextension(b::SineSeries) = true

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

function extension_operator(s1::SineSeries, s2::SineSeries; T=op_eltype(s1,s2), options...)
    @assert length(s2) >= length(s1)
    IndexExtensionOperator(s1, s2, 1:length(s1); T=T)
end

function restriction_operator(s1::SineSeries, s2::SineSeries; T=op_eltype(s1,s2), options...)
    @assert length(s2) <= length(s1)
    IndexRestrictionOperator(s1, s2, 1:length(s2); T=T)
end


## Inner products

hasmeasure(dict::SineSeries) = true

measure(dict::SineSeries{T}) where {T} = FourierMeasure{T}()

gramoperator(dict::SineSeries, ::FourierMeasure; T = coefficienttype(dict), options...) =
    ScalingOperator(dict, one(T)/2)

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
