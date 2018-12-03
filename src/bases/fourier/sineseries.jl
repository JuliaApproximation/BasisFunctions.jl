
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

instantiate(::Type{SineSeries}, n, ::Type{T}) where {T} = SineSeries{T}(n)

similar(b::SineSeries, ::Type{T}, n::Int) where {T} = SineSeries{T}(n)

is_basis(b::SineSeries) = true
is_orthogonal(b::SineSeries) = true


has_grid(b::SineSeries) = false
has_derivative(b::SineSeries) = false #for now
has_antiderivative(b::SineSeries) = false #for now
has_transform(b::SineSeries, d::GridBasis{G}) where {G <: PeriodicEquispacedGrid} = false #for now
has_extension(b::SineSeries) = true

size(b::SineSeries) = (b.n,)

period(b::SineSeries{T}, idx) where {T} = T(2)

grid(b::SineSeries{T}) where {T} = EquispacedGrid(b.n, T(0), T(1))

##################
# Native indices
##################

const SineFrequency = NativeIndex{:sine}

frequency(idxn::SineFrequency) = value(idxn)

"""
`SineIndices` defines the map from native indices to linear indices
for a finite number of sines. It is merely the identity map.
"""
struct SineIndices <: IndexList{SineFrequency}
	n	::	Int
end

size(list::SineIndices) = (list.n,)

getindex(list::SineIndices, idx::Int) = SineFrequency(idx)
getindex(list::SineIndices, idxn::SineFrequency) = value(idxn)

ordering(b::SineSeries) = SineIndices(length(b))

##################
# Evaluation
##################

support(b::SineSeries) = UnitInterval{domaintype(b)}()

unsafe_eval_element(b::SineSeries{T}, idx::SineFrequency, x) where {T} =
    sinpi(T(x) * frequency(idx))

function unsafe_eval_element_derivative(b::SineSeries{T}, idx::SineFrequency, x) where {T}
    arg = T(pi) * frequency(idx)
    arg * cos(arg * x)
end

function extension_operator(s1::SineSeries, s2::SineSeries; options...)
    @assert length(s2) >= length(s1)
    IndexExtensionOperator(s1, s2, 1:length(s1))
end

function restriction_operator(s1::SineSeries, s2::SineSeries; options...)
    @assert length(s2) <= length(s1)
    IndexRestrictionOperator(s1, s2, 1:length(s2))
end


Gram(s::SineSeries; options...) = ScalingOperator(s, s, one(coefficienttype(s))/2)
