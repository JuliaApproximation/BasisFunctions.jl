# sineseries.jl

############################################
# Sine series
############################################


"""
Sine series on the interval `[0,1]`.
"""
struct SineSeries{T} <: Dictionary{T,T}
    n   ::  Int
end

const SineSpan{A,S,T,D <: SineSeries} = Span{A,S,T,D}

name(b::SineSeries) = "Sine series"

SineSeries(n::Int) = SineSeries{Float64}(n)

SineSeries{T}(n::Int, a::Number, b::Number) where {T} =
    rescale(SineSeries{T}(n), a, b)

function SineSeries(n::Int, a::Number, b::Number)
    T = float(promote_type(typeof(a),typeof(b)))
    SineSeries{T}(n, a, b)
end

instantiate(::Type{SineSeries}, n, ::Type{T}) where {T} = SineSeries{T}(n)

dict_promote_domaintype(b::SineSeries, ::Type{S}) where {S} = SineSeries{S}(b.n)

resize(b::SineSeries{T}, n) where {T} = SineSeries{T}(n)

is_basis(b::SineSeries) = true
is_orthogonal(b::SineSeries) = true


has_grid(b::SineSeries) = false
has_derivative(b::SineSeries) = false #for now
has_antiderivative(b::SineSeries) = false #for now
has_transform{G <: PeriodicEquispacedGrid}(b::SineSeries, d::DiscreteGridSpace{G}) = false #for now
has_extension(b::SineSeries) = true

length(b::SineSeries) = b.n

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

function apply!(op::Extension, dest::SineSeries, src::SineSeries, coef_dest, coef_src)
    @assert length(dest) > length(src)

    for i = 1:length(src)
        coef_dest[i] = coef_src[i]
    end
    for i = length(src)+1:length(dest)
        coef_dest[i] = 0
    end
    coef_dest
end


function apply!(op::Restriction, dest::SineSeries, src::SineSeries, coef_dest, coef_src)
    @assert length(dest) < length(src)

    for i = 1:length(dest)
        coef_dest[i] = coef_src[i]
    end
    coef_dest
end

Gram(s::SineSpan; options...) = ScalingOperator(s, s, one(coeftype(s))/2)
