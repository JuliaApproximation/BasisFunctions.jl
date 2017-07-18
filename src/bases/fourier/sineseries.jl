# sineseries.jl

############################################
# Sine series
############################################


"""
Sine series on the interval [0,1].
"""
struct SineSeries{T} <: FunctionSet{T}
    n           ::  Int
end

const SineSpan{A, F <: SineSeries} = Span{A,F}

name(b::SineSeries) = "Sine series"


SineSeries(n, ::Type{T} = Float64) where {T} = SineSeries{T}(n)

SineSeries(n, a, b, ::Type{T} = promote_type(typeof(a),typeof(b))) where {T} = rescale( SineSeries(n,float(T)), a, b)

instantiate(::Type{SineSeries}, n, ::Type{T}) where {T} = SineSeries{T}(n)

set_promote_domaintype(b::SineSeries, ::Type{S}) where {S} = SineSeries{S}(b.n)

resize(b::SineSeries, n) = SineSeries(n, domaintype(b))

is_basis(b::SineSeries) = true
is_orthogonal(b::SineSeries) = true


has_grid(b::SineSeries) = false
has_derivative(b::SineSeries) = false #for now
has_antiderivative(b::SineSeries) = false #for now
has_transform{G <: PeriodicEquispacedGrid}(b::SineSeries, d::DiscreteGridSpace{G}) = false #for now
has_extension(b::SineSeries) = true

length(b::SineSeries) = b.n

left(b::SineSeries{T}) where {T} = T(0)
left(b::SineSeries, idx) = left(b)

right(b::SineSeries{T}) where {T} = T(1)
right(b::SineSeries, idx) = right(b)

period(b::SineSeries{T}, idx) where {T} = T(2)

grid(b::SineSeries{T}) where {T} = EquispacedGrid(b.n, T(0), T(1))


eval_element(b::SineSeries{T}, idx::Int, x) where {T} = sin(x * T(pi) * idx)

function eval_element_derivative(b::SineSeries{T}, idx::Int, x) where {T}
    arg = T(pi) * idx
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
