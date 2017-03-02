# sineseries.jl

############################################
# Sine series
############################################


"""
Sine series on the interval [0,1].
"""
immutable SineSeries{T} <: FunctionSet1d{T}
    n           ::  Int

    SineSeries(n) = new(n)
end

name(b::SineSeries) = "Sine series"


SineSeries{T}(n, ::Type{T} = Float64) = SineSeries{T}(n)

SineSeries{T}(n, a, b, ::Type{T} = promote_type(typeof(a),typeof(b))) = rescale( SineSeries(n,floatify(T)), a, b)

instantiate{T}(::Type{SineSeries}, n, ::Type{T}) = SineSeries{T}(n)

set_promote_eltype{T,S}(b::SineSeries{T}, ::Type{S}) = SineSeries{S}(b.n)

resize(b::SineSeries, n) = SineSeries(n, eltype(b))

is_basis(b::SineSeries) = true
is_orthogonal(b::SineSeries) = true


has_grid(b::SineSeries) = false
has_derivative(b::SineSeries) = false #for now
has_antiderivative(b::SineSeries) = false #for now
has_transform{G <: PeriodicEquispacedGrid}(b::SineSeries, d::DiscreteGridSpace{G}) = false #for now
has_extension(b::SineSeries) = true

length(b::SineSeries) = b.n

left{T}(b::SineSeries{T}) = T(0)
left(b::SineSeries, idx) = left(b)

right{T}(b::SineSeries{T}) = T(1)
right(b::SineSeries, idx) = right(b)

period{T}(b::SineSeries{T}, idx) = T(2)

grid{T}(b::SineSeries{T}) = EquispacedGrid(b.n, T(0), T(1))


eval_element{T}(b::SineSeries{T}, idx::Int, x) = sin(x * T(pi) * idx)


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

Gram{T}(b::SineSeries{T}; options...) = ScalingOperator(b, b ,T(1)/2)
