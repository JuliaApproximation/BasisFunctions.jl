# cosineseries.jl

############################################
# Cosine series
############################################


"""
Cosine series on the interval [0,1].
"""
immutable CosineSeries{T} <: AbstractBasis1d{T}
    n           ::  Int

    CosineSeries(n) = new(n)
end

name(b::CosineSeries) = "Cosine series"


CosineSeries{T}(n, ::Type{T} = Float64) = CosineSeries{T}(n)

CosineSeries{T}(n, a, b, ::Type{T} = promote_type(typeof(a),typeof(b))) =
    rescale( CosineSeries(n,floatify(T)), a, b)

instantiate{T}(::Type{CosineSeries}, n, ::Type{T}) = CosineSeries{T}(n)

promote_eltype{T,S}(b::CosineSeries{T}, ::Type{S}) = CosineSeries{promote_type(T,S)}(b.n)

resize(b::CosineSeries, n) = CosineSeries(n, eltype(b))



has_grid(b::CosineSeries) = true
has_derivative(b::CosineSeries) = false #for now
has_antiderivative(b::CosineSeries) = false #for now
has_transform{G <: PeriodicEquispacedGrid}(b::CosineSeries, d::DiscreteGridSpace{G}) = false #for now
has_extension(b::CosineSeries) = true

length(b::CosineSeries) = b.n

left(b::CosineSeries) = 0
left(b::CosineSeries, idx) = left(b)

right(b::CosineSeries) = 1
right(b::CosineSeries, idx) = right(b)

period(b::CosineSeries, idx) = 2

grid(b::CosineSeries) = MidpointEquispacedGrid(b.n, zero(numtype(b)), one(numtype(b)))


natural_index(b::CosineSeries, idx) = idx-1
logical_index(b::CosineSeries, idxn) = idxn+1

call_element{T}(b::CosineSeries{T}, idx::Int, x) = cos(x * T(pi) * (idx-1))


function apply!(op::Extension, dest::CosineSeries, src::CosineSeries, coef_dest, coef_src)
    @assert length(dest) > length(src)

    for i = 1:length(src)
        coef_dest[i] = coef_src[i]
    end
    for i = length(src)+1:length(dest)
        coef_dest[i] = 0
    end
    coef_dest
end


function apply!(op::Restriction, dest::CosineSeries, src::CosineSeries, coef_dest, coef_src)
    @assert length(dest) < length(src)

    for i = 1:length(dest)
        coef_dest[i] = coef_src[i]
    end
    coef_dest
end
