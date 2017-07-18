# cosineseries.jl

##################
# Cosine series
##################


"""
Cosine series on the interval [0,1].
"""
struct CosineSeries{T} <: FunctionSet{T}
    n           ::  Int
end

const CosineSpan{A, F <: CosineSeries} = Span{A,F}

name(b::CosineSeries) = "Cosine series"


CosineSeries(n, ::Type{T} = Float64) where {T} = CosineSeries{T}(n)

CosineSeries(n, a, b, ::Type{T} = promote_type(typeof(a),typeof(b))) where {T} =
    rescale( CosineSeries(n,float(T)), a, b)

instantiate(::Type{CosineSeries}, n, ::Type{T}) where {T} = CosineSeries{T}(n)

set_promote_domaintype(b::CosineSeries, ::Type{S}) where {S} = CosineSeries{S}(b.n)

resize(b::CosineSeries, n) = CosineSeries(n, domaintype(b))

is_basis(b::CosineSeries) = true
is_orthogonal(b::CosineSeries) = true


has_grid(b::CosineSeries) = true
has_derivative(b::CosineSeries) = false #for now
has_antiderivative(b::CosineSeries) = false #for now
has_transform{G <: PeriodicEquispacedGrid}(b::CosineSeries, d::DiscreteGridSpace{G}) = false #for now
has_extension(b::CosineSeries) = true

length(b::CosineSeries) = b.n

left(b::CosineSeries{T}) where {T} = T(0)
left(b::CosineSeries, idx) = left(b)

right(b::CosineSeries{T}) where {T} = T(1)
right(b::CosineSeries, idx) = right(b)

period(b::CosineSeries{T}, idx) where {T} = T(2)

grid(b::CosineSeries) = MidpointEquispacedGrid(b.n, zero(domaintype(b)), one(domaintype(b)))


native_index(b::CosineSeries, idx::Int) = idx-1
linear_index(b::CosineSeries, idxn::Int) = idxn+1

eval_element(b::CosineSeries{T}, idx::Int, x) where {T} = cos(x * T(pi) * (idx-1))

function eval_element_derivative(b::CosineSeries{T}, idx::Int, x) where {T}
    arg = T(pi) * (idx-1)
    -arg * sin(arg * x)
end

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

function Gram(s::CosineSpan; options...)
    T = coeftype(s)
    diag = ones(T,length(s))/2
    diag[1] = 1
    DiagonalOperator(s, s, diag)
end

function UnNormalizedGram(s::CosineSpan, oversampling)
    T = coeftype(s)
    d = T(length_oversampled_grid(s, oversampling))/2*ones(T,length(s))
    d[1] = length_oversampled_grid(s, oversampling)
    DiagonalOperator(s, s, d)
end
