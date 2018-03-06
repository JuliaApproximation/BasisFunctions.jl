# cosineseries.jl

##################
# Cosine series
##################


"""
Cosine series on the interval `[0,1]`.
"""
struct CosineSeries{T} <: Dictionary{T,T}
    n   ::  Int
end

const CosineSpan{A,S,T,D <: CosineSeries} = Span{A,S,T,D}

name(b::CosineSeries) = "Cosine series"

Cosineseries(n::Int) = CosineSeries{Float64}(n)

CosineSeries{T}(n::Int, a::Number, b::Number) where {T} =
    rescale(CosineSeries{T}(n), a, b)

function CosineSeries(n::Int, a::Number, b::Number)
    T = float(promote_type(typeof(a),typeof(b)))
    CosineSeries{T}(n, a, b)
end

instantiate(::Type{CosineSeries}, n, ::Type{T}) where {T} = CosineSeries{T}(n)

dict_promote_domaintype(b::CosineSeries, ::Type{S}) where {S} = CosineSeries{S}(b.n)

resize(b::CosineSeries{T}, n) where {T} = CosineSeries{T}(n)

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

grid(b::CosineSeries{T}) where {T} = MidpointEquispacedGrid(b.n, zero(T), one(T))


##################
# Native indices
##################

const CosineFrequency = NativeIndex{:cosine}

frequency(idxn::CosineFrequency) = value(idxn)

"""
`CosineIndices` defines the map from native indices to linear indices
for a finite number of cosines.
"""
struct CosineIndices <: IndexList{CosineFrequency}
	n	::	Int
end

size(list::CosineIndices) = (list.n,)

getindex(list::CosineIndices, idx::Int) = CosineFrequency(idx-1)
getindex(list::CosineIndices, idxn::CosineFrequency) = value(idxn)+1

ordering(b::CosineSeries) = CosineIndices(length(b))


##################
# Evaluation
##################

domain(b::CosineSeries) = UnitInterval{domaintype(b)}()

support(b::CosineSeries, i) = domain(b)


unsafe_eval_element(b::CosineSeries{T}, idx::CosineFrequency, x) where {T} =
    cospi(T(x) * frequency(idx))

function unsafe_eval_element_derivative(b::CosineSeries{T}, idx::CosineFrequency, x) where {T}
    arg = T(pi) * frequency(idx)
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
    T = dict_codomaintype(s)
    diag = ones(T,length(s))/2
    diag[1] = 1
    DiagonalOperator(s, s, diag)
end

function UnNormalizedGram(s::CosineSpan, oversampling)
    T = dict_codomaintype(s)
    d = T(length_oversampled_grid(dictionary(s), oversampling))/2*ones(T,length(s))
    d[1] = length_oversampled_grid(dictionary(s), oversampling)
    DiagonalOperator(s, s, d)
end
