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

name(b::CosineSeries) = "Cosine series"

CosineSeries(n::Int) = CosineSeries{Float64}(n)

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
has_transform(b::CosineSeries, d::GridBasis{G}) where {G <: PeriodicEquispacedGrid} = false #for now
has_extension(b::CosineSeries) = true

length(b::CosineSeries) = b.n


support(b::CosineSeries{T}) where {T} = UnitInterval{T}()

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


unsafe_eval_element(b::CosineSeries{T}, idx::CosineFrequency, x) where {T} =
    cospi(T(x) * frequency(idx))

function unsafe_eval_element_derivative(b::CosineSeries{T}, idx::CosineFrequency, x) where {T}
    arg = T(pi) * frequency(idx)
    -arg * sin(arg * x)
end

function extension_operator(s1::CosineSeries, s2::CosineSeries; options...)
    @assert length(s2) >= length(s1)
    IndexExtensionOperator(s1, s2, 1:length(s1))
end

function restriction_operator(s1::CosineSeries, s2::CosineSeries; options...)
    @assert length(s2) <= length(s1)
    IndexRestrictionOperator(s1, s2, 1:length(s2))
end

function Gram(s::CosineSeries; options...)
    T = codomaintype(s)
    diag = ones(T,length(s))/2
    diag[1] = 1
    DiagonalOperator(s, s, diag)
end

function UnNormalizedGram(s::CosineSeries, oversampling)
    T = codomaintype(s)
    d = T(length_oversampled_grid(s, oversampling))/2*ones(T,length(s))
    d[1] = length_oversampled_grid(s, oversampling)
    DiagonalOperator(s, s, d)
end
