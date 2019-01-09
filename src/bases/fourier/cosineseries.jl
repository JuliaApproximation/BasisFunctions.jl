
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

similar(b::CosineSeries, ::Type{T}, n::Int) where {T} = CosineSeries{T}(n)

isbasis(b::CosineSeries) = true
isorthogonal(b::CosineSeries) = true


has_interpolationgrid(b::CosineSeries) = true
has_derivative(b::CosineSeries) = false #for now
has_antiderivative(b::CosineSeries) = false #for now
has_transform(b::CosineSeries, d::GridBasis{T,G}) where {T,G <: PeriodicEquispacedGrid} = false #for now
has_extension(b::CosineSeries) = true

size(b::CosineSeries) = (b.n,)


support(b::CosineSeries{T}) where {T} = UnitInterval{T}()

period(b::CosineSeries{T}, idx) where {T} = T(2)

interpolation_grid(b::CosineSeries{T}) where {T} =
	MidpointEquispacedGrid(b.n, zero(T), one(T))


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


## Inner products

has_measure(dict::CosineSeries) = true
measure(dict::CosineSeries{T}) where {T} = FourierMeasure{T}()

function gramoperator(dict::CosineSeries; T = coefficienttype(dict), options...)
    diag = ones(T,length(dict))/2
    diag[1] = 1
    DiagonalOperator(diag, src=dict)
end

innerproduct_native(b1::CosineSeries, i::CosineFrequency, b2::CosineSeries, j::CosineFrequency, m::FourierMeasure;
			T = coefficienttype(b1), options...) =
	innerproduct_cosine_full(i, j, T)

function innerproduct_cosine_full(i, j, T)
	if i == j
		if i == CosineFrequency(0)
			one(T)
		else
			one(T)/2
		end
	else
		zero(T)
	end
end
