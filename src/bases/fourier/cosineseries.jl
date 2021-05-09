
##################
# Cosine series
##################


"""
Cosine series on the interval `[0,1]`.
"""
struct CosineSeries{T} <: Dictionary{T,T}
    n   ::  Int
end

CosineSeries(n::Int) = CosineSeries{Float64}(n)

name(::CosineSeries) = "Cosine series"

similar(b::CosineSeries, ::Type{T}, n::Int) where {T} = CosineSeries{T}(n)

isbasis(b::CosineSeries) = true
isorthogonal(b::CosineSeries, ::FourierWeight) = true


hasinterpolationgrid(b::CosineSeries) = true
hasderivative(b::CosineSeries) = false #for now
hasantiderivative(b::CosineSeries) = false #for now
hastransform(b::CosineSeries, d::GridBasis{T,G}) where {T,G <: PeriodicEquispacedGrid} = false #for now
hasextension(b::CosineSeries) = true

size(b::CosineSeries) = (b.n,)


support(b::CosineSeries{T}) where {T} = UnitInterval{T}()

period(b::CosineSeries{T}, idx) where {T} = T(2)

interpolation_grid(b::CosineSeries{T}) where {T} =
	MidpointEquispacedGrid(b.n, zero(T), one(T))

show(io::IO, b::CosineSeries{Float64}) = print(io, "CosineSeries($(length(b)))")

##################
# Native indices
##################

struct CosineFrequency <: AbstractShiftedIndex{1}
	value	::	Int
end

Base.show(io::IO, idx::CosineFrequency) =
	print(io, "Cosine frequency $(value(idx))")

frequency(idxn::CosineFrequency) = value(idxn)

ordering(b::CosineSeries) = ShiftedIndexList(length(b), CosineFrequency)


##################
# Evaluation
##################


unsafe_eval_element(b::CosineSeries{T}, idx::CosineFrequency, x) where {T} =
    cospi(T(x) * frequency(idx))

function unsafe_eval_element_derivative(b::CosineSeries{T}, idx::CosineFrequency, x, order) where {T}
	@assert order == 1
    arg = T(pi) * frequency(idx)
    -arg * sin(arg * x)
end



extension(::Type{T}, src::CosineSeries, dest::CosineSeries; options...) where {T} = IndexExtension{T}(src, dest, 1:length(src))
restriction(::Type{T}, src::CosineSeries, dest::CosineSeries; options...) where {T} = IndexRestriction{T}(src, dest, 1:length(dest))


## Inner products

hasmeasure(dict::CosineSeries) = true
measure(dict::CosineSeries{T}) where {T} = FourierWeight{T}()

function gram(::Type{T}, dict::CosineSeries, ::FourierWeight; options...) where {T}
    diag = ones(T,length(dict))/2
    diag[1] = 1
    DiagonalOperator(dict, diag)
end

innerproduct_native(b1::CosineSeries, i::CosineFrequency, b2::CosineSeries, j::CosineFrequency, m::FourierWeight;
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
