
"""
A trigonometric series consists of sines and cosines on the interval `[0,1]`.

The span of `TrigSeries(n)` is the same as that of `Fourier(n)`. To that end,
`TrigSeries(n)` has `ceil(n/2)` cosine and `floor(n/2)` sine functions.
"""
struct TrigSeries{T} <: FourierLike{T,T}
	n	::	Int
end

TrigSeries(n) = TrigSeries{Float64}(n)

convert(::Type{TrigSeries{T}}, d::TrigSeries) where {T} = TrigSeries{T}(d.n)

size(d::TrigSeries) = (d.n,)

sinhalf(d::TrigSeries) = sinhalf(length(d))
coshalf(d::TrigSeries) = coshalf(length(d))
sinhalf(n::Int) = nhalf(n)
coshalf(n::Int) = n-nhalf(n)

similar(d::TrigSeries, ::Type{T}, n::Int) where {T} = TrigSeries{T}(n)

show(io::IO, d::TrigSeries{Float64}) = print(io, "TrigSeries($(length(d)))")
show(io::IO, d::TrigSeries{T}) where T = print(io, "TrigSeries{$(T)}($(length(d)))")


## Properties

# isorthonormal(d::TrigSeries, μ::FourierWeight) = oddlength(d)
# isorthonormal(d::TrigSeries, μ::Weight) = isorthogonal(d, μ) && oddlength(d)
# isorthonormal(d::TrigSeries, μ::DiscreteWeight) =
# 	isorthogonal(d, μ) && isnormalized(μ) && (length(d)==length(μ) || oddlength(d))

hasextension(d::TrigSeries) = false

unsafe_eval_element(dict::TrigSeries{T}, idx::SincIndex, x) where {T} =
    dirichlet_kernel(dict.n, T(x), value(idx))
