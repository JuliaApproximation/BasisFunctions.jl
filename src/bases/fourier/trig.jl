
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

coshalf(d::TrigSeries) = coshalf(length(d))
sinhalf(d::TrigSeries) = sinhalf(length(d))
coshalf(n::Int) = 1+nhalf(n)
sinhalf(n::Int) = n-coshalf(n)

similar(d::TrigSeries, ::Type{T}, n::Int) where {T} = TrigSeries{T}(n)

show(io::IO, d::TrigSeries{Float64}) = print(io, "TrigSeries($(length(d)))")
show(io::IO, d::TrigSeries{T}) where T = print(io, "TrigSeries{$(T)}($(length(d)))")


## Properties

# isorthonormal(d::TrigSeries, μ::FourierWeight) = oddlength(d)
# isorthonormal(d::TrigSeries, μ::Weight) = isorthogonal(d, μ) && oddlength(d)
# isorthonormal(d::TrigSeries, μ::DiscreteWeight) =
# 	isorthogonal(d, μ) && isnormalized(μ) && (length(d)==length(μ) || oddlength(d))

hasextension(d::TrigSeries) = false

hasgrid_transform(d::TrigSeries, gb, grid) = false

## Evaluation and indexing

trig_offsets(d::TrigSeries) = trig_offsets(length(d))
trig_offsets(n::Int) = (0,coshalf(n),n)

ordering(d::TrigSeries) = MultilinearIndexList(trig_offsets(d))

function unsafe_eval_element(d::TrigSeries{T}, idx::MultilinearIndex, x) where {T}
	i,j = outerindex(idx), innerindex(idx)
	if i == 1
		cospi(T(2)*(j-1)*x)
	else
		sinpi(T(2)*j*x)
	end
end

## Coefficients

function zeros(::Type{T}, d::TrigSeries) where {T}
    Z = BlockArray{T}(undef,[coshalf(d),sinhalf(d)])
    fill!(Z, 0)
    Z
end
tocoefficientformat(a, d::TrigSeries) = BlockVector(a, [coshalf(d),sinhalf(d)])

hasconstant(b::TrigSeries) = true
coefficients_of_one(b::TrigSeries) = (c=zeros(b); c[1]=1; c)

function expansion_real(dict::Fourier, coef)
	n = length(dict)
	@assert length(coef) == n
	m = iseven(n) ? n+1 : n
	coef2 = zeros(real(eltype(coef)), m)
	nh = coshalf(m)
	coef2[1] = real(coef[1])
	for i in 2:nh
		coef2[i] = real(coef[i]) + real(coef[end-i+2])
	end
	for i in 1:nh-1
		coef2[nh+i] = -imag(coef[1+i]) + imag(coef[end-i+1])
	end
	TrigSeries{eltype(coef2)}(m), coef2
end

function expansion_imag(dict::Fourier, coef)
	n = length(dict)
	@assert length(coef) == n
	m = iseven(n) ? n+1 : n
	coef2 = zeros(real(eltype(coef)), m)
	nh = coshalf(m)
	coef2[1] = imag(coef[1])
	for i in 2:nh
		coef2[i] = imag(coef[i]) + imag(coef[end-i+2])
	end
	for i in 1:nh-1
		coef2[nh+i] = real(coef[1+i]) - real(coef[end-i+1])
	end
	TrigSeries{eltype(coef2)}(m), coef2
end

# converting trigonometric series to Fourier series
function conversion(::Type{T}, src::TrigSeries, dest::Fourier) where {T}
	@assert !isreal(T)
	n = length(src)
	m = length(dest)
	@assert (m >= n)
	A = zeros(T, m, n)
	A[1,1] = 1
	nh = coshalf(n)
	for i in 2:nh
		A[i,i] += one(T)/2
		A[end-i+2,i] += one(T)/2
	end
	for i in 1:sinhalf(n)
		A[1+i,nh+i] = -im*one(T)/2
		A[end-i+1,nh+i] = im*one(T)/2
	end
	ArrayOperator(A, src, dest)
end
