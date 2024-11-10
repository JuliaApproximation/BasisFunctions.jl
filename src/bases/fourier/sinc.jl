"""
A basis of shifted periodic sinc functions on `[0,1]`.

The space of this basis is the same as that of Fourier series, except that
expansions are real-valued by default.
"""
struct PeriodicSincFunctions{T} <: FourierLike{T,T}
    n   ::  Int
end

PeriodicSincFunctions(n::Int) = PeriodicSincFunctions{Float64}(n)

show(io::IO, b::PeriodicSincFunctions{Float64}) =
    print(io, "PeriodicSincFunctions($(length(b)))")
show(io::IO, b::PeriodicSincFunctions{T}) where T =
    print(io, "PeriodicSincFunctions{$(T)}($(length(b)))")

similar(b::PeriodicSincFunctions, ::Type{T}, n::Int) where {T} = PeriodicSincFunctions{T}(n)
isequaldict(b1::PeriodicSincFunctions, b2::PeriodicSincFunctions) = length(b1)==length(b2)

size(b::PeriodicSincFunctions) = (b.n,)

hasderivative(b::PeriodicSincFunctions) = false #for now
hasantiderivative(b::PeriodicSincFunctions) = false #for now

hasconstant(b::PeriodicSincFunctions) = true
coefficients_of_one(b::PeriodicSincFunctions) = ones(numtype(b), size(b))

"Indices of translated sinc functions start at 0."
struct SincIndex <: AbstractShiftedIndex{1}
	value	::	Int
end

ordering(d::PeriodicSincFunctions) = ShiftedIndexList(length(d), SincIndex)

## Evaluation

"The Dirichlet kernel is the periodic sinc function."
dirichlet_kernel(n, x) = (abs(x) < 100eps(x)) || (abs(1-x) < 100eps(x)) ? one(x) : one(x)/n*sin( n*(pi*x)) / sin(pi*x)
"Shifts of the Dirichlet kernel."
dirichlet_kernel(n, x, k) = dirichlet_kernel(n, (n*x-k)/n)

unsafe_eval_element(dict::PeriodicSincFunctions{T}, idx::SincIndex, x) where {T} =
    dirichlet_kernel(dict.n, T(x), value(idx))

transform_from_grid(T, src::GridBasis, dest::PeriodicSincFunctions, grid; options...) =
    IdentityOperator{T}(src, dest)
transform_to_grid(T, src::PeriodicSincFunctions, dest::GridBasis, grid; options...) =
    IdentityOperator{T}(src, dest)

function evaluation(::Type{T}, dict::PeriodicSincFunctions, gb::GridBasis, grid; options...) where {T}
	if iscompatiblegrid(dict, grid)
		transform_to_grid(T, dict, gb, grid; options...)
	else
		default_evaluation(T, dict, gb; options...)
	end
end

dict_innerproduct_native(d1::PeriodicSincFunctions{T}, idx1, d2::PeriodicSincFunctions, idx2, μ::FourierWeight; options...) where {T} =
	idx1 == idx2 ? one(T)/length(d1) : zero(T)

span_issubset(d1::PeriodicSincFunctions{T}, d2::Fourier{T}) where {T} =
	length(d1) < length(d2) || (length(d1)==length(d2) && oddlength(d2))
