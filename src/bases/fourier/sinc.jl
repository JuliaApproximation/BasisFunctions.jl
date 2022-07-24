"""
A basis of periodic sinc functions on [0,1].

The space of this basis is the same as that of Fourier series.
"""
struct PeriodicSincFunctions{T} <: Dictionary{T,T}
    n   ::  Int
end

PeriodicSincFunctions(n::Int) = PeriodicSincFunctions{Float64}(n)

show(io::IO, b::PeriodicSincFunctions{Float64}) =
    print(io, "PeriodicSincFunctions($(length(b)))")
show(io::IO, b::PeriodicSincFunctions{T}) where T =
    print(io, "PeriodicSincFunctions{$(T)}($(length(b)))")

similar(b::PeriodicSincFunctions, ::Type{T}, n::Int) where {T} = PeriodicSincFunctions{T}(n)

size(b::PeriodicSincFunctions) = (b.n,)
support(b::PeriodicSincFunctions{T}) where {T} = UnitInterval{T}()

hasmeasure(b::PeriodicSincFunctions) = true
measure(b::PeriodicSincFunctions{T}) where T = FourierWeight{T}()

isbasis(b::PeriodicSincFunctions) = true
isorthogonal(b::PeriodicSincFunctions, ::FourierWeight) = true


hasinterpolationgrid(b::PeriodicSincFunctions) = true
hasderivative(b::PeriodicSincFunctions) = false #for now
hasantiderivative(b::PeriodicSincFunctions) = false #for now


period(b::PeriodicSincFunctions{T}, idx) where {T} = T(2)

interpolation_grid(b::PeriodicSincFunctions{T}) where {T} = FourierGrid{T}(b.n)

## Evaluation

dirichlet_kernel(n, x) = (abs(x) < 100eps(x)) || (abs(1-x) < 100eps(x)) ? one(x) : one(x)/n*sin( n*(pi*x)) / sin(pi*x)
dirichlet_kernel(n, x, k) = dirichlet_kernel(n, (n*x-k)/n)

unsafe_eval_element(dict::PeriodicSincFunctions{T}, idx, x) where {T} =
    dirichlet_kernel(dict.n, T(x), idx-1)

iscompatible(dict::PeriodicSincFunctions, grid) = iscompatible(Fourier(length(dict)), grid)

hasgrid_transform(dict::PeriodicSincFunctions, gb, grid) =
    iscompatible(dict, grid)

transform_from_grid(T, src::GridBasis, dest::PeriodicSincFunctions, grid; options...) =
    IdentityOperator{T}(src, dest)

transform_to_grid(T, src::PeriodicSincFunctions, dest::GridBasis, grid; options...) =
    IdentityOperator{T}(src, dest)

function evaluation(::Type{T}, dict::PeriodicSincFunctions, gb::GridBasis, grid; options...) where {T}
	if iscompatible(dict, grid)
		transform_to_grid(T, dict, gb, grid; options...)
	else
		dense_evaluation(T, dict, gb; options...)
	end
end
