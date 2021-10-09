
export RationalFunctions,
    RationalBasisFunction

"A basis of functions of the form `1/(z-z_j)` where `z_j` are the poles."
struct RationalFunctions{T} <: Dictionary{T,T}
    poles   :: Vector{T}
end

RationalFunctions(n::Int) = RationalFunctions{Float64}(n)
RationalFunctions{T}(n::Int) where {T} = RationalFunctions(rand(T, n))

size(dict::RationalFunctions) = (length(dict.poles),)

pole(dict::RationalFunctions, idx) = dict.poles[idx]

unsafe_eval_element(dict::RationalFunctions, idx::Int, x) = 1/(x-dict.poles[idx])

support(dict::RationalFunctions{T}) where {T} = DomainSets.FullSpace{T}()

function unsafe_eval_element_derivative(dict::RationalFunctions, idx, x, order)
    @assert order == 1
    -1/(x-dict.poles[idx])^2
end

function similar(dict::RationalFunctions, ::Type{T}, n::Int) where {T}
    @assert n == length(dict)
    RationalFunctions{T}(dict.poles)
end


struct RationalBasisFunction{T} <: Polynomial{T}
    pole    ::  T
end

pole(r::RationalBasisFunction) = r.pole

RationalBasisFunction{T}(r::RationalBasisFunction) where {T} = RationalBasisFunction{T}(r.pole)

show(io::IO, d::RationalBasisFunction{T}) where {T<:Real} =
    print(io, "1/(x-$(pole(r)))  (rational function)")
show(io::IO, d::RationalBasisFunction{T}) where {T<:Complex} =
    print(io, "1/(z-$(pole(r)))  (rational function)")

convert(::Type{TypedFunction{T,T}}, r::RationalBasisFunction) where {T} = RationalBasisFunction{T}(r.pole)

(r::RationalBasisFunction)(x) = 1/(x-pole(r))

basisfunction(dict::RationalFunctions{T}, idx) where {T} = RationalBasisFunction{T}(pole(dict, idx))
