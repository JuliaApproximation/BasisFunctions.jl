
export Rationals,
    RationalBasisFunction

"A basis of functions of the form `1/(z-z_j)` where `z_j` are the poles."
struct Rationals{T} <: Dictionary{T,T}
    poles   :: Vector{T}
end

Rationals(n::Int) = Rationals{Float64}(n)

Rationals{T}(n::Int) where {T} = Rationals(rand(T, n))

name(dict::Rationals) = "Rational functions"

size(dict::Rationals) = (length(dict.poles),)

pole(dict::Rationals, idx) = dict.poles[idx]

unsafe_eval_element(dict::Rationals, idx::Int, x) = 1/(x-dict.poles[idx])

support(dict::Rationals{T}) where {T} = DomainSets.FullSpace{T}()

function unsafe_eval_element_derivative(dict::Rationals, idx, x, order)
    @assert order == 1
    -1/(x-dict.poles[idx])^2
end

function similar(dict::Rationals, ::Type{T}, n::Int) where {T}
    @assert n == length(dict)
    Rationals{T}(dict.poles)
end


struct RationalBasisFunction{T} <: Polynomial{T}
    pole    ::  T
end

pole(r::RationalBasisFunction) = r.pole

RationalBasisFunction{T}(r::RationalBasisFunction) where {T} = RationalBasisFunction{T}(r.pole)

name(r::RationalBasisFunction) = _name(r, domaintype(r))
_name(r::RationalBasisFunction, ::Type{<:Real}) = "1/(x-$(pole(r)))  (rational function)"
_name(r::RationalBasisFunction, ::Type{<:Complex}) = "1/(z-$(pole(r)))  (rational function)"

convert(::Type{TypedFunction{T,T}}, r::RationalBasisFunction) where {T} = RationalBasisFunction{T}(r.pole)

(r::RationalBasisFunction)(x) = 1/(x-pole(r))

basisfunction(dict::Rationals{T}, idx) where {T} = RationalBasisFunction{T}(pole(dict, idx))
