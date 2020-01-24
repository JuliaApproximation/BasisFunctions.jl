
export Rationals,
    RationalBasisFunction

"A basis of functions of the form `1/(z-z_j)` where `z_j` are the poles."
struct Rationals{T} <: Dictionary{T,T}
    poles   :: Vector{T}
end

Rationals(n::Int) = Rationals{Float64}(n)

Rationals{T}(n::Int) where {T} = Rationals(rand(T, n))

size(dict::Rationals) = (length(dict.poles),)

pole(dict::Rationals, idx) = dict.poles[idx]

unsafe_eval_element(dict::Rationals, idx::Int, x) = 1/(x-dict.poles[idx])

support(dict::Rationals{T}) where {T} = DomainSets.FullSpace{T}()

unsafe_eval_element_derivative(dict::Rationals, idx, x) = -1/(x-dict.poles[idx])^2

function similar(dict::Rationals, ::Type{T}, n::Int) where {T}
    @assert n == length(dict)
    Rationals{T}(dict.poles)
end


struct RationalBasisFunction{T} <: Polynomial{T}
    pole    ::  T
end

pole(r::RationalBasisFunction) = r.pole

RationalBasisFunction{T}(r::RationalBasisFunction) where {T} = RationalBasisFunction{T}(r.pole)

convert(::Type{TypedFunction{T,T}}, r::RationalBasisFunction) where {T} = RationalBasisFunction{T}(r.pole)

(r::RationalBasisFunction)(x) = 1/(x-pole(r))

basisfunction(dict::Rationals{T}, idx) where {T} = RationalBasisFunction{T}(pole(dict, idx))
